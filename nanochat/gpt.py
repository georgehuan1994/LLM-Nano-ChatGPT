"""
这个文件定义了项目里的 GPT 神经网络本体。

如果把整个训练流程拆开看：
- tokenizer 负责把字符串变成 token id
- dataloader 负责把 token id 组织成训练批次
- 本文件里的 GPT 则负责真正执行 “神经网络计算”

也就是说：

    token ids
        -> embedding 向量
        -> 多层 Transformer block
        -> logits（对词表里每个 token 的打分）
        -> loss（训练时）

这是整个项目最核心的模型定义文件。

这份实现相较于 “教科书版 GPT” 有一些工程/研究改造，主要包括：
1. 使用 rotary embeddings，而不是单独的绝对位置 embedding
2. 对 Q/K 做 norm
3. token embedding 与 lm_head 不共享权重
4. MLP 使用 relu^2 激活
5. token embedding 后立刻做 norm
6. RMSNorm 不带可学习参数
7. 线性层不带 bias
8. 支持 GQA(Grouped / Group-Query Attention) 以提高推理效率
9. 集成 Flash Attention 3 / SDPA

对初学者来说，最重要的主线不是记住所有花样，
而是先搞清楚下面这四层结构：

1. Embedding：把 token id 变成向量
2. Attention：让每个位置“看见前文中和自己相关的位置”
3. MLP：对每个位置自己的表示做非线性变换
4. LM Head：把隐藏向量投影回“词表打分”
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# 项目自己的 attention 后端封装：
# - 如果硬件支持，就优先用 Flash Attention 3
# - 否则退回到 PyTorch 自带的 SDPA
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    # sequence_len:
    #   模型训练/推理时允许看到的最大上下文长度（以 token 为单位）
    sequence_len: int = 2048
    # vocab_size:
    #   tokenizer 词表大小。决定 embedding 输入空间和 lm_head 输出空间大小。
    vocab_size: int = 32768
    # n_layer:
    #   Transformer block 堆多少层，也就是模型有多“深”。
    n_layer: int = 12
    # n_head:
    #   query 头的数量。多头注意力会把隐藏维度切成多个头并行处理。
    n_head: int = 6
    # n_kv_head:
    #   key/value 头的数量。若小于 n_head，就形成 GQA。
    n_kv_head: int = 6
    # n_embd:
    #   每个 token 在网络内部对应的隐藏向量长度。
    n_embd: int = 768
    # window_pattern:
    #   用字符串指定每层的注意力窗口模式。
    #
    # 其中：
    # - L = long / full context，全上下文注意力
    # - S = short / sliding window，较短滑动窗口
    #
    # 例子：
    # - "L"    ：所有层都看完整上下文
    # - "SL"   ：长短交替
    # - "SSL"  ：两层短窗口 + 一层长窗口循环
    #
    # 最后一层总会被强制设成 L，让最终输出至少有一次完整上下文汇总。
    window_pattern: str = "SSSL"


def norm(x):
    # 这里统一使用 RMSNorm。
    #
    # 对初学者可以先把 norm 理解成：
    # “把向量的数值尺度整理得更稳定，方便后续层继续处理”
    #
    # 它不会改变张量形状，只会调整数值分布。
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
    """
    对 `nn.Linear` 的一个轻量改造版本。

    直觉上：
    - 参数本体（weight）尽量保留更高精度，方便优化器更新
    - 真正做矩阵乘法时，让 weight 临时转成和输入激活相同的 dtype

    这样可以在不改网络结构的前提下，同时兼顾：
    1. 参数更新的稳定性
    2. 前向/反向计算的效率
    """
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """
    判断某一层是否启用 Value Embedding。

    当前策略是交替启用，并保证最后一层一定启用。
    这是一个架构实验设计，不是标准 GPT 必备组件。
    """
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    """
    对 attention 里的 q / k 应用 Rotary Position Embedding。

    Rotary 的核心思想不是“给每个位置额外加一个位置向量”，
    而是通过旋转不同维度，让 q / k 自带相对位置信息。

    这里输入 x 的形状是：
        (B, T, H, D)
    分别表示：
    - B: batch size
    - T: 序列长度
    - H: 注意力头数
    - D: 每个头的通道维度
    """
    assert x.ndim == 4  # 多头注意力张量
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # 把最后一个维度切成前后两半，做二维平面旋转
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层。

    这是 GPT 最核心的模块之一。

    它解决的问题是：
    “当前 token 在生成/理解时，应该重点参考前面哪些 token？”

    “因果(causal)” 的意思是：
    当前时刻只能看自己和前文，不能偷看未来 token。

    这个模块的主线流程是：
    1. 把输入向量投影成 q / k / v
    2. 给 q / k 注入位置信息（rotary）
    3. 计算注意力，把相关位置的信息聚合回来
    4. 再投影回残差流维度
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # 这四个线性层就是注意力模块里的核心投影：
        # c_q: 输入 -> query
        # c_k: 输入 -> key
        # c_v: 输入 -> value
        # c_proj: 多头聚合后的输出 -> 再投影回残差流维度
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        # ve_gate 是一个额外实验设计，用来控制 Value Embedding 注入强度。
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # 第一步：把输入 hidden states 投影成 q / k / v。
        #
        # 对初学者可以这样记：
        # - q(query)：当前这个位置 “想找什么”
        # - k(key)  ：每个位置 “我这里有什么标签可供匹配”
        # - v(value)：每个位置 “我真正携带的信息内容”
        #
        # 最终张量形状会变成：
        #   (B, T, H, D)
        # 即 batch, time, head, per-head-dim
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 如果启用了 Value Embedding，就把额外的 value-like 信息注入 v。
        # 这里 gate 是输入相关的，也就是“当前输入不同，注入强度也不同”。
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # 取值范围约在 (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # 给 q / k 注入 rotary 位置信息。
        # 注意：v 一般不加 rotary。
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm：让 q/k 数值尺度更稳定
        q = q * 1.2  # 稍微拉尖注意力分布
        k = k * 1.2

        # 真正执行注意力计算。
        #
        # 数学上你可以粗略理解成：
        #   attention_weights = softmax(q @ k^T)
        #   output = attention_weights @ v
        #
        # 这里只是没有手写这个公式，而是交给更高效的 Flash Attention / SDPA 内核。
        #
        # window_size 形如 (left, right)：
        # - (-1, 0) 表示可看完整前文
        # - (N, 0)  表示只能看左边最近 N 个 token
        if kv_cache is None:
            # 训练阶段：整段序列一次性喂进来，做 causal attention
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # 推理阶段：使用 KV cache，避免每生成一个 token 都把全部前文重算一遍。
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # 只有最后一层处理完当前 token 后，才推进缓存位置。
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # 把多头输出重新拼回一个完整隐藏向量，再投影回残差流维度。
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Transformer block 里的前馈网络（MLP）。

    如果说 attention 负责 “跨位置交换信息”，
    那 MLP 更像是 “每个位置自己在本地做更复杂的非线性变换”。

    常见直觉是：
    - attention 让 token 彼此沟通
    - MLP 让单个 token 的表示被深加工
    """
    def __init__(self, config):
        super().__init__()
        # 先升维到 4 * n_embd，再投影回 n_embd。
        # 这是一种很常见的 Transformer MLP 结构。
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        # relu^2 可以理解成项目选择的一种激活函数变体。
        # 先 relu，再平方，使较大的正激活进一步被强调。
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    一个完整的 Transformer block。

    它由两大子模块组成：
    1. Attention
    2. MLP

    并通过残差连接(residual connection)串起来。

    最粗略的结构可以记成：

        x = x + Attention(Norm(x))
        x = x + MLP(Norm(x))

    这是一种 Pre-Norm Transformer 结构。
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # 残差连接的直觉：
        # 不让新子层把旧信息完全覆盖掉，而是在旧表示基础上“增量修正”。
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        GPT 模型总装。

        这里有一个很重要的工程细节：
        这个 `__init__` 经常会运行在 meta device 上。

        也就是说：
        - 这里只适合定义网络结构、张量形状、模块关系
        - 不适合在这里真的分配大块参数数据

        所以真正的参数初始化放在 `init_weights()` 里做。
        """
        super().__init__()
        self.config = config

        # 先预计算每一层 attention 该使用什么窗口大小。
        self.window_sizes = self._compute_window_sizes(config)

        # 为了硬件效率，把词表大小 pad 到某个更整齐的倍数。
        # 注意：
        # 这只是内部计算优化，最终 logits 会切回原始 vocab_size。
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        # transformer.wte:
        #   token embedding 表，把 token id 映射成隐藏向量
        # transformer.h:
        #   多层 Transformer block 堆叠
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })

        # lm_head:
        #   把最终隐藏向量再映射回词表维度，得到每个 token 的 logits。
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # 下面这些是一些额外的可学习标量/小模块，不是最标准 GPT 的必备组件。
        #
        # resid_lambdas:
        #   控制每层残差流强度
        # x0_lambdas:
        #   控制每层重新混入初始 embedding 的强度
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # 这里只占位，真正初始化在 init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        # smear:
        #   把前一个 token 的 embedding 轻微混到当前位置里，
        #   相当于注入一点廉价的 bigram 风味信息。
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))

        # backout:
        #   在最后输出前减去中间层缓存的一部分表示，属于实验性设计。
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

        # Value embeddings:
        #   额外为部分层准备的 value embedding 表。
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        # 预先生成一大段 rotary embedding 缓存。
        # rotary 很省内存，所以这里直接多算一些，减少后续动态扩容复杂度。
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # persistent=False 表示这些 buffer 不写进 checkpoint。
        # 因为它们可以随时按配置重新算出来。
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        统一初始化整个模型。

        对初学者来说，可以先记住：
        神经网络不是“随便填点随机数”就结束了，
        初始化尺度会显著影响训练是否稳定、是否容易学起来。

        这里作者把不同部位用不同初始化策略处理：
        - embedding 一种尺度
        - lm_head 一种尺度
        - attention / MLP 里的线性层又有各自设计
        """

        # token embedding 和 lm_head 的初始化。
        #
        # wte:
        #   token id -> 向量
        # lm_head:
        #   向量 -> 词表 logits
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer block 里的大矩阵参数初始化。
        #
        # 这里使用 uniform，并控制其方差与目标 normal 初始化相匹配。
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # 每层标量参数初始化。
        # 这里不是全设成完全一样，而是给早层/深层一点不同的先验。
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # smear / backout 的小参数也要显式初始化。
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # value embedding 初始化方式与 value 投影风格保持接近。
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # gate 初始给一点小正值，让它一开始略高于完全中性状态。
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # rotary cache 也在这里重建为真实张量。
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # 允许 embedding/value embeddings 使用 COMPUTE_DTYPE 节省内存。
        # 但 fp16 是特例，因为 GradScaler 处理上需要更谨慎。
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        """
        预计算 rotary embedding 所需的 cos / sin 表。

        可以把它理解成：
        提前为“每个位置、每个旋转频率”把三角函数值算好，
        前向传播时直接查表使用。
        """
        if device is None:
            device = self.transformer.wte.weight.device
        # 生成偶数通道位置对应的频率刻度
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 生成时间位置索引
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 对每个 (位置, 通道频率) 配对计算旋转角度
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        # 补出 batch/head 维，便于后续广播
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        为每一层生成对应的 attention 窗口大小。

        返回一个列表，元素形如 `(left, right)`：
        - left : 当前 token 往左最多能看多少个位置；-1 表示不限制
        - right: 当前 token 往右最多能看多少个位置；因果模型里固定为 0

        这样就可以让不同层使用不同范围的注意力：
        - 有的层只看近邻
        - 有的层看完整上下文
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # 把模式字符映射到实际窗口
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # 向上对齐到 FA3 的 tile 粒度
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # 按 pattern 在各层上循环铺开
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # 最后一层强制使用全上下文
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        估算“每个 token 训练一次”大约要多少 FLOPs。

        对初学者可以先把 FLOPs 理解成：
        “这次计算大概要做多少次浮点运算”

        它常用于：
        - 估训练成本
        - 做 scaling laws 分析
        - 估算硬件吞吐利用率
        """
        nparams = sum(p.numel() for p in self.parameters())
        # 把不属于大矩阵乘法主干的参数排除出去，便于估算主成本。
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # attention 部分的 FLOPs 要额外考虑每层窗口大小。
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        返回更细粒度的参数量统计，供 scaling laws 分析使用。

        为什么不只返回一个 total？
        因为不同论文、不同经验规则，对“到底哪些参数该计入规模”并不完全一致。

        所以这里把：
        - embedding
        - value_embeds
        - lm_head
        - transformer_matrices
        - scalars
        分开统计，方便上层自己决定采用哪种口径。
        """
        # 各类参数分组统计；分组方式和 setup_optimizer 中的大类相呼应。
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        """
        为模型构造优化器，并按参数类型分组。

        这里不是所有参数都用同一种优化策略：
        - 大矩阵参数主要走 Muon
        - embedding / lm_head / 若干标量参数走 AdamW

        这种“按参数类型分组”的做法在大模型训练里很常见，
        因为不同参数的数值行为和训练敏感性往往不同。
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # 先按功能把参数拆成几类。
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # 对 AdamW 参数组再做一个按 d_model 缩放的学习率修正。
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # 构造 AdamW 参数组。
        param_groups = [
            # embeddings / lm_head / scalars 都走 AdamW，但超参数可各不相同
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        # 大矩阵参数按 shape 分组走 Muon。
        # 这样做有助于 Muon 内部更高效地批处理同形状矩阵。
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        GPT 的前向传播。

        这是整个模型最关键的函数。

        输入：
        - idx: token ids，形状 (B, T)
        - targets: 可选，训练时提供；若给出则函数直接返回 loss
        - kv_cache: 推理时可选，用于缓存历史 key/value

        输出：
        - 训练时：loss
        - 推理时：logits

        可以把这整个 forward 粗略记成：

            token ids
                -> embedding
                -> 多层 Transformer
                -> lm_head
                -> logits
                -> cross entropy loss（如果给了 targets）
        """
        B, T = idx.size()

        # 先取出当前序列长度需要的 rotary cos/sin。
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # 如果使用 KV cache，当前位置不再从 0 开始，而要接着缓存中的历史位置往后算。
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # 第一步：token embedding。
        #
        # idx 里每个元素只是一个离散 token id。
        # wte 会把每个 id 查表变成长度为 n_embd 的向量。
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # smear：
        # 把前一个 token 的 embedding 轻微混进当前位置，
        # 相当于给模型额外补一点“相邻 token 局部搭配”的廉价先验。
        if kv_cache is None:
            # 训练时整段序列都在，所以可以直接用切片处理位置 1..T-1。
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # 推理时可能是一个 token 一个 token 地往后生成，所以要借助缓存拿到上一个位置的 embedding。
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # prefill：一次喂很多 token，处理方式和训练基本一致
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # decode：一次只来一个 token，就从缓存读前一个 embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # 开始穿过 Transformer 主干。
        #
        # x0 保存最初 embedding，后面某些层会把它按比例重新混回来。
        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            # 在进入每层 block 前，先做两种线性混合：
            # 1. 当前残差流 x 乘以 resid_lambdas[i]
            # 2. 初始 embedding x0 乘以 x0_lambdas[i]
            #
            # 可以理解成：每层都可学习地决定“保留多少当前表示”和“重新拉回多少初始输入信息”。
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x

        # backout：
        # 在最终投影到 logits 前，减去一部分中间层表示。
        # 这是实验性结构，可以把它粗略理解成一种“移除部分低层特征残留”的尝试。
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # lm_head：把隐藏向量映射回词表维度，得到 logits。
        #
        # logits 的含义不是概率，而是“对每个 token 的原始打分”。
        # 后面 softmax 才会把它们变成概率分布。
        softcap = 15 # 用平滑方式限制 logits 过大，提升数值稳定性
        logits = self.lm_head(x) # 形状：(B, T, padded_vocab_size)
        logits = logits[..., :self.config.vocab_size] # 去掉为了硬件对齐而 pad 出来的多余词表部分
        logits = logits.float() # loss 计算前切到 fp32，更稳
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            # 训练模式：
            # 给定 targets，直接算交叉熵 loss。
            #
            # 这里的交叉熵就是“语言模型训练的标准目标函数”：
            # 模型在每个位置输出一个对全词表的分布，
            # 然后看真实下一个 token 的概率是否高。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理模式：
            # 不算 loss，只把 logits 返回给上层采样器。
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        一个最朴素的自回归生成器。

        这里假设：
        - batch size = 1
        - 输入/输出 token 使用简单 Python 列表和整数

        自回归生成的主线是：

        1. 把当前已有 token 序列送进模型
        2. 只取最后一个位置的 logits
        3. 从这个分布里选出下一个 token
        4. 把新 token 拼到序列后面
        5. 重复上述过程
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # 补上 batch 维度
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # 只关心最后一个位置，因为我们要预测“下一个 token”
            if top_k is not None and top_k > 0:
                # top-k 截断：只保留得分最高的前 k 个候选，其他全设成 -Inf
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                # temperature > 0 时做采样：
                # - temperature 越高，分布越平，生成更发散
                # - temperature 越低，分布越尖，生成更保守
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                # temperature = 0 时直接贪心取 argmax
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
