"""
这个脚本负责训练项目里的 base model

这里训练的还不是聊天助手，而是一个标准的自回归语言模型：
给它前面的 token，让它预测下一个 token。

从学习角度看，这个脚本主要做了五件事：

1. 解析训练配置，例如模型深度、上下文长度、batch size。
2. 加载 tokenizer，并据此确定词表大小。
3. 按配置构建 GPT 模型，初始化参数，准备优化器。
4. 持续从数据集中取出 token 序列，执行前向传播、反向传播和参数更新。
5. 训练过程中周期性做 BPB / CORE / sample 等评估，并保存 checkpoint。

运行方式：

    python -m scripts.base_train

或分布式训练：

    torchrun --nproc_per_node=8 -m scripts.base_train

如果只在 CPU / MacBook 上体验流程，需要把模型和 batch 调得很小，例如：

    python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
# 让 PyTorch 的 CUDA 显存分配器使用 “可扩展段”，
# 通俗讲：减少长训练里因为反复分配/释放造成的 “显存碎片化”，降低 OOM 概率。
# 必须在 import torch 之前设置才会生效。
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc                                  # Python 垃圾回收，后面训练循环里会手动控制
import json                                # 打印模型配置时用
import time                                # 测每个 step 耗时
import math                                # 计算学习率/调度器需要 cos、log2 等
import argparse                            # 解析命令行参数
from dataclasses import asdict             # 把 GPTConfig 转成 dict 方便打印/保存
from contextlib import contextmanager      # 用来写 disable_fp8(model) 这样的 with 语句

import wandb                               # 训练可视化日志（Weights & Biases），可选
import torch                               # PyTorch 主体
import torch.distributed as dist           # 分布式训练（DDP）所需通信原语

# 下面这些 import 都是 nanochat 项目自己写的模块。
# 学习时可以把它们当成 “工具箱”：基础脚本只关心调度，具体实现在对应模块里。

from nanochat.gpt import GPT, GPTConfig, Linear                     # 模型主体（核心，建议精读 nanochat/gpt.py）
from nanochat.dataloader import (                                   # 训练/验证数据迭代器（生成 (x, y) batch）
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.common import (                                       # 通用工具
    compute_init,           # 初始化设备、DDP 进程组
    compute_cleanup,        # 训练结束时清理资源
    print0,                 # 只让 rank 0 打印，避免多卡时刷屏
    DummyWandb,             # 当不用 wandb 时的 “假” logger
    print_banner,           # 打印项目 banner
    get_base_dir,           # 项目数据/输出根目录
    autodetect_device_type, # 自动检测 cuda/mps/cpu
    get_peak_flops,         # 查询 GPU 理论峰值算力，用于算 MFU
    COMPUTE_DTYPE,          # 训练用的数值精度（bf16 / fp16 / fp32）
    COMPUTE_DTYPE_REASON,   # 选这个精度的原因（用于打印解释）
    is_ddp_initialized,     # 检查 DDP 是否已初始化
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes        # 加载 tokenizer + 每个 token 对应的字节数（算 BPB 用）
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint  # 保存/加载训练快照
from nanochat.loss_eval import evaluate_bpb                          # 计算验证集 bits-per-byte
from nanochat.engine import Engine                                   # 推理引擎，用于训练中“采样几个例子看看”
from nanochat.flash_attention import HAS_FA3                         # 当前环境是否安装了 Flash Attention 3
from scripts.base_eval import evaluate_core                          # CORE 评估（多任务能力评估）
print_banner()                                                       # 仅打印项目 logo，不影响逻辑

# -----------------------------------------------------------------------------
# 命令行参数
#
# 1. 运行环境参数：用什么设备、是否记录日志
# 2. 模型结构参数：几层、隐藏维度多大、上下文多长
# 3. 优化参数：batch size、多大学习率、训练多少步
# 4. 评估参数：多久评一次、评哪些指标
parser = argparse.ArgumentParser(description="Pretrain base model")
# ── Logging（日志相关）────────────────────────────────────────────────
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# ── Runtime（运行时设备）─────────────────────────────────────────────
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--compile", type=int, default=1, help="compile model with torch.compile on CUDA (0 disables; useful for driver/Inductor issues)")
# ── FP8 training（FP8 低精度训练，仅在 H100/Hopper 及以上 GPU 上有意义）──
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# ── Model architecture（模型结构）────────────────────────────────────
# depth = Transformer block 的层数；越深通常表达能力越强，但也越贵
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
# aspect_ratio 用来由 depth 反推 hidden dim：model_dim ≈ depth * aspect_ratio
# 这是 nanochat 项目的一种简化 “宽深比” 约定
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
# 每个注意力 head 的通道数；FA3 偏好 64 / 128 这类规整尺寸
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
# 上下文长度：模型一次最多能 “看到” 多少个连续 token
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# 滑动窗口注意力模式：每层用全注意力(L)还是半窗口(S)，可以省显存/加速
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# ── Training horizon（训练时长，三选一）─────────────────────────────────
# 三种方式按优先级： num_iterations > target_flops > target_param_data_ratio
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
# Chinchilla 经验：每个参数大约配 20 个 token；这里默认 12 是“偏少 token”的省钱配置
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# ── Optimization（优化相关）─────────────────────────────────────────────
# device_batch_size：单卡一次前向看多少条样本（micro-batch）。OOM 时优先把它调小
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
# total_batch_size：一次参数更新前累计的 token 总数（跨所有卡 + grad accumulation 后）
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
# nanochat 用的是混合优化器：embedding/unembedding/scalar 走 Adam，矩阵参数走 Muon。所以学习率分四套
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
# warmup：开头 N 步把学习率从小线性升到目标值，避免初期梯度爆炸
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
# warmdown：末段把学习率线性降到 final_lr_frac，让训练平稳收尾
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
# 断点续训：从某个保存过的 step 继续训练
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# ── Evaluation（评估相关）───────────────────────────────────────────────
# BPB = bits per byte，每个字节平均要花多少 bit 才能编码（越小越好）
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
# CORE = 一组开放学术任务的综合得分（zero-shot 能力）
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
# 训练中偶尔生成几句话给人类直观看模型 “学没学到点什么”
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
# 默认只在训练结束时存 checkpoint；想多存一点就把它调成正数
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# ── Output（输出目录命名）──────────────────────────────────────────────
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # 把 argparse 解析后的所有参数原样存一份，后面写日志/checkpoint 元数据要用
# -----------------------------------------------------------------------------
# 计算/设备初始化 + 日志初始化
#
# 这一步的目标是：
# 1. 确定当前用 CPU / CUDA / MPS 哪种设备；
# 2. 如果是多卡训练，初始化 DDP；
# 3. 准备好 wandb 或 dummy logger。

# 1) 设备类型：cuda / cpu / mps（苹果芯片）
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# 2) compute_init 内部会：检测是否有 torchrun 启动的环境变量、初始化 DDP 进程组、绑定当前进程到对应 GPU。
#    返回值：
#    - ddp:           是否启用了分布式数据并行（多卡）
#    - ddp_rank:      全局编号（0 ~ world_size-1），只有 rank 0 是“主进程”
#    - ddp_local_rank:本机内编号（多机训练时和 rank 不同）
#    - ddp_world_size:总共多少个进程（≈ 总 GPU 数）
#    - device:        当前进程实际跑在哪个 device 上，比如 cuda:0
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # 只有 rank 0 负责日志、保存 checkpoint、打印主要信息等
# torch.cuda.synchronize 会“等当前设备所有 kernel 都跑完才返回”，用于精准计时
# CPU/MPS 上不需要这种同步，所以直接给个空函数
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
# 训练结束打印峰值显存用；CPU/MPS 给 0
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    # gpu_peak_flops：GPU 在 BF16 下的理论峰值算力（FLOPS）。
    # 后面用它和实际吞吐对比，得到 MFU（Model FLOPs Utilization，模型算力利用率），
    # MFU 越高代表越“没浪费 GPU”。
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # CPU/MPS 上算 MFU 没意义，置为 ∞ 让 mfu = 0%
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb 日志初始化。
# 如果 run=dummy，或者当前不是主进程，就退化成一个空 logger，避免重复上传日志。
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention 状态检查。
#
# attention 本身是 Transformer 最核心、也最贵的计算之一。
# Flash Attention 是更高效的 attention 实现。
# 如果能用上 FA3，训练速度和显存效率都会更好。

# A800 (sm80) → FA3 不加载 → SDPA 回退 → 自动覆盖为 L
# 4090 (sm89) → 同上 → 自动覆盖为 L
# H100/H200 (sm90) + bf16 → FA3 可用 → 保留传入的 --window-pattern（比如 SSSL）
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        # SDPA 在带任意 mask 的滑窗下会退化成 O(N^2) 朴素实现，
        # A800 等非 Hopper 卡上跑 SSSL 这种模式会非常慢，自动改成 L 全注意力。
        print0(f"NOTE: Overriding --window-pattern from '{args.window_pattern}' to 'L' because SDPA fallback is in use (non-Hopper GPU, e.g. A800).")
        args.window_pattern = "L"
    print0("!" * 80)

# -----------------------------------------------------------------------------
# 加载 tokenizer 及其配套元数据
#
# 为什么训练模型前必须先有 tokenizer？
# 因为模型的输入/输出维度都和词表大小直接相关。
# 模型并不直接看字符串，而是看 token id；
# 同时最后一层 lm_head 也要输出 “对整个词表中每个 token 的打分(logits)”。
# tokenizer：把字符串切成 token id 的 “分词器”。
# get_tokenizer() 内部会去之前 tok_train.py 训练好的目录里加载它。
tokenizer = get_tokenizer()
# token_bytes：长度 = vocab_size 的张量，第 i 个元素表示 token id=i 解码后占多少字节。
# 计算 BPB 时，用 “总 cross-entropy(loss, 单位 nat) → 总 bit → 除以总 bytes”，
# 而每个 token 对应几个字节，正是来自这个张量。
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# 构建模型
#
# 这里最关键的概念有三个：
# 1. `depth`：Transformer block 堆多少层，也就是网络有多深
# 2. `n_embd`：每个 token 在网络内部表示成多长的向量，也就是隐藏维度
# 3. `n_head`：注意力头的数量，表示把注意力机制分成多少个并行子空间去看信息
#
# 注意：
# `base_train.py` 负责“决定模型规模并训练它”；
# 真正的神经网络内部实现细节（embedding、attention、MLP、lm_head 等）
# 定义在 `nanochat/gpt.py` 里。

def build_model_meta(depth) -> GPT:
    """
    先在 meta device 上“空壳化”构建模型。

    meta device 可以理解成：
    只创建张量的形状和 dtype，不真正分配实际数据内存。

    这样做的好处是：
    1. 可以先安全地推导模型结构和参数规模；
    2. 避免一上来就占满显存；
    3. 方便后续做更可控的初始化。
    """
    # 这里先根据 depth 推出模型隐藏维度 model_dim。
    #
    # 设计上作者让：
    #   model_dim = depth * aspect_ratio
    #
    # 直觉上可以理解成：
    # 模型越深，通常也配套给它更宽的表示空间。
    #
    # 然后再把 model_dim 向上取整到 head_dim 的整数倍，
    # 这样可以保证：
    #   model_dim % head_dim == 0
    #
    # 因为多头注意力里，每个 head 的通道数 = model_dim / n_head，
    # 不整除的话就没法自然地切成多个头。
    #
    # 同时 FA3 也偏好更规整的 head_dim。
    base_dim = depth * args.aspect_ratio
    # ((x + a - 1) // a) * a 是“把 x 向上取整到 a 的整数倍”的常用写法
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim  # 已保证整除

    # 这里构造的是 GPTConfig，不是实际训练数据。
    # 你可以把它看成 “神经网络蓝图”：
    # - sequence_len: 最长上下文长度（每条样本最多几个 token）
    # - vocab_size:   词表大小，决定 embedding/lm_head 的输入输出规模
    # - n_layer:      堆多少层 Transformer block
    # - n_head:       Q 的多头数量
    # - n_kv_head:    K/V 的头数；= n_head 是标准 MHA，< n_head 就是 GQA（节省 KV 显存）
    # - n_embd:       每个 token 的隐藏向量维度
    # - window_pattern: 每层注意力窗口模式（L = 全窗口，S = 半窗口/省显存）
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    # `with torch.device("meta")`：进入这个 with 后，所有新建张量/参数都落在 meta device，
    # 不会真分配显存，仅记录 shape/dtype。
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# 真正建模分成三步：
# 1. 先在 meta device 上搭出 “网络结构空壳”
# 2. 再把这些参数张量搬到目标设备上，但此时里面还是未初始化的垃圾值
# 3. 最后显式调用 init_weights() 做可控初始化
#
# 这种写法比 “构造函数里直接偷偷初始化一切” 更适合教学，也更省显存。
model = build_model_meta(args.depth) # 1) 只构建结构，不分配真实参数数据
model_config = model.config
model_config_kwargs = asdict(model_config)  # dataclass → dict，方便 json 打印 / 存 checkpoint
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
# to_empty：把 meta 张量替换成 device 上“同形状但未初始化”的真实张量。
# 注意不能用 .to(device)：to() 会尝试拷贝数据，但 meta 上没有数据可拷。
model.to_empty(device=device)        # 2) 在目标设备上分配参数存储，但尚未初始化
model.init_weights()                 # 3) 按各模块自定义的规则做可控初始化（见 nanochat/gpt.py）

# 如果是断点续训，就用 checkpoint 里的参数覆盖刚初始化好的随机参数。
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # 例如 d12 表示 depth=12 的模型
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    # 同时拿回模型权重、优化器状态（动量/二阶矩等）、训练元数据（步数/loss/dataloader 进度）
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    # assign=True：直接 “接管” 张量，避免再多拷一份占显存
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # 拷贝完赶紧释放，否则相当于显存里多了一份模型权重

# -----------------------------------------------------------------------------
# FP8 训练相关初始化
#
# 这是一个更偏工程优化的部分。
# 对初学者来说可以先知道：
# - 不开 FP8 也完全可以理解主流程
# - 开 FP8 是为了在支持的 GPU 上进一步提升吞吐/省显存
# - 它不会改变“语言模型训练”的基本数学逻辑，只是改了数值表示方式

# 如果启用 --fp8，就把一部分 Linear 层替换成 Float8 版本。
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # 不是所有线性层都适合转成 FP8。
        # 这里做筛选，主要原因是：
        # 1. 某些硬件要求维度能被 16 整除
        # 2. 太小的层转 FP8 收益不大，复杂度反而更高
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# 这个上下文管理器用于 “临时关闭 FP8”。
# 原因是训练时可以为速度启用 FP8，但评估时通常更希望数值稳定、口径一致，
# 所以会临时切回 BF16/普通 Linear。
@contextmanager
def disable_fp8(model):
    """
    临时把 Float8Linear 换回普通 Linear，供评估阶段使用。

    这里不是 “修改训练逻辑”，而是做一个短暂的模块替换：
    进入 with 时替换，退出 with 时恢复。
    """
    import torch.nn as nn

    # 第一步：扫一遍模型，记下所有 Float8 模块挂在哪个父模块上、属性名是什么。
    # 注意 named_modules() 里 name 是“点分路径”，比如 "transformer.h.0.attn.c_q"。
    fp8_locations = []  # 形如 (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                # 拆成 "transformer.h.0.attn" 和 "c_q"
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                # 顶层模块，没有 parent_name
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # 没有 FP8 模块就什么都不做，with 块照常执行
        return

    # 第二步：原位替换为普通 Linear。
    # 注意 weight/bias 是“共享引用”，没有真正拷贝数据，所以替换非常便宜。
    # 使用 device="meta" 是为了不再额外申请一份显存（外壳本身就不放数据）。
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # 用 meta，避免占用显存
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # 共享同一个 nn.Parameter，不拷贝
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)  # 真正“替换”掉父模块的子模块

    try:
        yield  # 进入 with 块，执行评估等代码
    finally:
        # 第三步：无论 with 块里是否抛异常，都把 FP8 模块装回去
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# 编译模型
#
# torch.compile 可以把 Python 级的 eager 执行图优化成更高效的执行形式。
# 对初学者来说，可以先把它理解成：
# “不改模型数学定义，只是尽量把执行变快”。

orig_model = model # 保留未 compile 的原始模型，便于保存、推理和某些变长输入场景
if device_type == "cuda" and args.compile:
    # torch.compile：把 PyTorch 的 eager 执行图编译成更高效的形式（fuse kernel、减少 Python 开销等）。
    # dynamic=False 告诉 compiler “输入形状是固定的”，可以做更激进的优化。
    # 训练时我们的 (batch, seq_len) 永远固定，所以放心开。
    model = torch.compile(model, dynamic=False)
else:
    print0(f"Skipping torch.compile on {device_type}.")

# -----------------------------------------------------------------------------
# 根据 scaling laws / muP 风格经验规则，自动估算更合理的训练预算和超参数
#
# 这部分对初学者最容易看晕，所以先抓住主线：
# 1. 模型参数量越大，通常也应该训练更多 token
# 2. token 预算确定后，可以估一个比较合理的 batch size
# 3. batch size 变化后，学习率和 weight decay 往往也要跟着缩放

# 先统计模型参数量。
# 参数量是理解模型规模最基础的指标之一。
# num_scaling_params() 返回一个 dict，按类别拆分（embedding / lm_head / transformer 矩阵 / 标量等）
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
# estimate_flops()：估算 “处理一个 token” 需要多少 FLOPs（浮点运算数）。
# 后面会用 FLOPs/token × token 总数 来估总训练算力开销，用来算 MFU 和 ETA。
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) 先根据 scaling laws 估算 “应该训练多少 token”
#
# 直觉上：
# 小模型喂太多数据可能收益变小，大模型喂太少数据又学不满。
# 所以常见经验做法是维持一个 Tokens : Params 的目标比例。
# 比如 Chinchilla 论文经验值是 20:1，本项目默认 12:1（更省时间）。
def get_scaling_params(m):
    # 这里作者发现：
    # 用 transformer_matrices + lm_head 作为 scaling params，
    # 在经验拟合上更 “干净”（不算 embedding、layernorm 等小头）。
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # 估计这次训练总共应该看多少 token

# 这里选一个 d12 模型作为 “参考尺度”。
# 可以理解成：很多经验超参数最初是在这个参考模型上调出来的，
# 然后再按 muP/scaling rules 外推到更深的模型。
# muP = Maximal Update Parametrization，一种让“在小模型上调好的超参直接用在大模型”的参数化方式。
d12_ref = build_model_meta(12) # 只构建结构，不分配真实参数
# D_REF：参考模型在“同样的 Tokens:Params 比例”下应训练的总 token 数
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)  # 经验值，实测得来
B_REF = 2**19 # ≈ 524,288 tokens，d12 上实测的“最优 batch size”

# 2) 再根据 token 预算估 batch size
#
# batch size 可以理解成：
# 一次参数更新之前，总共看多少 token。
# 太小训练噪声大，太大又可能不划算或占太多显存。
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    # 经验幂律：B ≈ B_ref * (D / D_ref)^0.383
    # 数据量越大，batch size 也按一定指数缩放；指数 0.383 是经验拟合得来
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    # 截断到最近的 2 的幂；GPU 上 2 的幂尺寸通常更对齐、更高效
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) batch size 变了，学习率通常也要跟着缩放。
#
# 很粗的直觉：
# 一次更新里看得更多，梯度平均起来更稳，学习率可以适当放大一点。
# - SGD 经验：LR ∝ B（线性缩放）
# - AdamW/Muon 经验：LR ∝ √B（平方根缩放，更保守）
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) 最后再按相同思路缩放 weight decay。
#
# 这部分完全是 “经验/论文启发下的工程规则”，
# 不是 Transformer 数学定义本身的一部分。
# 直觉：训练 token 越多，weight decay 越小（避免过度收缩）；batch 越大，weight decay 越大。
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# 构建优化器
#
# 这里用的是混合优化器思路：
# - 大量矩阵参数（例如注意力和 MLP 里的线性层）走 Muon
# - embedding、lm_head、一些标量参数走 AdamW
#
# 为什么要混合？
# Muon 是一种较新的 “正交化梯度” 优化器，对大权重矩阵更有效；
# 但 embedding / lm_head / scalar 这类参数结构不是 “方阵”，用传统 Adam 更稳。
#
# 对初学者可以先把优化器理解成：
# “根据 loss 反向传播得到梯度后，真正负责更新参数的规则”。
optimizer = model.setup_optimizer(
    # AdamW hyperparameters（用于 embedding / lm_head / 标量）
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters（用于矩阵参数）
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    # 优化器里有动量/二阶矩等 “状态”，断点续训时必须一起恢复，否则训练曲线会错乱
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# FP16 下的 GradScaler
#
# FP16 数值范围更小（最大约 65504），loss 缩放后梯度容易下溢成 0。
# GradScaler 的作用是先把 loss 放大 N 倍再反传，让小梯度别变 0；
# 在更新参数前再把梯度缩回 1/N，等价于没放大过。
# BF16 数值范围和 FP32 一样大（只是精度低），所以不需要 scaler。
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# 构建 train / val dataloader
#
# 这里的数据已经不是原始字符串，而是会被 tokenizer 动态编码后，
# 组织成适合语言模型训练的 `(x, y)`：
#
# - x: 当前看到的 token 序列   shape = (device_batch_size, max_seq_len)
# - y: 右移一位后的目标 token 序列  shape 同上
#
# 也就是说，模型学的是：
# “看到 x 的前缀后，y 这个位置下一个 token 应该是什么”
#
# `bos_bestfit` 的含义：
#   bos     = 每条文档前会插一个 <|bos|>（beginning of sequence）token
#   bestfit = 用“箱子打包(bin packing)”策略把不等长文档打包到固定 seq_len，
#             尽量塞满，减少 padding 浪费。
# `with_state` 版本会额外维护 dataloader 的进度（epoch/位置），用于断点续训。
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device=device, resume_state_dict=dataloader_resume_state_dict,
)
# 验证集 loader 用 lambda 包起来：每次评估时 build 一个新的，
# 这样每次评估都从 val 数据集开头重新开始，结果可比。
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
)
# 提前取第一批数据。后面的循环里采用“先用 (x,y)，再 next() 拿下一个”的“流水线”模式，
# 这样可以让 GPU 计算和数据准备尽量重叠。
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# 计算训练步数，并定义各种 scheduler
#
# 这里的 scheduler 可以理解成：
# “随着训练进行，动态调整某些超参数的规则”。
# 最常见的是学习率调度器。

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# 学习率调度：
# 1. 开始先 warmup，避免刚开始更新太猛
# 2. 中间保持常数（保持目标 LR 不变，让模型主要在这一段学习）
# 3. 结尾 warmdown，平滑收尾（线性降到 final_lr_frac 倍）
#
# 返回的是 “学习率倍率(multiplier)”，最终 LR = initial_lr × lrm。
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        # warmup 阶段：从 1/warmup 线性升到 1.0
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # 中间稳态：始终保持 1.0
        return 1.0
    else:
        # warmdown 阶段：从 1.0 线性降到 final_lr_frac
        # progress 从 1 → 0；最终倍率 = progress*1 + (1-progress)*final_lr_frac
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Muon 的 momentum 也做一个随训练进度变化的调度。
# 直觉：
#  - 训练初期 (前 400 步)：momentum 从 0.85 升到 0.97（先“试探”再“坚定”）
#  - 训练中期：保持 0.97（常用经验值）
#  - warmdown 阶段：再降到 0.90（接近收敛时减小动量，避免冲过头）
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# weight decay 也不是常数，而是随训练进度余弦衰减到 0。
# 即：训练越往后，正则越弱，让模型在尾段更专注于拟合数据本身。
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# 训练主循环
#
# 这里是整个脚本最核心的部分。
# 你可以把每次 step 的生命周期理解成：
#
# 1. 偶尔做评估 / 采样 / 存 checkpoint
# 2. 取一个或多个 micro-batch 做前向传播
# 3. 计算 loss
# 4. 反向传播，得到梯度
# 5. 优化器根据梯度更新参数
# 6. 进入下一步

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0                       # 当前优化步数（每完成一次 optimizer.step() 加 1）
    val_bpb = None                 # 最近一次验证集 BPB；首次评估前是 None
    min_val_bpb = float("inf")     # 训练全程见过的最小 val BPB（越小越好）
    smooth_train_loss = 0          # 训练 loss 的指数滑动平均（EMA），让打印曲线更平稳
    total_training_time = 0        # 训练总 wall-clock 时间（秒），不含初始化和前 10 个 warmup step
else:
    # 断点续训：从 checkpoint 元数据里恢复以上所有状态
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# 计算梯度累积步数。
#
# 这是初学者非常值得理解的点：
# 如果单卡一次塞不下那么大的 total batch size，
# 就把一次“大更新”拆成多个 micro-step。
#
# 每个 micro-step:
# - 前向一次
# - 反向一次
# - 只累积梯度，不立刻更新参数
#
# 等累积够若干个 micro-step 后，再统一 optimizer.step() 一次。
#
# 公式：
#   tokens_per_fwdbwd       = device_batch_size × max_seq_len   （单卡单次前向看的 token 数）
#   world_tokens_per_fwdbwd = tokens_per_fwdbwd × world_size    （所有卡一次前向合计）
#   grad_accum_steps        = total_batch_size / world_tokens_per_fwdbwd
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
# total_batch_size 必须是 world_tokens_per_fwdbwd 的整数倍，否则没法均匀拆分
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# 正式开始训练
while True:
    # last_step：当 step 已经到达 num_iterations 时为 True。
    # 注意循环里到了 last_step 仍然会先做一轮“评估 + 保存”，再 break，
    # 这样保证最终模型一定有评估和 checkpoint。
    last_step = step == num_iterations
    # 累计已消耗的算力（用于日志）
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 周期性评估验证集 BPB。
    # BPB(bits per byte)：把模型 cross-entropy 换算成 “每个字节平均要花多少 bit”。
    # 单位归一到字节是为了和 tokenizer 无关，跨模型可比。
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()  # 关闭 dropout 等训练态行为（虽然这里没 dropout，但仍是好习惯）
        val_loader = build_val_loader()  # 重新构造 val loader，从头开始评估
        # 评估总 token 数 / 一次前向 token 数 = 评估要做多少步
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):  # 评估时切回普通 Linear，保证数值口径稳定
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()  # 切回训练态

    # 周期性做 CORE 评估。
    # CORE 更像 “任务表现” 的外部能力测试，不只是下一个 token 的 loss。
    # 比如 HellaSwag、ARC 之类的常识/推理小任务的 zero-shot 平均得分。
    #
    # 这里用 orig_model（未 compile 的版本），因为 CORE 输入长度变化更大，
    # compiled model 在变长输入上要么报错要么频繁重编译，开销大。
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # 周期性让模型生成一些样例文本，帮助人类直观感受模型学到了什么。
    # 这不是严格指标，但很有解释力。
    # 只在 master_process（rank 0）做：避免多卡同时打印重复内容。
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # 用 orig_model 避免 torch.compile 在变长输入下重编译
        for prompt in prompts:
            # prepend="<|bos|>": 在 prompt 前加 bos token，跟训练时数据格式保持一致
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                # temperature=0：贪心解码，每次选概率最高的 token；结果可复现
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # 保存 checkpoint。
    # checkpoint 可以理解成 “训练快照”：
    # 包含模型参数、优化器状态、dataloader 进度、循环状态等。
    # 触发条件：是最后一步，或者用户开了 save_every 且当前步数命中。
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # 保存未 compile 的版本，加载时不需要 torch.compile 环境
            optimizer.state_dict(),   # 优化器状态（动量/二阶矩等），续训必须
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,  # 数据集进度，续训时从同一位置继续
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,  # save_checkpoint 内部只让 rank 0 写盘
        )

    # 终止条件：训练步数达到预定上限。
    # 注意上面已经在 last_step 时跑完最后一次评估和保存了，这里直接退出循环。
    if last_step:
        break

    # -------------------------------------------------------------------------
    # 单个训练 step 的核心数值过程
    #
    # 对初学者，下面这段是最值得建立心智模型的地方：
    #
    # 前向传播：
    #   x -> model -> loss
    #
    # 反向传播：
    #   loss.backward()
    #   让每个参数都拿到“自己该往哪个方向改”的梯度
    #
    # 参数更新：
    #   optimizer.step()
    #   真正按梯度规则把参数改掉
    #
    # 整体流程（带梯度累积）：
    #   for micro_step in range(grad_accum_steps):
    #       loss = model(x, y)        # 前向，得到本 micro-batch 的平均 loss
    #       loss.backward()           # 反向，把梯度“累加”到每个 .grad 上
    #       x, y = next(train_loader) # 预取下一批数据
    #   optimizer.step()              # 累计够梯度后，统一更新一次参数
    #   model.zero_grad()             # 清空梯度，准备下一 step
    synchronize()           # 等 GPU 上一波 kernel 全部执行完，t0 才能精准代表本 step 起点
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # model(x, y) 在训练模式下返回的是交叉熵 loss（标量张量）。
        #
        # 这里内部已经做了完整 GPT 前向：
        # 1. token embedding：把 token id 映射成 n_embd 维向量
        # 2. 多层 Transformer block（attention + MLP + 残差 + 归一化）
        # 3. lm_head 输出每个位置对全词表的 logits，shape = (B, T, vocab_size)
        # 4. 与目标 y 计算 cross entropy（对每个位置 next-token 的预测）
        loss = model(x, y)
        # detach 把 loss 从计算图中“摘出来”：
        # - 用于日志（避免后续 .backward() 改变图带来的副作用）
        # - 不会影响 backward 自身（backward 用的是原 loss，不是 detach 后的副本）
        train_loss = loss.detach()

        # 因为后面会做 grad accumulation，
        # 多次 backward 相当于把每个 micro-batch 的梯度直接累加到 .grad 上。
        # 由于 model(x, y) 内部对 batch 内 loss 取的是平均，跨 micro-step 累加时
        # 梯度尺度其实和“一次性看完 grad_accum_steps × batch”是等价的，所以这里不需要再 /grad_accum_steps。
        # （注意这点和某些教程不一样——取决于 loss 内部用 mean 还是 sum）
        if scaler is not None:
            # FP16 路径：先把 loss 放大避免梯度下溢，再 backward
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # 预取下一批数据：让 “数据准备” 和 “GPU 计算” 尽量重叠。
        # 注意这一行是在 for 循环里、optimizer.step() 之前，
        # 所以最后一次 next() 拿到的 (x, y) 实际上是“下一轮 step 的第一个 micro-batch”。
        x, y, dataloader_state_dict = next(train_loader)

    # 走到这里，梯度已经累计完毕，开始真正更新参数。
    # 1) 先按当前 step 调度好 LR / momentum / weight_decay
    lrm = get_lr_multiplier(step)            # 学习率倍率
    muon_momentum = get_muon_momentum(step)  # Muon 动量
    muon_weight_decay = get_weight_decay(step)  # 衰减后的 weight decay
    # optimizer.param_groups 是“参数组”列表，每组可有自己的超参。
    # 这里遍历每组，根据它的初始 LR 缩放出本步的 LR；如果是 muon 组还要更新 momentum/wd。
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    # 2) 真正更新参数
    if scaler is not None:
        # FP16 路径下，需要先 unscale 梯度（除回放大倍数）再 step。
        # 否则 optimizer 会误以为梯度很大，错误更新。
        scaler.unscale_(optimizer)
        # 多卡 + FP16 时，可能某张卡梯度里出现 inf/nan（数值溢出）。
        # 这种情况下 scaler 会让该卡跳过这一步；但其他卡如果不跳过，参数就不同步了。
        # 所以这里用 all_reduce(MAX) 统一“是否检测到 inf/nan”的状态：只要任一卡出问题，所有卡都跳过。
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)  # 等价于 optimizer.step()，但内部会判断要不要 skip
        scaler.update()         # 根据是否溢出，动态调整放大倍数
    else:
        # BF16 / FP32 路径：直接 step 即可
        optimizer.step()
    # 清空梯度，为下一 step 做准备。
    # set_to_none=True 比 zero_() 更省（直接把 .grad 置 None，下次 backward 再分配），
    # 也能让某些优化器表现更稳定。
    model.zero_grad(set_to_none=True)
    # .item() 会把 GPU 上的标量张量同步搬到 CPU，是 GPU↔CPU 同步点（会强制等待 GPU 完成）。
    # 但我们这里反正马上要 synchronize() 计时，所以代价可以接受。
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0   # 本 step 总耗时（秒）
    # -------------------------------------------------------------------------

    # 下面主要是日志统计，不参与模型数学本身。
    ema_beta = 0.9  # EMA 衰减系数；越接近 1，平滑得越狠（“记忆”越长）
    # 指数滑动平均：smooth = β·prev + (1-β)·cur
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    # EMA 在前几步会偏向初始值 0，导致曲线“看起来超低”。
    # 这一步做去偏：除以 (1-β^(t+1)) 还原回真实尺度（Adam 偏置修正同款套路）。
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)                        # 训练吞吐：每秒处理多少 token
    flops_per_sec = num_flops_per_token * total_batch_size / dt     # 训练算力消耗：每秒多少 FLOPs
    # MFU = 实际算力消耗 / GPU 理论峰值算力。常见目标：30%~50% 已经很不错。
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        # 前 10 步通常包含 torch.compile 预热和异步 kernel 启动，时间不准，扔掉
        total_training_time += dt
    # 计算 ETA（剩余时间）：剩余步数 × 当前平均每步耗时
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    # 打印数据集进度：epoch 数、当前优先队列索引(pq)、随机组索引(rg)
    # 这三个数共同标识 dataloader “走到了哪里”，断点续训会用到
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    # wandb 日志写得稀疏一点：每 100 步一次，避免日志条数爆炸
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # 更新循环状态
    # first_step_of_run: 本次训练运行的第一步（新跑或续训的第一步都算）
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # 这里手动管理 Python GC，纯工程优化。
    # 与 “神经网络怎么训练” 无关，主要是为了减少长训练时的额外停顿。
    # 思路：
    # - 训练初期会创建大量临时对象（编译器中间产物等）；先 collect 一次清干净
    # - freeze() 把当前还活着的对象“冻结”：以后 GC 不再扫描它们，扫描成本大幅下降
    # - disable() 关掉自动 GC，避免训练循环中突然停顿做扫描
    # - 每隔很久手动 collect 一次，作为防守性兜底
    if first_step_of_run:
        gc.collect() # 先把初始化阶段遗留的垃圾收干净
        gc.freeze()  # 冻结当前幸存对象，减少后面反复扫描
        gc.disable() # 训练主循环里先彻底关掉自动 GC
    elif step % 5000 == 0:
        gc.collect() # 很长训练里偶尔手动清一次，防守式处理

# 收尾打印一些总体统计
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# 把本次训练的关键信息写入项目报告（report.md），方便日后回顾、对比不同实验
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config,  # 命令行参数（实验配置全貌）
    {  # 训练设置概况
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    {  # 训练结果
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish()  # 关闭 wandb 上传线程，确保最后的日志全部落盘
compute_cleanup()   # 销毁 DDP 进程组、释放设备资源
