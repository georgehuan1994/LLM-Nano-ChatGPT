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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# 命令行参数
#
# 对初学者来说，可以先把这些参数分成四类看：
# 1. 运行环境参数：用什么设备、是否记录日志
# 2. 模型结构参数：几层、隐藏维度多大、上下文多长
# 3. 优化参数：batch size、多大学习率、训练多少步
# 4. 评估参数：多久评一次、评哪些指标
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# 计算/设备初始化 + 日志初始化
#
# 这一步的目标是：
# 1. 确定当前用 CPU / CUDA / MPS 哪种设备；
# 2. 如果是多卡训练，初始化 DDP；
# 3. 准备好 wandb 或 dummy logger。

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # 只有 rank 0 负责日志、保存 checkpoint、打印主要信息等
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
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
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# 加载 tokenizer 及其配套元数据
#
# 为什么训练模型前必须先有 tokenizer？
# 因为模型的输入/输出维度都和词表大小直接相关。
# 模型并不直接看字符串，而是看 token id；
# 同时最后一层 lm_head 也要输出“对整个词表中每个 token 的打分(logits)”。
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# 构建模型
#
# 对纯小白来说，这里最关键的概念有三个：
# 1. `depth`：Transformer block 堆多少层，也就是网络有多深
# 2. `n_embd`：每个 token 在网络内部表示成多长的向量，也就是隐藏维度
# 3. `n_head`：注意力头的数量，表示把注意力机制分成多少个并行子空间去看信息
#
# 注意：
# `base_train.py` 负责“决定模型规模并训练它”；
# 真正的神经网络内部实现细节（embedding、attention、MLP、lm_head 等）
# 定义在 `nanochat/gpt.py` 里。

def build_model_meta(depth):
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
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim

    # 这里构造的是 GPTConfig，不是实际训练数据。
    # 你可以把它看成“神经网络蓝图”：
    # - sequence_len: 最长上下文长度
    # - vocab_size: 词表大小，决定 embedding/lm_head 的输入输出规模
    # - n_layer: 堆多少层 Transformer block
    # - n_head: 多头注意力头数
    # - n_embd: 每个 token 的隐藏向量维度
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# 真正建模分成三步：
# 1. 先在 meta device 上搭出“网络结构空壳”
# 2. 再把这些参数张量搬到目标设备上，但此时里面还是未初始化的垃圾值
# 3. 最后显式调用 init_weights() 做可控初始化
#
# 这种写法比“构造函数里直接偷偷初始化一切”更适合教学，也更省显存。
model = build_model_meta(args.depth) # 1) 只构建结构，不分配真实参数数据
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) 在目标设备上分配参数存储，但尚未初始化
model.init_weights() # 3) 真正初始化参数

# 如果是断点续训，就用 checkpoint 里的参数覆盖刚初始化好的随机参数。
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

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

# 这个上下文管理器用于“临时关闭 FP8”。
# 原因是训练时可以为速度启用 FP8，但评估时通常更希望数值稳定、口径一致，
# 所以会临时切回 BF16/普通 Linear。
@contextmanager
def disable_fp8(model):
    """
    临时把 Float8Linear 换回普通 Linear，供评估阶段使用。

    这里不是“修改训练逻辑”，而是做一个短暂的模块替换：
    进入 with 时替换，退出 with 时恢复。
    """
    import torch.nn as nn

    # 先找到模型里所有 Float8Linear 所在的位置，
    # 记住它们的父模块和属性名，后面才能原位替换再恢复。
    fp8_locations = []  # 形如 (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # 把 Float8Linear 暂时替换成普通 Linear。
    # 这里用 meta device 先造外壳，是为了避免替换时额外占用一大块显存。
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# 编译模型
#
# torch.compile 可以把 Python 级的 eager 执行图优化成更高效的执行形式。
# 对初学者来说，可以先把它理解成：
# “不改模型数学定义，只是尽量把执行变快”。

orig_model = model # 保留未 compile 的原始模型，便于保存、推理和某些变长输入场景
if device_type == "cuda":
    model = torch.compile(model, dynamic=False) # 训练时输入形状固定，所以 dynamic=False 更稳也更快
else:
    print0(f"Skipping torch.compile on {device_type}. This avoids torch-inductor compiler requirements on CPU/MPS.")

# -----------------------------------------------------------------------------
# 根据 scaling laws / muP 风格经验规则，自动估算更合理的训练预算和超参数
#
# 这部分对初学者最容易看晕，所以先抓住主线：
# 1. 模型参数量越大，通常也应该训练更多 token
# 2. token 预算确定后，可以估一个比较合理的 batch size
# 3. batch size 变化后，学习率和 weight decay 往往也要跟着缩放

# 先统计模型参数量。
# 参数量是理解模型规模最基础的指标之一。
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) 先根据 scaling laws 估算“应该训练多少 token”
#
# 直觉上：
# 小模型喂太多数据可能收益变小，大模型喂太少数据又学不满。
# 所以常见经验做法是维持一个 Tokens : Params 的目标比例。
def get_scaling_params(m):
    # 这里作者发现：
    # 用 transformer_matrices + lm_head 作为 scaling params，
    # 在经验拟合上更“干净”。
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # 估计这次训练总共应该看多少 token

# 这里选一个 d12 模型作为“参考尺度”。
# 可以理解成：很多经验超参数最初是在这个参考模型上调出来的，
# 然后再按 muP/scaling rules 外推到更深的模型。
d12_ref = build_model_meta(12) # 只构建结构，不分配真实参数
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) 再根据 token 预算估 batch size
#
# batch size 可以理解成：
# 一次参数更新之前，总共看多少 token。
# 太小训练噪声大，太大又可能不划算或占太多显存。
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) batch size 变了，学习率通常也要跟着缩放。
#
# 很粗的直觉：
# 一次更新里看得更多，梯度平均起来更稳，学习率可以适当放大一点。
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
# 这部分完全是“经验/论文启发下的工程规则”，
# 不是 Transformer 数学定义本身的一部分。
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
# 对初学者可以先把优化器理解成：
# “根据 loss 反向传播得到梯度后，真正负责更新参数的规则”。
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# FP16 下的 GradScaler
#
# FP16 数值范围更小，容易梯度下溢。
# GradScaler 的作用是先把 loss 放大再反传，最后更新前再缩回来，提高数值稳定性。
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# 构建 train / val dataloader
#
# 这里的数据已经不是原始字符串，而是会被 tokenizer 动态编码后，
# 组织成适合语言模型训练的 `(x, y)`：
#
# - x: 当前看到的 token 序列
# - y: 右移一位后的目标 token 序列
#
# 也就是说，模型学的是：
# “看到 x 的前缀后，y 这个位置下一个 token 应该是什么”
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # 先取出第一批数据，顺便启动后续取数流程

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
# 2. 中间保持常数
# 3. 结尾 warmdown，平滑收尾
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Muon 的 momentum 也做一个随训练进度变化的调度。
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
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
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
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# 正式开始训练
while True:
    last_step = step == num_iterations # 多跑一轮，方便在最后也做一次评估和保存
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 周期性评估验证集 BPB。
    # 这是当前项目里最核心的语言建模指标之一。
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
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
        model.train()

    # 周期性做 CORE 评估。
    # CORE 更像“任务表现”的外部能力测试，不只是下一个 token 的 loss。
    #
    # 这里用 orig_model，是因为 CORE 输入长度变化更大，
    # compiled model 未必合适。
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
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # 保存 checkpoint。
    # checkpoint 可以理解成“训练快照”：
    # 包含模型参数、优化器状态、dataloader 进度、循环状态等。
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # 终止条件：训练步数达到预定上限。
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
    #   真正把参数改掉
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        # model(x, y) 在训练模式下返回的是交叉熵 loss。
        #
        # 这里内部已经做了完整 GPT 前向：
        # 1. token embedding
        # 2. 多层 Transformer block（attention + MLP）
        # 3. lm_head 输出每个位置对全词表的 logits
        # 4. 与目标 y 计算 cross entropy
        loss = model(x, y)
        train_loss = loss.detach() # 单独留一份给日志，避免后续图被修改

        # 因为后面会做 grad accumulation，
        # 每个 micro-step 的 loss 要先除以累积步数，保证总梯度尺度正确。
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # 预取下一批数据，让取数和 GPU 计算尽量重叠，提高吞吐。
        x, y, dataloader_state_dict = next(train_loader)

    # 走到这里，梯度已经累计完毕，开始真正更新参数。
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        # FP16 路径下，需要先 unscale，再 step。
        scaler.unscale_(optimizer)
        # 多卡时如果任何一张卡出现 inf/nan 梯度，所有卡都必须一致地跳过这一步，
        # 否则参数会不同步。
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 下面主要是日志统计，不参与模型数学本身。
    ema_beta = 0.9 # 对训练 loss 做 EMA 平滑，让曲线更稳定、更好读
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # 前期做去偏，避免 EMA 初始值影响太大
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
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
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # 这里手动管理 Python GC，纯工程优化。
    # 与“神经网络怎么训练”无关，主要是为了减少长训练时的额外停顿。
    if first_step_of_run:
        gc.collect() # 先把初始化阶段遗留的垃圾收干净
        gc.freeze() # 冻结当前幸存对象，减少后面反复扫描
        gc.disable() # 训练主循环里先彻底关掉自动 GC
    elif step % 5000 == 0:
        gc.collect() # 很长训练里偶尔手动清一次，防守式处理

# 收尾打印一些总体统计
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
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
    { # stats about training outcomes
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
wandb_run.finish() # wandb run finish
compute_cleanup()
