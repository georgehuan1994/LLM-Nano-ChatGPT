#!/bin/bash

set -euo pipefail

# Run Karpathy 发布的 d34 预训练 nanochat base model（来自 huggingface.co/karpathy/nanochat-d34）。
#
# 前置：先用 runs/download_nanochat_d34.sh 把模型下载到
#   $NANOCHAT_BASE_DIR/pretrained/nanochat-d34/
# 本脚本会自动把下载下来的文件“硬链接”到 nanochat 期望的标准目录：
#   $NANOCHAT_BASE_DIR/tokenizer/{tokenizer.pkl, token_bytes.pt}
#   $NANOCHAT_BASE_DIR/base_checkpoints/d34/{model_*.pt, meta_*.json}
# 硬链接不占双倍磁盘（NTFS 支持），失败时会回退为复制。
#
# Examples from Git Bash / bash:
#   bash runs/run_base_d34_4060.sh
#   PROMPT="The meaning of life is" bash runs/run_base_d34_4060.sh
#   MODE=sample bash runs/run_base_d34_4060.sh
#   MODE=bpb SPLIT_TOKENS=65536 bash runs/run_base_d34_4060.sh
#
# 显存提醒:
#   d34 在 BF16 下权重 ~8.6GB；RTX 4060 8GB 极易 OOM，建议 4060 Ti 16GB 或更高。
#   若 OOM，可尝试 DEVICE_TYPE=cpu（极慢）或减小 DEVICE_BATCH_SIZE。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/.nanochat}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MODEL_TAG="${MODEL_TAG:-d34}"
MODE="${MODE:-prompt}"
PROMPT="${PROMPT:-The capital of France is}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-50}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
SPLIT_TOKENS="${SPLIT_TOKENS:-32768}"

PRETRAINED_DIR="${PRETRAINED_DIR:-$NANOCHAT_BASE_DIR/pretrained/nanochat-d34}"

# 激活虚拟环境
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "error: no activation script found in .venv"
    exit 1
fi
PYTHON="python"
echo "使用 Python: $(command -v python)"

if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "error: 未找到预训练目录: $PRETRAINED_DIR"
    echo "hint: 先运行 bash runs/download_nanochat_d34.sh"
    exit 1
fi

# -----------------------------------------------------------------------------
# 把下载下来的文件“安装”到 nanochat 期望的标准目录结构。
# 优先硬链接（os.link），失败时回退为复制。幂等：已存在则跳过。
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
mkdir -p "$TOKENIZER_DIR" "$CHECKPOINT_DIR"

PRETRAINED_DIR="$PRETRAINED_DIR" \
TOKENIZER_DIR="$TOKENIZER_DIR" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
"$PYTHON" - <<'PY'
import os
import shutil
from pathlib import Path

src_dir = Path(os.environ["PRETRAINED_DIR"])
tok_dir = Path(os.environ["TOKENIZER_DIR"])
ckpt_dir = Path(os.environ["CHECKPOINT_DIR"])

mappings = [
    (src_dir / "tokenizer.pkl",  tok_dir / "tokenizer.pkl"),
    (src_dir / "token_bytes.pt", tok_dir / "token_bytes.pt"),
]
for f in src_dir.glob("model_*.pt"):
    mappings.append((f, ckpt_dir / f.name))
for f in src_dir.glob("meta_*.json"):
    mappings.append((f, ckpt_dir / f.name))

for src, dst in mappings:
    if not src.exists():
        raise FileNotFoundError(f"missing source file: {src}")
    if dst.exists():
        try:
            same_inode = dst.stat().st_ino != 0 and dst.stat().st_ino == src.stat().st_ino
        except OSError:
            same_inode = False
        same_size = dst.stat().st_size == src.stat().st_size
        if same_inode or same_size:
            # NTFS hardlink check via st_ino is unreliable; fall back to size match.
            print(f"  ok   {dst}  (matches source)")
            continue
        # Size differs -> destination is stale (e.g. left over from a previous model).
        # Replace it so we always reflect what was just downloaded.
        print(f"  stale {dst}  (size {dst.stat().st_size} != {src.stat().st_size}); replacing")
        dst.unlink()
    try:
        os.link(src, dst)
        print(f"  link {dst}  <- {src}")
    except OSError as e:
        print(f"  link failed ({e}); falling back to copy: {dst}")
        shutil.copy2(src, dst)
        print(f"  copy {dst}")
PY

"$PYTHON" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "nanochat") if importlib.util.find_spec(name) is None]
if missing:
    print(f"error: missing Python packages: {', '.join(missing)}")
    print("hint: run: UV_EXTRA=gpu sh runs/setup_uv_env.sh")
    sys.exit(1)
PY

echo "============================================================"
echo " nanochat d34 base model runner (RTX 4060 / 16G+)"
echo " NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo " MODEL_TAG=$MODEL_TAG  MODE=$MODE  DEVICE_TYPE=$DEVICE_TYPE"
echo " CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "============================================================"

case "$MODE" in
    prompt)
        export NANOCHAT_PROMPT="$PROMPT"
        export NANOCHAT_MAX_TOKENS="$MAX_TOKENS"
        export NANOCHAT_TEMPERATURE="$TEMPERATURE"
        export NANOCHAT_TOP_K="$TOP_K"
        export NANOCHAT_MODEL_TAG="$MODEL_TAG"
        export NANOCHAT_DEVICE_TYPE="$DEVICE_TYPE"
        "$PYTHON" - <<'PY'
import os
import torch

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

device_type = os.environ.get("NANOCHAT_DEVICE_TYPE", "") or autodetect_device_type()
model_tag = os.environ.get("NANOCHAT_MODEL_TAG") or None
prompt = os.environ["NANOCHAT_PROMPT"]
max_tokens = int(os.environ["NANOCHAT_MAX_TOKENS"])
temperature = float(os.environ["NANOCHAT_TEMPERATURE"])
top_k = int(os.environ["NANOCHAT_TOP_K"])

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag)
engine = Engine(model, tokenizer)

tokens = tokenizer(prompt, prepend="<|bos|>")
print(f"\nPrompt: {prompt}")
print("Completion: ", end="", flush=True)
with torch.inference_mode():
    for token_column, token_masks in engine.generate(tokens, num_samples=1, max_tokens=max_tokens, temperature=temperature, top_k=top_k):
        token = token_column[0]
        print(tokenizer.decode([token]), end="", flush=True)
print()
PY
        ;;
    sample)
        "$PYTHON" -m scripts.base_eval \
            --eval sample \
            --model-tag "$MODEL_TAG" \
            --device-type "$DEVICE_TYPE"
        ;;
    bpb)
        "$PYTHON" -m scripts.base_eval \
            --eval bpb \
            --model-tag "$MODEL_TAG" \
            --device-type "$DEVICE_TYPE" \
            --device-batch-size "$DEVICE_BATCH_SIZE" \
            --split-tokens "$SPLIT_TOKENS"
        ;;
    *)
        echo "error: unsupported MODE=$MODE (must be prompt, sample, or bpb)"
        exit 1
        ;;
esac
