#!/bin/bash

set -euo pipefail

# Run a pretrained nanochat base model on a local RTX 4060-class GPU.
#
# Expected files:
#   $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl
#   $NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt
#   $NANOCHAT_BASE_DIR/base_checkpoints/d12/model_*.pt
#   $NANOCHAT_BASE_DIR/base_checkpoints/d12/meta_*.json
#
# Default layout for this repository on your PC:
#   .nanochat/tokenizer
#   .nanochat/base_checkpoints/d12
#
# Examples from Git Bash / bash:
#   bash runs/run_base_d12_4060.sh
#   PROMPT="The meaning of life is" bash runs/run_base_d12_4060.sh
#   MODE=sample bash runs/run_base_d12_4060.sh
#   MODE=bpb SPLIT_TOKENS=65536 bash runs/run_base_d12_4060.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/.nanochat}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MODEL_TAG="${MODEL_TAG:-d12}"
MODE="${MODE:-prompt}"
PROMPT="${PROMPT:-The capital of France is}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-50}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
SPLIT_TOKENS="${SPLIT_TOKENS:-32768}"

find_executable() {
    for candidate in "$@"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON="$(find_executable .venv/bin/python .venv/Scripts/python.exe .venv/Scripts/python || true)"
if [ -z "$PYTHON" ]; then
    echo "error: Python executable was not found in .venv"
    echo "hint: run: UV_EXTRA=gpu sh runs/setup_uv_env.sh"
    exit 1
fi

CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"

if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "error: tokenizer directory not found: $TOKENIZER_DIR"
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "error: checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if ! ls "$CHECKPOINT_DIR"/model_*.pt >/dev/null 2>&1; then
    echo "error: no model_*.pt checkpoint found in: $CHECKPOINT_DIR"
    exit 1
fi

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
echo " nanochat base model runner for local RTX 4060"
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