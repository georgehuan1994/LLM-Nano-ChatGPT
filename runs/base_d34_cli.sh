#!/bin/bash

set -euo pipefail

# 使用 d34 base checkpoint 做命令行补全/采样。
# 默认从本地 .nanochat 加载：
#   $NANOCHAT_BASE_DIR/tokenizer/
#   $NANOCHAT_BASE_DIR/base_checkpoints/d34/
#
# 交互式补全：
#   bash runs/base_d34_cli.sh
#
# 指定提示词：
#   PROMPT="Why sky is blue?" bash runs/base_d34_cli.sh
#
# 常用覆盖项：
#   DEVICE_TYPE=cuda bash runs/base_d34_cli.sh
#   MAX_TOKENS=256 TEMPERATURE=0.7 TOP_K=40 bash runs/base_d34_cli.sh
#   NANOCHAT_BASE_DIR=/root/autodl-fs/.nanochat bash runs/base_d34_cli.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/.nanochat}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! "${OMP_NUM_THREADS:-1}" =~ ^[0-9]+$ ]]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi

PYTHON="${PYTHON:-python}"
MODEL_TAG="${MODEL_TAG:-d34}"
STEP="${STEP:-}"
PROMPT="${PROMPT:-}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-50}"
DEVICE_TYPE="${DEVICE_TYPE:-cpu}"

if [ "$DEVICE_TYPE" = "cpu" ]; then
    export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float32}"
fi

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python command not found: $PYTHON"
    echo "hint: use the activated Python environment, or set PYTHON=/path/to/python"
    exit 1
fi

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "error: missing tokenizer files in $TOKENIZER_DIR"
    echo "hint: make sure NANOCHAT_BASE_DIR points to the directory used by the d34 checkpoint"
    exit 1
fi

if ! ls "$BASE_CKPT_DIR"/model_*.pt >/dev/null 2>&1 || ! ls "$BASE_CKPT_DIR"/meta_*.json >/dev/null 2>&1; then
    echo "error: missing base checkpoint in $BASE_CKPT_DIR"
    echo "hint: expected model_*.pt and meta_*.json under base_checkpoints/$MODEL_TAG"
    echo "hint: if you downloaded karpathy/nanochat-d34 to pretrained/nanochat-d34, run runs/run_base_d34_4060.sh once to install links"
    exit 1
fi

# 单卡推理即可。如果用户没有显式指定显卡，非 CPU/MPS 模式默认使用 0 号卡。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${DEVICE_TYPE:-cuda}" != "cpu" ] && [ "${DEVICE_TYPE:-cuda}" != "mps" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "============================================================"
echo " nanochat d34 base command-line completion"
echo " NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo " BASE_CKPT_DIR=$BASE_CKPT_DIR"
echo " PYTHON=$PYTHON  MODEL_TAG=$MODEL_TAG  STEP=${STEP:-latest}"
echo " DEVICE_TYPE=$DEVICE_TYPE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " MAX_TOKENS=$MAX_TOKENS  TEMPERATURE=$TEMPERATURE  TOP_K=$TOP_K"
echo "============================================================"

BASE_ARGS=(
    -i base
    -g "$MODEL_TAG"
    -m "$MAX_TOKENS"
    -t "$TEMPERATURE"
    -k "$TOP_K"
    --device-type "$DEVICE_TYPE"
)

if [ -n "$PROMPT" ]; then
    BASE_ARGS+=(-p "$PROMPT")
fi

if [ -n "$STEP" ]; then
    BASE_ARGS+=(-s "$STEP")
fi

"$PYTHON" -m scripts.base_cli "${BASE_ARGS[@]}"
