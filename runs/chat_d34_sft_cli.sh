#!/bin/bash

set -euo pipefail

# 使用已经训练好的 d34 SFT checkpoint 启动命令行对话。
# 默认从 AutoDL 持久化数据盘加载：
#   $NANOCHAT_BASE_DIR/tokenizer/
#   $NANOCHAT_BASE_DIR/chatsft_checkpoints/d34/
#
# 交互式聊天：
#   bash runs/chat_d34_sft_cli.sh
#
# 单轮提问：
#   PROMPT="who are you?" bash runs/chat_d34_sft_cli.sh
#
# 常用覆盖项：
#   CUDA_VISIBLE_DEVICES=0 bash runs/chat_d34_sft_cli.sh
#   TEMPERATURE=0.7 TOP_K=40 bash runs/chat_d34_sft_cli.sh
#   DEVICE_TYPE=cpu bash runs/chat_d34_sft_cli.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NANOCHAT_BASE_DIR="$PWD/.nanochat"
# export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/autodl-fs/.nanochat}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! "${OMP_NUM_THREADS:-1}" =~ ^[0-9]+$ ]]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi

PYTHON="${PYTHON:-python}"
MODEL_TAG="${MODEL_TAG:-d34}"
SOURCE="${SOURCE:-sft}"
STEP="${STEP:-}"
PROMPT="${PROMPT:-}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_K="${TOP_K:-50}"

DEVICE_TYPE="${DEVICE_TYPE:-cpu}"
# DEVICE_TYPE="${DEVICE_TYPE:-cuda}"

if [ "$DEVICE_TYPE" = "cpu" ]; then
    export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float32}"
fi

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python command not found: $PYTHON"
    echo "hint: use the cloud image's activated environment, or set PYTHON=/path/to/python"
    exit 1
fi

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "error: missing tokenizer files in $TOKENIZER_DIR"
    echo "hint: make sure NANOCHAT_BASE_DIR points to the directory used during training"
    exit 1
fi

if ! ls "$SFT_CKPT_DIR"/model_*.pt >/dev/null 2>&1 || ! ls "$SFT_CKPT_DIR"/meta_*.json >/dev/null 2>&1; then
    echo "error: missing SFT checkpoint in $SFT_CKPT_DIR"
    echo "hint: expected model_*.pt and meta_*.json under chatsft_checkpoints/$MODEL_TAG"
    exit 1
fi

# 单卡推理即可。如果用户没有显式指定显卡，默认使用 0 号卡。
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${DEVICE_TYPE:-cuda}" != "cpu" ] && [ "${DEVICE_TYPE:-cuda}" != "mps" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "============================================================"
echo " nanochat d34 SFT command-line chat"
echo " NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo " SFT_CKPT_DIR=$SFT_CKPT_DIR"
echo " PYTHON=$PYTHON  SOURCE=$SOURCE  MODEL_TAG=$MODEL_TAG  STEP=${STEP:-latest}"
echo " DEVICE_TYPE=${DEVICE_TYPE:-auto}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " TEMPERATURE=$TEMPERATURE  TOP_K=$TOP_K"
echo "============================================================"

CHAT_ARGS=(
    -i "$SOURCE"
    -g "$MODEL_TAG"
    -t "$TEMPERATURE"
    -k "$TOP_K"
)

if [ -n "$STEP" ]; then
    CHAT_ARGS+=(-s "$STEP")
fi

if [ -n "$DEVICE_TYPE" ]; then
    CHAT_ARGS+=(--device-type "$DEVICE_TYPE")
fi

if [ -n "$PROMPT" ]; then
    CHAT_ARGS+=(-p "$PROMPT")
fi

"$PYTHON" -m scripts.chat_cli "${CHAT_ARGS[@]}"