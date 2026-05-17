#!/bin/bash

set -euo pipefail

# Continue from an existing d34 base checkpoint and train the chat SFT model on
# a single A800. This script intentionally skips tokenizer/base pretraining.
#
# Expected inputs:
#   $NANOCHAT_BASE_DIR/tokenizer/{tokenizer.pkl,token_bytes.pt}
#   $NANOCHAT_BASE_DIR/base_checkpoints/d34/{model_*.pt,meta_*.json}
#
# Output:
#   $NANOCHAT_BASE_DIR/chatsft_checkpoints/d34/model_*.pt
#   $NANOCHAT_BASE_DIR/chatsft_checkpoints/d34/meta_*.json
#
# Recommended cloud launch:
#   tmux new -s sft-d34 "bash runs/run_sft_d34_a800.sh 2>&1 | tee $HOME/autodl-fs/.nanochat/sft_d34_a800.log"
#   tmux attach -t sft-d34

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/autodl-fs/.nanochat}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

MODEL_TAG="${MODEL_TAG:-d34}"
NGPU="${NGPU:-1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
WANDB_RUN="${WANDB_RUN:-dummy}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-0}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_TOKENS="${EVAL_TOKENS:-20971520}"
CHATCORE_EVERY="${CHATCORE_EVERY:--1}"
CHATCORE_MAX_SAMPLE="${CHATCORE_MAX_SAMPLE:-24}"
CHATCORE_MAX_CAT="${CHATCORE_MAX_CAT:--1}"
SFT_EXTRA_ARGS="${SFT_EXTRA_ARGS:-}"
RUN_CHAT_EVAL="${RUN_CHAT_EVAL:-1}"
CHAT_EVAL_MAX_PROBLEMS="${CHAT_EVAL_MAX_PROBLEMS:-200}"
CHAT_EVAL_BATCH_SIZE="${CHAT_EVAL_BATCH_SIZE:-8}"
PROMPT="${PROMPT:-你好，简单介绍一下你自己。}"

case "$NGPU" in
    1) ;;
    *) echo "error: this script is for a single A800; set NGPU=1"; exit 1 ;;
esac

if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "error: no activation script found in .venv"
    echo "hint: run: UV_EXTRA=gpu sh runs/setup_uv_env.sh"
    exit 1
fi

PYTHON="python"
TORCHRUN="$(command -v torchrun || true)"
if [ -z "$TORCHRUN" ]; then
    echo "error: torchrun not found in the active environment"
    echo "hint: run: UV_EXTRA=gpu sh runs/setup_uv_env.sh"
    exit 1
fi

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"

if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "error: missing tokenizer files in $TOKENIZER_DIR"
    echo "hint: copy the tokenizer from the base training run or run the same tokenizer preparation step first"
    exit 1
fi

if ! ls "$BASE_CKPT_DIR"/model_*.pt >/dev/null 2>&1 || ! ls "$BASE_CKPT_DIR"/meta_*.json >/dev/null 2>&1; then
    echo "error: missing d34 base checkpoint in $BASE_CKPT_DIR"
    echo "hint: expected model_*.pt and meta_*.json under base_checkpoints/$MODEL_TAG"
    exit 1
fi

mkdir -p "$NANOCHAT_BASE_DIR"
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "============================================================"
echo " nanochat d34 SFT on single A800"
echo " NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo " BASE_CKPT_DIR=$BASE_CKPT_DIR"
echo " SFT_CKPT_DIR=$SFT_CKPT_DIR"
echo " DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE  LOAD_OPTIMIZER=$LOAD_OPTIMIZER"
echo " WANDB_RUN=$WANDB_RUN  CHATCORE_EVERY=$CHATCORE_EVERY"
echo "============================================================"

"$TORCHRUN" --standalone --nproc_per_node="$NGPU" -m scripts.chat_sft -- \
    --model-tag "$MODEL_TAG" \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --load-optimizer "$LOAD_OPTIMIZER" \
    --eval-every "$EVAL_EVERY" \
    --eval-tokens "$EVAL_TOKENS" \
    --chatcore-every "$CHATCORE_EVERY" \
    --chatcore-max-cat "$CHATCORE_MAX_CAT" \
    --chatcore-max-sample "$CHATCORE_MAX_SAMPLE" \
    --run "$WANDB_RUN" \
    $SFT_EXTRA_ARGS

echo ""
echo "SFT checkpoint files:"
ls -lh "$SFT_CKPT_DIR"

if [ "$RUN_CHAT_EVAL" = "1" ]; then
    "$TORCHRUN" --standalone --nproc_per_node="$NGPU" -m scripts.chat_eval -- \
        -i sft \
        -g "$MODEL_TAG" \
        -b "$CHAT_EVAL_BATCH_SIZE" \
        -x "$CHAT_EVAL_MAX_PROBLEMS"
fi

"$PYTHON" -m scripts.chat_cli \
    -i sft \
    -g "$MODEL_TAG" \
    -p "$PROMPT"

echo ""
echo "Final chat model is in: $SFT_CKPT_DIR"