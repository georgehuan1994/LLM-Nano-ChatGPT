#!/bin/bash

set -euo pipefail

# 在 4x A800 上从已经训练好的 d34 base checkpoint 继续做 chat SFT。
#
# 运行前先确认：
#   bash runs/check_cloud_env.sh
#   sh runs/prepare_resources.sh
#
# 必须已经存在：
#   $NANOCHAT_BASE_DIR/tokenizer/{tokenizer.pkl,token_bytes.pt}
#   $NANOCHAT_BASE_DIR/base_checkpoints/d34/{model_*.pt,meta_*.json}
#
# AutoDL 推荐用 tmux 启动：
#   tmux new -s sft-d34-4gpu "bash runs/run_sft_d34_a800_4gpu.sh 2>&1 | tee $HOME/autodl-fs/.nanochat/sft_d34_a800_4gpu.log"
#   tmux attach -t sft-d34-4gpu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NGPU=4
export MODEL_TAG="${MODEL_TAG:-d34}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
export COMPILE="${COMPILE:-0}"
export SKIP_VAL_BPB="${SKIP_VAL_BPB:-1}"
export CHATCORE_EVERY="${CHATCORE_EVERY:--1}"
export RUN_CHAT_EVAL="${RUN_CHAT_EVAL:-1}"
export CHAT_EVAL_BATCH_SIZE="${CHAT_EVAL_BATCH_SIZE:-8}"
export CHAT_EVAL_MAX_PROBLEMS="${CHAT_EVAL_MAX_PROBLEMS:-200}"
export WANDB_RUN="${WANDB_RUN:-sft-d34-a800-4gpu}"
export NANOCHAT_COMPILE_OPTIMIZER="${NANOCHAT_COMPILE_OPTIMIZER:-0}"

case "${CUDA_VISIBLE_DEVICES:-}" in
    "") export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
esac

echo "Launching d34 SFT on 4x A800 via runs/run_sft_d34_a800.sh"
echo "NGPU=$NGPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MODEL_TAG=$MODEL_TAG DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"

exec bash "$SCRIPT_DIR/run_sft_d34_a800.sh"