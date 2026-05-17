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
# 默认关闭 wandb 上传，避免云端首次运行时卡在交互式登录提示。
# 如需记录曲线，先运行 wandb login，再设置 WANDB_RUN=sft-d34-a800-4gpu。
export WANDB_RUN="${WANDB_RUN:-dummy}"
export NANOCHAT_COMPILE_OPTIMIZER="${NANOCHAT_COMPILE_OPTIMIZER:-0}"
export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-warning}"

# AutoDL/torchrun 环境里有时会继承到 libgomp 不接受的 OMP_NUM_THREADS 值。
# 4 卡训练这里默认固定为 1，避免每个 rank 抢 CPU 线程。
export OMP_NUM_THREADS=1
PREPARE_SFT_DATA="${PREPARE_SFT_DATA:-1}"

case "${CUDA_VISIBLE_DEVICES:-}" in
    "") export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
esac

echo "Launching d34 SFT on 4x A800 via runs/run_sft_d34_a800.sh"
echo "NGPU=$NGPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MODEL_TAG=$MODEL_TAG DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"

if [ "$PREPARE_SFT_DATA" = "1" ]; then
    echo "Prewarming SFT datasets in a single process before torchrun..."
    "${PYTHON:-python}" - <<'PY'
import logging

for name in ("httpx", "httpcore", "huggingface_hub"):
    logging.getLogger(name).setLevel(logging.WARNING)

from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K

datasets = [
    SmolTalk(split="train"),
    SmolTalk(split="test"),
    MMLU(subset="all", split="auxiliary_train"),
    MMLU(subset="all", split="test"),
    GSM8K(subset="main", split="train"),
    GSM8K(subset="main", split="test"),
]
print("SFT dataset cache is ready:", ", ".join(f"{type(ds).__name__}={ds.num_examples()}" for ds in datasets))
PY
fi

exec bash "$SCRIPT_DIR/run_sft_d34_a800.sh"