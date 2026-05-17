#!/bin/bash

set -euo pipefail

# Train a d34 base model on 4x A800 using the cloud image's active Python env.
#
# Typical AutoDL usage:
#   tmux new -s base-d34 "NGPU=4 bash runs/run_base_d34_a800_4gpu.sh 2>&1 | tee $HOME/autodl-fs/.nanochat/base_d34_a800_4gpu.log"
#   tmux attach -t base-d34
#
# This script assumes resources are already prepared, or lets speedrun.sh prepare them.
# For a separate resource step:
#   sh runs/prepare_resources.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/autodl-fs/.nanochat}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The current A800 image has shown Inductor/Triton instability, so keep optimizer
# compile disabled by default. Set NANOCHAT_COMPILE_OPTIMIZER=1 after validating
# a newer image/driver stack.
export NANOCHAT_COMPILE_OPTIMIZER="${NANOCHAT_COMPILE_OPTIMIZER:-0}"

PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
NGPU="${NGPU:-4}"
DEPTH="${DEPTH:-34}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
BASE_COMPILE="${BASE_COMPILE:-0}"
WANDB_RUN="${WANDB_RUN:-dummy}"
MODEL_TAG="${MODEL_TAG:-d34}"
EVAL_EVERY="${EVAL_EVERY:--1}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
SAVE_EVERY="${SAVE_EVERY:--1}"
# Extra arguments go directly to scripts.base_train, for example:
#   BASE_EXTRA_ARGS="--num-iterations 1000 --total-batch-size 1048576"
BASE_EXTRA_ARGS="${BASE_EXTRA_ARGS:-}"
RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"
BASE_EVAL_BATCH_SIZE="${BASE_EVAL_BATCH_SIZE:-4}"
BASE_EVAL_MODES="${BASE_EVAL_MODES:-core,bpb,sample}"

case "$NGPU" in
    4) ;;
    *) echo "error: this script is intended for 4x A800; set NGPU=4"; exit 1 ;;
esac

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python command not found: $PYTHON"
    echo "hint: use the cloud image's activated environment, or set PYTHON=/path/to/python"
    exit 1
fi

if command -v "$TORCHRUN" >/dev/null 2>&1; then
    TORCHRUN_CMD=("$TORCHRUN")
else
    "$PYTHON" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch.distributed.run") is None:
    print("error: neither torchrun nor torch.distributed.run is available")
    sys.exit(1)
PY
    TORCHRUN_CMD=("$PYTHON" -m torch.distributed.run)
fi

"$PYTHON" - <<'PY'
import importlib.util
import sys

required = ["torch", "nanochat", "datasets", "tokenizers", "tiktoken", "setuptools"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print(f"error: missing Python packages: {', '.join(missing)}")
    print("Install them in the current cloud environment, e.g. python -m pip install -e '.[gpu]'")
    sys.exit(1)

import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
PY

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "error: missing tokenizer files in $TOKENIZER_DIR"
    echo "hint: run sh runs/prepare_resources.sh first, or use runs/speedrun.sh to prepare resources"
    exit 1
fi

mkdir -p "$NANOCHAT_BASE_DIR"

echo "============================================================"
echo " nanochat d34 base training on 4x A800"
echo " NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo " PYTHON=$PYTHON  TORCHRUN=$TORCHRUN  NGPU=$NGPU"
echo " DEPTH=$DEPTH  MODEL_TAG=$MODEL_TAG  DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"
echo " TARGET_PARAM_DATA_RATIO=$TARGET_PARAM_DATA_RATIO  BASE_COMPILE=$BASE_COMPILE  OPT_COMPILE=$NANOCHAT_COMPILE_OPTIMIZER"
echo " WANDB_RUN=$WANDB_RUN"
echo "============================================================"

"${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="$NGPU" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --model-tag="$MODEL_TAG" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --compile="$BASE_COMPILE" \
    --eval-every="$EVAL_EVERY" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    --save-every="$SAVE_EVERY" \
    --run="$WANDB_RUN" \
    $BASE_EXTRA_ARGS

BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
echo ""
echo "Base checkpoint files:"
ls -lh "$BASE_CKPT_DIR"

if [ "$RUN_BASE_EVAL" = "1" ]; then
    "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="$NGPU" -m scripts.base_eval -- \
        --model-tag "$MODEL_TAG" \
        --eval "$BASE_EVAL_MODES" \
        --device-batch-size "$BASE_EVAL_BATCH_SIZE"
fi

echo ""
echo "Base model is in: $BASE_CKPT_DIR"
