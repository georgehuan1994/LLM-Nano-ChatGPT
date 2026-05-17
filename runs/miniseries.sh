#!/bin/bash

# See speedrun.sh for more comments
# Usage: ./miniseries.sh [series_name]
# Example: ./miniseries.sh jan11
# Default series name is today's date (e.g., jan11)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python command not found: $PYTHON"
    echo "hint: use the global Python 3.12 environment, or set PYTHON=/path/to/python"
    exit 1
fi

if ! command -v "$TORCHRUN" >/dev/null 2>&1; then
    echo "error: torchrun command not found: $TORCHRUN"
    echo "hint: use the global PyTorch 2.8.0 environment, or set TORCHRUN=/path/to/torchrun"
    exit 1
fi

# Setup (skip with SKIP_SETUP=1)
if [ -z "$SKIP_SETUP" ]; then
    # Tokenizer, download 1000 shards for pretraining
    # (probably this can be reduced but it's tricky to determine the exact right number, TODO).
    "$PYTHON" -m nanochat.dataset -n 1000
    "$PYTHON" -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
fi

# Series name: from arg, env var, or default to today's date (e.g., jan11)
SERIES_NAME="${1:-${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
# Depths to train (the "miniseries")
DEPTHS=(12 14 16 18 20 22 24 26)
# Hardware
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Logging
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}_miniseries}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_miniseries_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,model_dim,num_params,num_scaling_params,num_iterations,tokens_trained,param_data_ratio,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "${SERIES_NAME} Miniseries Training"
log "=============================================="

for d in "${DEPTHS[@]}"; do
    log "Training d=$d..."

    TAG="${SERIES_NAME}_miniseries_d${d}"
    START_TIME=$(date +%s)

    # Reduce --device-batch-size to avoid OOM at larger depths
    if [ $d -ge 28 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=8"
    elif [ $d -ge 20 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=16"
    else
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=32"
    fi

    "$TORCHRUN" --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$d \
        --run="${WANDB_RUN}_d${d}" \
        --model-tag="${TAG}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        $DEVICE_BATCH_SIZE_ARG \
        2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract stats from log
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
    NUM_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | head -1 | tr -d ',')
    NUM_SCALING_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" | tail -1 | grep -oP 'scaling: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
    TOKENS_TRAINED=$((NUM_ITERS * 524288))
    PARAM_DATA_RATIO=$("$PYTHON" -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
    MODEL_DIM=$((d * 64))
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')

    if [ -z "$CORE_SCORE" ]; then
        CORE_SCORE="0.0"
    fi

    log "  d=$d: params=$NUM_PARAMS, scaling=$NUM_SCALING_PARAMS, ratio=$PARAM_DATA_RATIO, bpb=$VAL_BPB, CORE=$CORE_SCORE, time=${TRAIN_TIME}s"

    # Append to CSV
    echo "$d,$MODEL_DIM,$NUM_PARAMS,$NUM_SCALING_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$PARAM_DATA_RATIO,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
done

log "=============================================="
log "${SERIES_NAME} Miniseries Complete!"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
