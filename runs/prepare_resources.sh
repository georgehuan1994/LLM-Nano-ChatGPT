#!/bin/sh

# Prepare data and reusable resources before GPU training.
#
# Usage after the uv environment is ready:
#   sh runs/prepare_resources.sh
#
# Defaults:
#   NANOCHAT_BASE_DIR=$HOME/autodl-fs/.nanochat
#   RESOURCE_SHARDS=170

set -eu

if [ ! -f "pyproject.toml" ]; then
    echo "error: please run this script from the repository root"
    exit 1
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/autodl-fs/.nanochat}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RESOURCE_SHARDS="${RESOURCE_SHARDS:-170}"
PYTHON=".venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON is not executable. Run: sh runs/setup_uv_env.sh"
    exit 1
fi

mkdir -p "$NANOCHAT_BASE_DIR"

echo "Preparing resources in: $NANOCHAT_BASE_DIR"
echo "Dataset shards: $RESOURCE_SHARDS"
echo "HuggingFace endpoint: $HF_ENDPOINT"

"$PYTHON" -m nanochat.dataset -n "$RESOURCE_SHARDS"
"$PYTHON" -m scripts.tok_train
"$PYTHON" -m scripts.tok_eval

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

"$PYTHON" - <<'PY'
from nanochat.common import download_file_with_lock
from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle

download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
PY

echo "Resources are ready."
