#!/bin/sh

# Prepare the local uv/.venv Python environment.
#
# Default usage for GPU training:
#   sh runs/setup_uv_env.sh
#
# CPU/no-GPU preparation usage:
#   UV_EXTRA=cpu sh runs/setup_uv_env.sh
#
# Defaults are chosen for GPU training:
#   UV_EXTRA=gpu
#   CN_MIRROR=1
#   PYTORCH_MIRROR=0

set -eu

UV_EXTRA="${UV_EXTRA:-gpu}"
CN_MIRROR="${CN_MIRROR:-1}"
PYTORCH_MIRROR="${PYTORCH_MIRROR:-0}"
PYPROJECT_BAK=""

cleanup_mirror() {
    if [ -n "$PYPROJECT_BAK" ] && [ -f "$PYPROJECT_BAK" ]; then
        mv "$PYPROJECT_BAK" pyproject.toml
        echo "[CN_MIRROR] restored pyproject.toml"
    fi
}

trap cleanup_mirror EXIT HUP INT TERM

if [ ! -f "pyproject.toml" ]; then
    echo "error: please run this script from the repository root"
    exit 1
fi

case "$UV_EXTRA" in
    cpu|gpu) ;;
    *)
        echo "error: unsupported UV_EXTRA=$UV_EXTRA (must be cpu or gpu)"
        exit 1
        ;;
esac

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not available after installation"
    exit 1
fi

if [ "$CN_MIRROR" = "1" ]; then
    echo "[CN_MIRROR] using Tsinghua PyPI mirror"
    export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
fi

if [ "$PYTORCH_MIRROR" = "1" ]; then
    echo "[PYTORCH_MIRROR] using Aliyun PyTorch mirror"
    if ! grep -q "mirrors.aliyun.com/pytorch-wheels" pyproject.toml; then
        PYPROJECT_BAK="pyproject.toml.bak.setup_uv_env.$$"
        cp pyproject.toml "$PYPROJECT_BAK"
        sed -i \
            -e 's|https://download.pytorch.org/whl/cu128|https://mirrors.aliyun.com/pytorch-wheels/cu128|g' \
            -e 's|https://download.pytorch.org/whl/cpu|https://mirrors.aliyun.com/pytorch-wheels/cpu|g' \
            pyproject.toml
    fi
fi

[ -d ".venv" ] || uv venv
uv sync --extra "$UV_EXTRA"

cleanup_mirror
PYPROJECT_BAK=""
trap - EXIT HUP INT TERM

PYTHON=".venv/bin/python"
TORCHRUN=".venv/bin/torchrun"

if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON is not executable"
    exit 1
fi

if [ ! -x "$TORCHRUN" ]; then
    echo "error: $TORCHRUN is not executable"
    exit 1
fi

"$PYTHON" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "tokenizers", "nanochat") if importlib.util.find_spec(name) is None]
if missing:
    print(f"error: missing Python packages after uv sync: {', '.join(missing)}")
    sys.exit(1)
PY

echo "uv environment is ready: UV_EXTRA=$UV_EXTRA"
echo "python: $PYTHON"
echo "torchrun: $TORCHRUN"
