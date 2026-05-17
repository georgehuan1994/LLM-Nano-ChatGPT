#!/bin/bash

# 下载 Karpathy 发布的 34 层预训练 nanochat 模型（含分词器与权重）
# 仓库主页: https://huggingface.co/karpathy/nanochat-d34
#
# 用法：
#   bash runs/download_nanochat_d34.sh                       # 默认下载到 $NANOCHAT_BASE_DIR/base_checkpoints/d34
#   TARGET_DIR=/data/nanochat-d34 bash runs/download_nanochat_d34.sh
#   REPO_ID=karpathy/nanochat-d20 bash runs/download_nanochat_d34.sh
#   HF_ENDPOINT=https://huggingface.co bash runs/download_nanochat_d34.sh   # 切回官方站
#
# 后台下载（推荐，8.58GB 文件，避免 SSH 断开）：
#   tmux new -s dl "bash runs/download_nanochat_d34.sh 2>&1 | tee runs/download_nanochat_d34.log"
#   分离: Ctrl+B D       重连: tmux attach -t dl
#
# 注意:
#   - 断点续传由 huggingface_hub 自带；中断后重新执行同一条命令即可
#   - d34 体积较大，请先确认目标盘有 ≥100GB 剩余空间

set -euo pipefail

if [ ! -f "pyproject.toml" ]; then
    echo "error: 请在仓库根目录下执行此脚本"
    exit 1
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/autodl-fs/.nanochat}"
# export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/.nanochat}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

REPO_ID="${REPO_ID:-karpathy/nanochat-d34}"
TARGET_DIR="${TARGET_DIR:-$NANOCHAT_BASE_DIR/base_checkpoints/d34}"
MAX_WORKERS="${MAX_WORKERS:-4}"

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python command not found: $PYTHON"
    echo "hint: use the global Python 3.12 environment, or set PYTHON=/path/to/python"
    exit 1
fi
echo "使用 Python: $(command -v "$PYTHON")"

# 确认 huggingface_hub 已安装
"$PYTHON" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("huggingface_hub") is None:
    print("error: 缺少依赖 huggingface_hub")
    print("可执行: pip install -U huggingface_hub")
    sys.exit(1)
PY

mkdir -p "$TARGET_DIR"

echo "============================================================"
echo " 下载 nanochat 预训练模型"
echo "   REPO_ID      = $REPO_ID"
echo "   TARGET_DIR   = $TARGET_DIR"
echo "   HF_ENDPOINT  = $HF_ENDPOINT"
echo "   MAX_WORKERS  = $MAX_WORKERS"
echo "============================================================"

EXTRA_ARGS=()
if [ -n "${INCLUDE:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(--include $INCLUDE)
fi
if [ -n "${EXCLUDE:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(--exclude $EXCLUDE)
fi
if [ -n "${REVISION:-}" ]; then
    EXTRA_ARGS+=(--revision "$REVISION")
fi

# 注意: 在 bash 3.x（Git Bash 自带版本之一）+ set -u 下，
# 直接展开空数组 "${EXTRA_ARGS[@]}" 会报 unbound variable 或被 shell 误解析。
# 这里用 ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 兼容写法：仅在数组非空时展开。
"$PYTHON" -m scripts.download_pretrained \
    --repo-id "$REPO_ID" \
    --target-dir "$TARGET_DIR" \
    --max-workers "$MAX_WORKERS" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "完成。文件位于: $TARGET_DIR"
