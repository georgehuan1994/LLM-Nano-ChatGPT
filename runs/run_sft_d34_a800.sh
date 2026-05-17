#!/bin/bash

set -euo pipefail

# 从已经训练好的 d34 base checkpoint 继续训练 chat SFT 模型，目标硬件是 A800。
# 这个脚本只负责“后半程”：不会重新训练 tokenizer，也不会重新做 base 预训练。
#
# 运行前必须已经存在这些输入文件：
#   $NANOCHAT_BASE_DIR/tokenizer/{tokenizer.pkl,token_bytes.pt}
#   $NANOCHAT_BASE_DIR/base_checkpoints/d34/{model_*.pt,meta_*.json}
#
# 训练完成后会输出最终对话模型到：
#   $NANOCHAT_BASE_DIR/chatsft_checkpoints/d34/model_*.pt
#   $NANOCHAT_BASE_DIR/chatsft_checkpoints/d34/meta_*.json
#
# 云端推荐用 tmux 启动，避免 SSH 断开导致训练中断：
#   tmux new -s sft-d34 "bash runs/run_sft_d34_a800.sh 2>&1 | tee $HOME/autodl-fs/.nanochat/sft_d34_a800.log"
#   tmux attach -t sft-d34

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# AutoDL 等云机器上建议把数据和 checkpoint 放在持久化数据盘 autodl-fs。
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/autodl-fs/.nanochat}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 优化器内部也有 torch.compile 版 fused AdamW/Muon。当前镜像缺 setuptools，且
# Inductor/Triton 组合已不稳定；默认关闭优化器编译，使用 eager optimizer。
export NANOCHAT_COMPILE_OPTIMIZER="${NANOCHAT_COMPILE_OPTIMIZER:-0}"

MODEL_TAG="${MODEL_TAG:-d34}"
NGPU="${NGPU:-1}"
# d34 在单张 A800 上关闭 torch.compile 后，eager backward 显存更高。
# 默认用 4 更稳；若仍 OOM，可运行前设 DEVICE_BATCH_SIZE=2 或 1。
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
WANDB_RUN="${WANDB_RUN:-dummy}"
# SFT 会重新初始化优化器；只做下游 SFT 时通常不需要 base 阶段的 optim_*.pt。
# 如果你明确保留了 base optimizer 并想 warm start，可运行前设置 LOAD_OPTIMIZER=1。
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-0}"
# 当前 AutoDL A800 镜像会在 torch.compile 的 Inductor/Triton kernel 启动时报
# "CUDA driver error: invalid argument"，而且训练 forward 也会触发。默认关闭 compile，
# 保留 BF16 eager 训练；如果更换镜像/驱动后想试编译加速，可运行前设置 COMPILE=1。
COMPILE="${COMPILE:-0}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_TOKENS="${EVAL_TOKENS:-20971520}"
# 默认跳过训练中的 val bpb，减少显存占用并避开验证路径的额外编译/评估开销。
# 训练结束后的 chat_eval 仍会运行；如需训练中验证，可运行前设置 SKIP_VAL_BPB=0。
SKIP_VAL_BPB="${SKIP_VAL_BPB:-1}"
# ChatCORE 完整评估很耗时，默认关闭；训练结束后脚本会另外跑一个小规模 chat_eval。
CHATCORE_EVERY="${CHATCORE_EVERY:--1}"
CHATCORE_MAX_SAMPLE="${CHATCORE_MAX_SAMPLE:-24}"
CHATCORE_MAX_CAT="${CHATCORE_MAX_CAT:--1}"
# 需要临时追加 chat_sft 参数时使用，例如：SFT_EXTRA_ARGS="--num-iterations 500"。
SFT_EXTRA_ARGS="${SFT_EXTRA_ARGS:-}"
# RUN_CHAT_EVAL=0 可跳过训练后的评测，只保留 checkpoint 和样例对话。
RUN_CHAT_EVAL="${RUN_CHAT_EVAL:-1}"
CHAT_EVAL_MAX_PROBLEMS="${CHAT_EVAL_MAX_PROBLEMS:-200}"
CHAT_EVAL_BATCH_SIZE="${CHAT_EVAL_BATCH_SIZE:-8}"
PROMPT="${PROMPT:-你好，简单介绍一下你自己。}"

case "$NGPU" in
    1|4) ;;
    *) echo "error: unsupported NGPU=$NGPU (set NGPU=1 or NGPU=4)"; exit 1 ;;
esac

if [ "$NGPU" = "1" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python command not found: $PYTHON"
    echo "hint: use the cloud image's activated environment, or set PYTHON=/path/to/python"
    exit 1
fi

if [ "$NGPU" != "1" ]; then
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
fi

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"

# tokenizer 必须和 base checkpoint 完全匹配，否则加载时会因为 vocab_size 不一致失败。
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "error: missing tokenizer files in $TOKENIZER_DIR"
    echo "hint: copy the tokenizer from the base training run or run the same tokenizer preparation step first"
    exit 1
fi

# 这里只检查模型权重和 meta。SFT 默认 LOAD_OPTIMIZER=0，所以不要求 base optimizer 存在。
if ! ls "$BASE_CKPT_DIR"/model_*.pt >/dev/null 2>&1 || ! ls "$BASE_CKPT_DIR"/meta_*.json >/dev/null 2>&1; then
    echo "error: missing d34 base checkpoint in $BASE_CKPT_DIR"
    echo "hint: expected model_*.pt and meta_*.json under base_checkpoints/$MODEL_TAG"
    exit 1
fi

mkdir -p "$NANOCHAT_BASE_DIR"
# 身份对话数据很小，用来让 SFT 后的模型稳定回答“你是谁”等身份问题。
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo "============================================================"
echo " nanochat d34 SFT on A800"
echo " NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo " BASE_CKPT_DIR=$BASE_CKPT_DIR"
echo " SFT_CKPT_DIR=$SFT_CKPT_DIR"
echo " PYTHON=$PYTHON  TORCHRUN=$TORCHRUN  NGPU=$NGPU  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE  LOAD_OPTIMIZER=$LOAD_OPTIMIZER"
echo " WANDB_RUN=$WANDB_RUN  COMPILE=$COMPILE  OPT_COMPILE=$NANOCHAT_COMPILE_OPTIMIZER  SKIP_VAL_BPB=$SKIP_VAL_BPB  CHATCORE_EVERY=$CHATCORE_EVERY"
echo "============================================================"

# 正式启动 SFT。--model-tag d34 会让脚本明确加载 base_checkpoints/d34，
# 并把最终 SFT checkpoint 保存到 chatsft_checkpoints/d34。
# 单卡直接用 python 启动；多卡必须用 torchrun 初始化 DDP/NCCL。
SFT_ARGS=(
    --model-tag "$MODEL_TAG"
    --device-batch-size "$DEVICE_BATCH_SIZE"
    --load-optimizer "$LOAD_OPTIMIZER"
    --compile "$COMPILE"
    --eval-every "$EVAL_EVERY"
    --eval-tokens "$EVAL_TOKENS"
    --skip-val-bpb "$SKIP_VAL_BPB"
    --chatcore-every "$CHATCORE_EVERY"
    --chatcore-max-cat "$CHATCORE_MAX_CAT"
    --chatcore-max-sample "$CHATCORE_MAX_SAMPLE"
    --run "$WANDB_RUN"
)

if [ "$NGPU" = "1" ]; then
    "$PYTHON" -m scripts.chat_sft "${SFT_ARGS[@]}" $SFT_EXTRA_ARGS
else
    "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="$NGPU" -m scripts.chat_sft -- "${SFT_ARGS[@]}" $SFT_EXTRA_ARGS
fi

echo ""
echo "SFT checkpoint files:"
ls -lh "$SFT_CKPT_DIR"

# 训练结束后做一个有限题量的评测，确认模型能加载、能跑通 ChatCORE 任务路径。
# 想节省时间可以 RUN_CHAT_EVAL=0 跳过。
if [ "$RUN_CHAT_EVAL" = "1" ]; then
    if [ "$NGPU" = "1" ]; then
        "$PYTHON" -m scripts.chat_eval \
            -i sft \
            -g "$MODEL_TAG" \
            -b "$CHAT_EVAL_BATCH_SIZE" \
            -x "$CHAT_EVAL_MAX_PROBLEMS"
    else
        "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="$NGPU" -m scripts.chat_eval -- \
            -i sft \
            -g "$MODEL_TAG" \
            -b "$CHAT_EVAL_BATCH_SIZE" \
            -x "$CHAT_EVAL_MAX_PROBLEMS"
    fi
fi

# 最后用最终 SFT 模型回答一个提示词，作为最直接的 smoke test。
"$PYTHON" -m scripts.chat_cli \
    -i sft \
    -g "$MODEL_TAG" \
    -p "$PROMPT"

echo ""
echo "Final chat model is in: $SFT_CKPT_DIR"