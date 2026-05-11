#!/bin/bash

# SMOKE=1 GPU=a800 NGPU=1 tmux new -s smoke "bash runs/speedrun.sh 2>&1 | tee runs/speedrun.log"
# 之后用 tmux attach -t smoke 重连，Ctrl+B D 分离会话

# 这个脚本用于训练一个达到 GPT-2 水平的 LLM（包含预训练 + 微调）
# 它被设计为在一台“干净的” 8 卡 H100 GPU 节点上运行，整个流程大约耗时 3 小时
# 注意：这是面向 GPU 服务器的“正式速通”脚本；如果你只有 CPU/MacBook，请改用 runcpu.sh

# 三种典型启动方式：
# 1) 最简单的方式：直接运行
#    bash runs/speedrun.sh
# 2) 在 tmux 会话中运行（因为整个跑完需要约 3 小时，避免 SSH 断开导致中断）
#    tmux new -s speedrun "bash runs/speedrun.sh 2>&1 | tee runs/speedrun.log"
#    分离会话：Ctrl+B 然后按 D；重连会话：tmux attach -t speedrun
#    列出所有会话：tmux ls；杀死会话：tmux kill-session -t speedrun
# 3) 在带 wandb 日志记录的情况下运行（wandb 配置见下方说明）
#    WANDB_RUN=speedrun 这种写法表示给本次脚本注入一个环境变量，作为 wandb 运行名
#    WANDB_RUN=speedrun tmux new -s speedrun "bash runs/speedrun.sh 2>&1 | tee runs/speedrun.log"

# 设置 NANOCHAT_BASE_DIR 环境变量，指定 nanochat 的数据、tokenizer、checkpoint 等中间产物的存储位置
# 默认情况下 nanochat 会把数据放到 ~/.cache/nanochat；这里改放到当前项目目录下的 .nanochat 子目录
# 这样所有产物都集中在仓库目录里，便于统一查看、备份和清理
# OMP_NUM_THREADS=1 限制 OpenMP 只使用 1 个线程，避免在多进程数据加载时
# 因为线程数过多互相争抢 CPU 反而拖慢速度（PyTorch 多进程训练的常见做法）
export OMP_NUM_THREADS=1

# export NANOCHAT_BASE_DIR="$PWD/.nanochat"
export NANOCHAT_BASE_DIR="$HOME/autodl-fs/.nanochat"

mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# GPU 配置切换
# 通过两个环境变量组合选择硬件配置：
#   GPU   = h100 | a800     （默认 h100；决定是否启用 FP8）
#   NGPU  = 1 | 4 | 8       （默认 8；决定 torchrun 的 nproc_per_node）
#
# 常用组合：
#   bash runs/speedrun.sh                           # 默认 8×H100，启用 FP8（原始配置）
#   GPU=a800 NGPU=1 bash runs/speedrun.sh           # 1×A800，BF16
#   GPU=a800 NGPU=4 bash runs/speedrun.sh           # 4×A800，BF16
#   GPU=h100 NGPU=4 bash runs/speedrun.sh           # 4×H100，FP8
#
# 说明：A800 是 Ampere 架构（sm_80），硬件不支持 FP8，因此 GPU=a800 时强制关闭 FP8。
GPU=${GPU:-h100}
NGPU=${NGPU:-8}

case "$GPU" in
    h100) FP8_FLAG="--fp8" ;;
    a800) FP8_FLAG="" ;;
    *) echo "error: unsupported GPU=$GPU (must be h100 or a800)"; exit 1 ;;
esac

case "$NGPU" in
    1|4|8) ;;
    *) echo "error: unsupported NGPU=$NGPU (must be 1, 4, or 8)"; exit 1 ;;
esac

# 单卡 micro-batch 可以按需调整，OOM 时降到 8/4
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}

# -----------------------------------------------------------------------------
# 冒烟测试模式（SMOKE=1）
# 用极小的模型 + 极少的步数跑通整条 pipeline，验证环境/代码无误
# 适合首次上机、换硬件后、或修改了核心代码后做 sanity check
#
# 默认 SMOKE=0（完整训练）。设置 SMOKE=1 后会自动覆盖以下三项：
#   - 模型深度 d24 → d12（参数量约 1/4）
#   - base_train 步数：自动算（约 ~21k）→ 200
#   - chat_sft 步数：跑完整个数据集 → 50
#   - 数据下载：170 shards → 仅 8 shards（足够 d12+200 iters 训练）
#
# 用法示例（推荐组合）：
#   SMOKE=1 GPU=a800 NGPU=1 bash runs/speedrun.sh    # 单卡 A800 冒烟测试
SMOKE=${SMOKE:-0}

if [ "$SMOKE" = "1" ]; then
    DEPTH=12
    BASE_TRAIN_EXTRA="--num-iterations=200"
    SFT_TRAIN_EXTRA="--num-iterations=50"
    DATASET_SHARDS_BG=8       # 冒烟时不需要额外的后台下载
else
    DEPTH=24
    BASE_TRAIN_EXTRA=""       # 用 --target-param-data-ratio 自动计算步数
    SFT_TRAIN_EXTRA=""
    DATASET_SHARDS_BG=170
fi

echo "============================================================"
echo " GPU 配置: GPU=$GPU  NGPU=$NGPU  DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE  FP8_FLAG=$FP8_FLAG"
echo " 训练规模: SMOKE=$SMOKE  DEPTH=$DEPTH  DATASET_SHARDS_BG=$DATASET_SHARDS_BG"
echo "============================================================"

# -----------------------------------------------------------------------------
# 使用 uv 配置 Python 虚拟环境
# uv 是一个非常快的 Python 包管理工具，类似 pip + venv 的现代替代品

# 如果系统中没有安装 uv，就通过官方安装脚本进行安装
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# uv 默认装到 ~/.local/bin，但当前 shell 的 PATH 可能还没包含它（首次安装时尤其常见）
# 这里显式把 ~/.local/bin 加进 PATH，避免后续 uv 命令找不到
export PATH="$HOME/.local/bin:$PATH"

# -----------------------------------------------------------------------------
# 国内镜像加速（CN_MIRROR=1）
# 在 AutoDL/腾讯云/阿里云等国内服务器上，PyPI 和 PyTorch 官方源很慢
# 启用后会做两件事：
#   1) PyPI 走清华镜像（通过 UV_DEFAULT_INDEX 环境变量，无侵入）
#   2) PyTorch 走阿里云镜像（临时改写 pyproject.toml，脚本结束自动还原）
# 用法：
#   CN_MIRROR=1 SMOKE=1 GPU=a800 NGPU=1 bash runs/speedrun.sh
CN_MIRROR=${CN_MIRROR:-0}
if [ "$CN_MIRROR" = "1" ]; then
    echo "[CN_MIRROR] 启用国内镜像源"
    # 1) PyPI → 清华
    export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
    # 2) PyTorch → 阿里云（临时改 pyproject.toml + 退出钩子还原）
    if ! grep -q "mirrors.aliyun.com/pytorch-wheels" pyproject.toml; then
        cp pyproject.toml pyproject.toml.bak
        sed -i \
            -e 's|https://download.pytorch.org/whl/cu128|https://mirrors.aliyun.com/pytorch-wheels/cu128|g' \
            -e 's|https://download.pytorch.org/whl/cpu|https://mirrors.aliyun.com/pytorch-wheels/cpu|g' \
            pyproject.toml
        # trap：无论脚本正常结束、报错、还是 Ctrl+C，都把原文件还原
        # 避免 git 状态被污染，也避免下次跑时镜像 URL 已经在文件里
        trap 'mv pyproject.toml.bak pyproject.toml 2>/dev/null && echo "[CN_MIRROR] 已还原 pyproject.toml" || true' EXIT
    fi
fi

# 如果当前目录下还没有 .venv，就用 uv 创建一个本地 Python 虚拟环境
[ -d ".venv" ] || uv venv
# 根据 pyproject.toml/uv.lock 安装项目依赖
# --extra gpu 表示安装 GPU 版本的额外依赖（包含 CUDA 相关包，例如 GPU 版的 PyTorch）
# 如果是 CPU 环境，应该改用 --extra cpu（参见 runcpu.sh）
uv sync --extra gpu
# 激活虚拟环境，让后续 python 命令使用 .venv 里的 Python 和依赖，而不是系统 Python
# Windows 上 uv/venv 通常生成 .venv/Scripts/activate；类 Unix 环境通常是 .venv/bin/activate
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "error: no activation script found in .venv"
    exit 1
fi

# -----------------------------------------------------------------------------
# wandb（Weights & Biases）配置
# wandb 是一个机器学习实验跟踪平台，可以记录 loss 曲线、评估指标、生成样例等，非常方便（强烈推荐）
# 如果你想使用 wandb 记录训练过程：
# 1) 先在终端登录 wandb：
#    `wandb login`
# 2) 运行脚本时通过环境变量指定本次运行的名字，例如：
#    `WANDB_RUN=d26 bash speedrun.sh`
# 如果用户没有提前设置 WANDB_RUN，就默认设为 dummy
# dummy 在代码里被作为特殊值处理，会跳过真正的 wandb 上传，等价于“关闭日志记录”
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 在整个训练过程中，脚本会持续向 NANOCHAT_BASE_DIR/report/ 目录写入 markdown 格式的报告片段
# 下面这条命令会清空旧的 report 目录，并写入一个新的“开头部分”
# 内容包括系统信息（CPU/GPU/内存/驱动版本等）和一个时间戳，标记本次运行的开始
# 训练全部跑完后，会把所有片段拼成一份完整的 report.md
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 训练分词器 Tokenizer

# 下载预训练数据集的前约 20 亿（2B）个字符
# 每个数据分片 shard 大约包含 2.5 亿（250M）字符
# 因此 2e9 / 250e6 = 8 个分片
# 每个 shard 解压前约 100MB（压缩文本），8 个 shard 大约 800MB 数据落到磁盘
# 数据是如何准备的可以查看 dev/repackage_data_reference.py
python -m nanochat.dataset -n 8
# 紧接着在“后台”继续下载更多分片，让下载和 tokenizer 训练并行进行，节省总时间
# 想达到 GPT-2 水平的预训练大约需要 150 个分片，这里多下载 20 个用于评估/缓冲，共 170 个
# 整个数据集最多有 6542 个分片可供下载
# `&` 把命令放到后台运行；$! 是 bash 内置变量，表示“最近一个后台进程的 PID”
# 后续我们会用 `wait $DATASET_DOWNLOAD_PID` 等待它结束
python -m nanochat.dataset -n $DATASET_SHARDS_BG &
DATASET_DOWNLOAD_PID=$!

# 训练 tokenizer：词表大小为 2**15 = 32768，使用约 20 亿字符的数据
# 词表越大，每个 token 能携带的信息越多，但 embedding 矩阵也越大
python -m scripts.tok_train
# 评估 tokenizer：报告压缩率、编码/解码是否正常等指标，帮助判断分词器是否可用
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model（基础语言模型）预训练
# base model 学的是“根据前文预测下一个 token”，还不是聊天助手

# 在开始训练之前，必须等待后台的数据集下载任务完成，否则训练会因数据不足而出错
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# 训练一个 d24 模型（depth=24，即 24 层 Transformer）
# 这里特意把模型“略微欠训练”以超越 GPT-2：
# 把 数据量:参数量 比例从 compute-optimal（计算最优）的默认 10.5 降低到 8
# 这一比例来自 Chinchilla 论文，表示“给定算力下应该训多少 token 才最划算”
# 降低这个比例意味着相对参数规模我们用了更少的数据，单步训练更快但训练不充分
# torchrun 是 PyTorch 官方的分布式启动器
# --standalone：单机模式（不需要外部 rendezvous server）
# --nproc_per_node=8：在每个节点（这里只有 1 个节点）启动 8 个进程，对应 8 张 H100 GPU
# 注意 `--` 之后的参数才会被传递给 base_train 脚本本身
# 关键超参解释：
# --depth=24：Transformer 层数为 24
# --target-param-data-ratio=8：参数与数据的目标比例为 8（详见上面注释）
# --device-batch-size=16：单卡一次处理 16 条序列
# --fp8：开启 FP8 低精度训练，H100 原生支持，速度大幅提升、显存占用更少
# --run=$WANDB_RUN：指定 wandb 运行名；dummy 表示不上传日志
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- --depth=$DEPTH --target-param-data-ratio=8 --device-batch-size=$DEVICE_BATCH_SIZE $FP8_FLAG $BASE_TRAIN_EXTRA --run=$WANDB_RUN
# 评估 base model：包括 CORE 综合指标、训练/验证集上的 BPB（Bits Per Byte），并采样生成一些文本
# 同样使用 $NGPU 卡并行评估，加快速度
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_eval -- --device-batch-size=$DEVICE_BATCH_SIZE

# -----------------------------------------------------------------------------
# SFT 监督微调（Supervised Fine-Tuning）
# 这一步教模型学会“对话相关的特殊 token、工具调用、多选题”等格式
# 直观地说，就是把 base model 微调成一个像 ChatGPT 那样能进行多轮对话的助手

# 下载约 2.3MB 的“合成身份对话”数据，用来给 nanochat 注入一个固定的“身份/人格”
# 例如让它在被问到“你是谁”时回答“我是 nanochat”而不是其他模型名
# 数据是如何合成的，以及如何按需修改身份，可以查看 dev/gen_synthetic_data.py
# curl -L：自动跟随 HTTP 跳转；-o 指定下载结果保存的本地路径
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 启动 SFT 训练，并在训练后评估
# device-batch-size=16 是单卡批大小；其他超参在脚本里有合理默认值
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- --device-batch-size=$DEVICE_BATCH_SIZE $SFT_TRAIN_EXTRA --run=$WANDB_RUN
# 评估 SFT 后的对话模型；-i sft 表示加载“sft 阶段”产出的 checkpoint 来评估
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_eval -- -i sft

# 通过命令行和模型进行交互式聊天！
# 不带 -p 参数时进入交互模式，可以连续多轮对话；带 -p 时是一次性提示词
# 取消下一行开头的 # 即可运行
# python -m scripts.chat_cli -p "Why is the sky blue?"

# 更进一步，通过类似 ChatGPT 风格的漂亮 WebUI 与你的模型对话
# 取消下一行开头的 # 即可启动网页服务，然后在浏览器中打开它提示的本地地址
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 把整个流程中各阶段产出的 markdown 片段拼接成一份完整的报告
# 输出文件是 report.md，会被复制到当前目录方便查看
# 这份报告里通常包含：系统信息、数据规模、tokenizer 评估、base 模型评估、SFT 模型评估等全部结果
python -m nanochat.report generate
