#!/bin/bash

# 这个脚本用于训练一个达到 GPT-2 水平的 LLM（包含预训练 + 微调）
# 它被设计为在一台“干净的” 8 卡 H100 GPU 节点上运行，整个流程大约耗时 3 小时
# 注意：这是面向 GPU 服务器的“正式速通”脚本；如果你只有 CPU/MacBook，请改用 runcpu.sh

# 三种典型启动方式：
# 1) 最简单的方式：直接运行
#    bash runs/speedrun.sh
# 2) 在 screen 会话中运行（因为整个跑完需要约 3 小时，避免 SSH 断开导致中断）
#    -L 表示开启日志记录；-Logfile 指定日志文件路径；-S 给会话起一个名字方便重连
#    screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) 在带 wandb 日志记录的情况下运行（wandb 配置见下方说明）
#    WANDB_RUN=speedrun 这种写法表示给本次脚本注入一个环境变量，作为 wandb 运行名
#    WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# 设置 NANOCHAT_BASE_DIR 环境变量，指定 nanochat 的数据、tokenizer、checkpoint 等中间产物的存储位置
# 默认情况下 nanochat 会把数据放到 ~/.cache/nanochat；这里改放到当前项目目录下的 .nanochat 子目录
# 这样所有产物都集中在仓库目录里，便于统一查看、备份和清理
# OMP_NUM_THREADS=1 限制 OpenMP 只使用 1 个线程，避免在多进程数据加载时
# 因为线程数过多互相争抢 CPU 反而拖慢速度（PyTorch 多进程训练的常见做法）
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$PWD/.nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 使用 uv 配置 Python 虚拟环境
# uv 是一个非常快的 Python 包管理工具，类似 pip + venv 的现代替代品

# 如果系统中没有安装 uv，就通过官方安装脚本进行安装
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
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
python -m nanochat.dataset -n 170 &
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
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# 评估 base model：包括 CORE 综合指标、训练/验证集上的 BPB（Bits Per Byte），并采样生成一些文本
# 同样使用 8 卡并行评估，加快速度
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

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
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
# 评估 SFT 后的对话模型；-i sft 表示加载“sft 阶段”产出的 checkpoint 来评估
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

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
