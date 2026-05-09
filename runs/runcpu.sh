#!/bin/bash

set -e

# 注意：训练 LLM 需要 GPU 算力和一定成本在 MacBook 上很难训练出很好的效果
# 请把这个脚本理解为一个用于学习/体验的有趣演示，不要期待它能训练出很强的模型
# 你也可以手动运行此脚本：把下面的命令一条一条复制到终端中执行

# 设置 NANOCHAT_BASE_DIR 环境变量，指定 nanochat 的数据、tokenizer、checkpoint 等文件的存储位置
# 这里改为放到当前项目目录下的 .nanochat 子目录，便于统一查看、备份和删除
export NANOCHAT_BASE_DIR="$PWD/.nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# 目录示例：
# - 预训练数据：$NANOCHAT_BASE_DIR/base_data_climbmix
# - tokenizer 文件：通常也会放在这个基础目录下
# - checkpoint / 中间产物：通常也会放在这个基础目录下

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# 如果是刚安装的 uv，当前 shell 可能还没有加载 ~/.local/bin 到 PATH
# 这里主动加载一次，确保后续 uv venv / uv sync 在同一个会话里可用
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# 如果到这里仍然找不到 uv，就直接报错退出，避免后续命令连续失败
command -v uv &> /dev/null || {
    echo "error: uv is not available in PATH. Try: source \"$HOME/.local/bin/env\""
    exit 1
}

# 如果当前目录下还没有 .venv，就用 uv 创建一个 Python 虚拟环境
[ -d ".venv" ] || uv venv

# 根据 pyproject.toml/uv.lock 安装项目依赖
# --extra cpu 表示安装适合 CPU 运行的额外依赖，不安装 CUDA/GPU 专用依赖
uv sync --extra cpu

# 激活虚拟环境，让后续 python 命令使用 .venv 里的 Python 和依赖
# Windows 上的 uv/venv 通常会生成 .venv/Scripts/activate；类 Unix 环境通常是 .venv/bin/activate
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "error: no activation script found in .venv"
    exit 1
fi

# 如果用户没有提前设置 WANDB_RUN，就默认设为 dummy
# WANDB 是 Weights & Biases，用于记录训练日志；dummy 通常表示不真正上传日志
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# 准备预训练数据集
# -n 8 等价于 --num-files 8，表示下载 8 个训练数据分片 shard
# shard 可以理解为数据集被切成的很多小文件；这里为了 CPU 演示只下载 8 份，比较快但数据很少
# 代码里的默认 -n -1 表示下载全部可用训练分片；完整训练会更慢、更占空间
python -m nanochat.dataset -n 8

# 训练 tokenizer
# --max-chars=2000000000 表示最多用 20 亿个字符训练分词器；如果数据不够，就用已有数据
# 字符越多，分词器越稳定，但训练时间也越长
python -m scripts.tok_train --max-chars=2000000000

# 评估刚训练好的 tokenizer
# 检查压缩率、编码/解码是否正常等指标，帮助判断分词器是否可用
python -m scripts.tok_eval

# 训练一个小型 base model（基础语言模型）
# base model 学的是根据前文预测下一个 token，还不是聊天助手
# 作者把这组参数调到可以在 MacBook Pro M3 Max 上约 30 分钟完成
# 如果想得到更好的结果，可以尝试增大 num_iterations，或让你常用的 LLM 给出更多调参建议
# 下面参数含义：
# --depth=6：Transformer 层数为 6；层数越多，模型越强但越慢、越占内存
# --head-dim=64：每个注意力头的维度为 64；影响注意力层大小
# --window-pattern=L：注意力窗口模式L 表示 full attention，可以看完整上下文；S 表示滑动窗口注意力
# --max-seq-len=512：最大上下文长度为 512 个 token；模型每次最多看 512 个 token
# --device-batch-size=32：单个设备一次处理 32 条序列；太大可能内存不够
# --total-batch-size=16384：全局 batch 的 token 数为 16384；通常等于多次梯度累积后的总 token 数
# --eval-every=100：每训练 100 步做一次验证集评估
# --eval-tokens=524288：每次评估使用 524288 个 token
# --core-metric-every=-1：关闭 CORE 指标评估；-1 在本项目里常表示禁用某功能
# --sample-every=100：每训练 100 步让模型生成一些样例文本，方便观察效果
# --num-iterations=5000：训练 5000 个优化步骤；越大通常效果越好，但耗时更长
# --run=$WANDB_RUN：指定日志运行名；这里通常是 dummy，不上传真实 wandb 日志
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN

# 评估 base model
# --device-batch-size=1：评估时单设备 batch 为 1，省内存，适合 CPU 演示
# --split-tokens=16384：每个数据 split 用 16384 个 token 做 BPB/损失评估；越大越稳定但越慢
# --max-per-task=16：每个核心评测任务最多评 16 道题；这是为了让 CPU 演示更快
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# 下载一份用于 “身份/聊天风格” 的 SFT 数据
# curl -L：如果链接发生跳转，继续跟随跳转下载
# -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl：把下载结果保存到指定路径
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 监督微调 SFT（Supervised Fine-Tuning）
# SFT 的作用：把 base model 微调成更像“聊天助手”的模型
# --max-seq-len=512：SFT 阶段仍然使用 512 token 的最大上下文
# --device-batch-size=32：单设备一次处理 32 条序列
# --total-batch-size=16384：全局 batch 的 token 数为 16384
# --eval-every=200：每 200 步做一次验证评估
# --eval-tokens=524288：每次评估使用 524288 个 token
# --num-iterations=1500：SFT 训练 1500 个优化步骤
# --run=$WANDB_RUN：指定日志运行名；dummy 表示基本不使用 wandb 上传
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --load-optimizer=0 \
    --embedding-lr=0.001 \
    --unembedding-lr=0.00003 \
    --matrix-lr=0.00015 \
    --scalar-lr=0.0015 \
    --init-lr-frac=1.0 \
    --warmup-ratio=0.05 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# 通过命令行和模型聊天
# 训练后的模型应该能回答 “法国首都是巴黎”
# 它甚至可能知道 “天空是蓝色的”
# 有时你先对它说 Hi，再提问，模型表现可能会更好
# 取消下一行开头的 # 后即可运行
# -p / --prompt 表示给模型的一次性提示词；模型会基于这个问题生成回答
# python -m scripts.chat_cli -p "What is the capital of France?"

# 通过一个类似 ChatGPT 风格的漂亮 WebUI 和模型聊天
# 取消下一行开头的 # 后即可启动网页服务，然后在浏览器中打开对应地址
# python -m scripts.chat_web
