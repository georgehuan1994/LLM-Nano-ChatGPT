# LLM Nano ChatGPT

这是 Andrej Karpathy (OpenAI 创始成员、特斯拉 AI 前总监) 写的全栈 ChatGPT 极简复刻，目标是把整个 LLM 流程压缩成几千行可读代码。

它不是一个 SDK 或框架，而是一个端到端的训练流水线：从原始文本 → 分词器 → 预训练 → 监督微调 (SFT) → 强化学习 (RL) → 网页 UI，全部包含。哲学是「最小可读、可全部 fork、不要配置地狱」。

## 目录速览

### nanochant/..

|文件|作用|
|----|----|
|[gpt.py](nanochat\gpt.py)|GPT Transformer 模型本体|
|[tokenizer.py](nanochat\tokenizer.py)|BPE 分词器 (GPT-4 风格) |
|[engine.py](nanochat\engine.py)|推理引擎 (KV Cache 加速) |
|[optim.py](nanochat\optim.py)|优化器 (AdamW + Muon) |
|[dataloader.py](nanochat\dataloader.py)|数据加载 (分布式)|
|[base_train.py](scripts\base_train.py)|预训练主循环 (前向+反向) |

### scripts/..

|文件|作用|
|----|----|
|[chat_sft.py](scripts\chat_sft.py)|监督微调 (教模型对话格式)|
|[chat_rl.py](scripts\chat_rl.py)|强化学习 (GRPO 类似 RLHF)|
|[chat_cli.py](scripts\chat_cli.py)|命令行聊天|
|[chat_web.py](scripts\chat_web.py)|FastAPI Web UI|

### runs/..

|文件|作用|
|----|----|
|[speedrun.sh](runs\speedrun.sh)|完整训练流水线 (8×H100, ~3小时)|
|[runcpu.sh](runs\runcpu.sh)|CPU/Mac 教学示例|

## 4 个训练阶段

```
原始网页文本 (ClimbMix dataset)
        │
        ▼  ① Tokenizer Train  (tok_train.py)
  BPE 分词器 (vocab=32768)
        │
        ▼  ② Pretrain  (base_train.py) ← 占 99% 算力
  Base Model：会"续写"，不会"对话"
        │
        ▼  ③ SFT  (chat_sft.py)
  Chat Model：学会 <|user_start|>...<|assistant_start|> 格式
        │
        ▼  ④ RL  (chat_rl.py，可选)
  对齐模型，提升数学/推理
  ```

每一阶段产生的 checkpoint 分别保存在：

- `~/.cache/nanochat/base_checkpoints/`
- `~/.cache/nanochat/chatsft_checkpoints/`
- `~/.cache/nanochat/chatrl_checkpoints/`

CLI 或 WebUI 默认加载 `sft` 版本。

## PyTorch / 神经网络

PyTorch 入门可以配合官方 60 分钟教程  —  [Deep Learning with PyTorch: A 60 Minute Blitz — PyTorch Tutorials 2.12.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，重点搞懂 Tensor、autograd、nn.Module 三个概念，回来读 gpt.py 就基本无障碍了。

1. `nn.Module` — 所有网络层的基类。[gpt.py](nanochat\gpt.py) 里 `class GPT(nn.Module)` 就是模型，里面 `forward(x)` 定义了 "输入 token → 输出概率分布" 的计算图。
2. `Tensor`  — 张量，即一个多维数组 (类似 numpy.ndarray，但能在 GPU 上跑、能自动求导)。
3. 训练循环：

    ```py
    loss = model(x, y)          # 前向：算预测和真实差距
    loss.backward()             # 反向：自动算每个参数的梯度
    optimizer.step()            # 用梯度更新参数
    optimizer.zero_grad()       # 清掉梯度，准备下一轮
    ```

4. KV Cache  — 推理时的加速技巧：生成第 N 个 token 不用重算前 N-1 个的 attention，缓存起来即可。

## 运行

> Windows 用户推荐通过 **Git Bash** (随 Git for Windows 安装) 或 **WSL2** 来执行 `.sh` 脚本。下面所有命令在 Git Bash 中可直接运行；如果你想用 PowerShell，把行尾的 `\` 换成反引号 `` ` ``、并把环境变量语法换成 `$env:XXX="..."` 即可。

### 环境准备 (Windows / Linux / macOS 通用)

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境并安装依赖
uv venv
uv sync --extra cpu        # 没有 NVIDIA GPU 选 cpu
# uv sync --extra gpu      # 有 CUDA GPU 选 gpu (会装 CUDA 12.8 版 PyTorch)

# 3. 激活虚拟环境
# Windows Git Bash:
source .venv/Scripts/activate
# Linux / macOS:
# source .venv/bin/activate
```

> 中间产物 (数据、tokenizer、checkpoint) 默认放在 `~/.cache/nanochat/`
>
> 在 Windows 上即 `C:\Users\<YOURNAME>\.cache\nanochat\`。如果想改位置，先 `export NANOCHAT_BASE_DIR=/your/path`。

### CPU 训练路线

仓库中专门为 CPU 准备的小规模教学训练，会训出一个非常笨但能跑通流程的小模型。

- 训练一个 6 层 (脚本里写的是 `--depth=6`) 的极小 GPT
- 训练数据缩小、序列长度缩小、迭代次数缩小
- 在 M3 Max MacBook 上约 30 分钟预训练 + 10 分钟 SFT；在普通 Windows CPU 上估计要几小时，但能跑完
- 训完后 `python -m scripts.chat_cli` 就能聊天 (当然回答会很傻)

完整脚本见 [runs/runcpu.sh](runs/runcpu.sh)，在 Git Bash 中可一键执行：

```bash
bash runs/runcpu.sh
```

如果想分步看每个阶段的输出，按下面的顺序手动执行：

```bash
# ① 下载预训练数据 (8 个 shard ≈ 800MB 文本)
python -m nanochat.dataset -n 8

# ② 训练 BPE 分词器 (vocab=32768)
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# ③ 预训练 base model (5000 步，CPU 上耗时最久的一步)
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
    --run=dummy

# ④ 评测 base model
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# ⑤ 下载 SFT 用的身份对话数据 (2.3MB)
curl -L -o "$HOME/.cache/nanochat/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# ⑥ SFT 微调 (教模型 <|user_start|>...<|assistant_start|> 对话格式)
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=dummy

# ⑦ 聊天！
python -m scripts.chat_cli -p "What is the capital of France?"
# 或者起一个 ChatGPT 风格的 Web UI (默认端口 8000)
python -m scripts.chat_web
```

> **Windows 注意事项**
> - 多进程 dataloader 可能在 Windows 上报 `freeze_support` 错误，遇到时把 `--device-batch-size` 调到 1 或在脚本入口用 `if __name__ == "__main__":` 包裹 (本仓库主要在 Linux 上测试)。
> - 如果显存/内存不够，依次把 `--device-batch-size` 从 32 降到 16、8、4、2、1。
> - 如果 `curl` 不可用，可在浏览器中打开 URL 手动下载到对应路径，或使用 PowerShell 的 `Invoke-WebRequest`。

### GPU 训练路线

完整流水线见 [runs/speedrun.sh](runs/speedrun.sh)，目标是在 **8×H100** 节点上 **~3 小时**训出一个 GPT-2 (1.6B) 级别的模型，云上成本约 \$48-72。

```bash
# 一键跑完整 pipeline (8×H100，建议放到 screen / tmux 里)
bash runs/speedrun.sh

# 配合 wandb 记录训练曲线
WANDB_RUN=d24 bash runs/speedrun.sh
```

关键步骤拆解：

```bash
# ① 下载更多数据 shard (≈ 170 个，GPT-2 等级训练所需)
python -m nanochat.dataset -n 170

# ② 训练 tokenizer
python -m scripts.tok_train
python -m scripts.tok_eval

# ③ 预训练 d24 model (8 卡分布式，DDP via torchrun)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN

# ④ 评测 base model (CORE 分数 / bits-per-byte / 采样)
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# ⑤ SFT
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 --run=$WANDB_RUN

# ⑥ 评测 chat model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# ⑦ (可选) 强化学习进一步对齐
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN

# ⑧ 启动 Web UI，多 GPU 数据并行推理
python -m scripts.chat_web --num-gpus 8
```

> **单卡 / 小显存 GPU**
> - 单卡用户：去掉 `torchrun --standalone --nproc_per_node=8 --` 前缀，直接 `python -m scripts.base_train ...` 即可，代码会自动切换到梯度累积，结果几乎一样，只是要等 8 倍时间。
> - 显存 <80GB：把 `--device-batch-size` 从默认 32 → 16 → 8 → 4 → 2 → 1 逐级下调直到不 OOM。
> - 想要更小更快的实验模型 (~5 分钟预训练)：`--depth=12` 是作者最爱的研究规模 (GPT-1 大小)。

### 加载已训练的模型

每次训练完成后，三个阶段的 checkpoint 分别存在：

| 阶段 | 路径 | `--source` 参数 |
|------|------|----------------|
| 预训练   | `~/.cache/nanochat/base_checkpoints/`    | `base` |
| SFT 微调 | `~/.cache/nanochat/chatsft_checkpoints/` | `sft` (默认) |
| RL 对齐  | `~/.cache/nanochat/chatrl_checkpoints/`  | `rl`  |

```bash
# 默认加载最大的 SFT 模型
python -m scripts.chat_cli

# 指定阶段、模型 tag、训练 step
python -m scripts.chat_cli -i sft -g d6 -s 1500
python -m scripts.chat_cli -i base -p "Once upon a time"   # 让 base 模型自由续写

# 起 Web UI (浏览器访问 http://localhost:8000)
python -m scripts.chat_web
```

> ⚠️注意：仓库**没有提供官方预训练 checkpoint 下载**，必须自己跑完上面的训练流程后才能 chat。如果只是想体验"和 LLM 聊天"，可以用 `transformers` 库直接加载 HuggingFace 上的开源 chat 模型 (如 `Qwen/Qwen2.5-0.5B-Instruct`)，但那不在本仓库范围内。
