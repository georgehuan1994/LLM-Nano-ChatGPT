"""
使用项目内置的 BPE Tokenizer 库训练一个分词器。

1. 先读取大量训练文本；
2. 统计哪些字节片段经常一起出现；
3. 把这些高频片段合并成 token；
4. 最终得到一个适合本项目语料的 tokenizer。

整体风格参考 GPT 系列常见的 BPE tokenizer。
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# 解析命令行参数
#
# 这几个参数决定了拿多少数据来训练 tokenizer，以及最终词表有多大。
# - max_chars: 最多看多少字符，控制训练数据总量
# - doc_cap: 每篇文档最多截取多少字符，避免超长文档占比过高
# - vocab_size: 词表大小，也就是最终 token id 的总数

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# 构造训练文本迭代器
#
# tokenizer 训练通常不需要一次性把所有文本都读进内存
# 更常见的做法是边读边喂给训练器，也就是使用 iterator
# 这样可以节省内存，也更适合大规模语料

def text_iterator():
    """
    这个生成器做了三件事：

    1. 把按 batch 读取的数据，展开成逐篇文档的文本流；
    2. 每篇文档最多保留 args.doc_cap 个字符，避免极长文档主导训练；
    3. 当累计字符数超过 args.max_chars 时停止。

    注意：这里限制的是字符数，不是 token 数。
    这样做的目的只是控制 tokenizer 训练成本。
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                # 对单篇文档做截断，避免少数超长样本影响统计分布。
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            # 每次 yield 一篇文本，供 tokenizer 训练器持续消费。
            yield doc_text
            if nchars > args.max_chars:
                return

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# 训练 tokenizer
#
# RustBPETokenizer.train_from_iterator 会从文本流中统计高频字节片段，
# 然后逐步执行 BPE merge，直到词表大小达到 args.vocab_size。
#
# 这里记录训练耗时，方便后续观察不同数据量、不同词表大小的成本差异。
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# 把训练好的 tokenizer 保存到磁盘
#
# 后续训练 base model、做评估、启动聊天脚本时，都会复用这里保存的 tokenizer。
# 所以 tokenizer 通常是一个“先离线训练好，再在整个项目中复用”的组件。
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# 做一个快速的编码/解码自检
#
# 一个最基本的正确性要求是：
# 原始文本 -> encode -> token ids -> decode -> 还原文本
# 最终结果必须和原文完全一致。
#
# 这里故意混合了：
# - 英文
# - 数字
# - 缩写
# - 特殊符号
# - Unicode / 中文 / emoji
# 用来检查 tokenizer 是否能稳定处理常见字符类型。
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# 额外缓存一个 token_id -> token 字节数 的映射表，为后续评估准备元数据
#
# 这是本脚本里一个很重要、但初看不太直观的步骤。
#
# 为什么要做这件事？
# 因为项目后面会关心 bits per byte, 简写 BPB。
#
# 直觉上：
# - 普通的平均 loss 是 “每个 token 的平均损失”；
# - 但不同 tokenizer 的 token 粒度不同，token 数量不能直接横向比较；
# - 如果换成 “每个字节的平均信息量”，就更容易跨 tokenizer 比较。
#
# 为了高效计算 BPB，我们预先知道每个 token 对应多少个 UTF-8 字节。
# 这样在评估时就不需要反复现算。
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())

# 对词表中的每个 token id，单独 decode 成字符串。
# 这样可以知道这个 token 实际代表什么文本片段。
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # 这个 token 对应的 Python 字符串形式
    if token_str in special_set:
        # 特殊 token（例如控制符、保留符号）不对应真实文本内容，
        # 在 BPB 统计里通常不计入字节数。
        token_bytes.append(0)
    else:
        # UTF-8 下一个字符串占多少字节，才是 BPB 真正关心的长度。
        # 注意：字符数和字节数不一样，尤其是中文、emoji 往往占多个字节。
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)

token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# 记录训练摘要到 report 系统，方便后续统一查看实验结果。
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # 命令行参数快照，便于复现实验
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        # 这些统计量可以帮助我们直观看到：
        # 一个 token 平均覆盖多少字节，以及词表中 token 长度分布是否极端。
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
