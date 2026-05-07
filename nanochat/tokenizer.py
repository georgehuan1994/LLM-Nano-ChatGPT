"""
这个模块实现了 nanochat 项目里使用的 BPE tokenizer。

可以先把 tokenizer 理解成：
“负责把原始文本切成 token id，再把 token id 还原成文本”的那层基础设施。

它采用的是 GPT 风格的 BPE (Byte Pair Encoding) 思路，核心目标是：
1. 让高频文本片段尽量以更紧凑的 token 形式出现；
2. 兼容普通自然语言、代码、符号、中文等多种文本分布；
3. 为后续训练、评估、聊天渲染提供统一的编码/解码接口。

这个文件里同时保留了两套实现：
1. `HuggingFaceTokenizer`
   使用 HuggingFace `tokenizers` 库，既能训练也能推理，适合理解标准流程。
2. `RustBPETokenizer`
   训练时使用项目自己的 `rustbpe`，推理时使用 `tiktoken`，更贴近项目实际使用路径。

简单说：
- 第一套实现更“通用框架化”；
- 第二套实现更“项目实战化”。
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # 每篇文档都会以 BOS(Beginning of Sequence) 开头，
    # 用来告诉模型：这里是一个新序列/新文档的起点。
    "<|bos|>",
    # 下面这些 special token 主要用于聊天微调阶段。
    # 它们不是普通文本的一部分，而是 “对话结构标记”：
    # 哪一段是 user，哪一段是 assistant，哪里开始/结束工具调用等。
    "<|user_start|>", # 用户消息开始
    "<|user_end|>",
    "<|assistant_start|>", # assistant 消息开始
    "<|assistant_end|>",
    "<|python_start|>", # assistant 调用 python 工具
    "<|python_end|>",
    "<|output_start|>", # python 工具输出开始
    "<|output_end|>",
]

# 这是 tokenizer 在正式做 BPE merge 之前的预切分 regex
#
# 直觉上可以把它理解成：
# 先把原始文本切成一批比较合理的小段，再在这些小段内部继续做字节级 BPE 合并。
#
# 这样做的好处是：
# 1. 不会完全无边界地在整段文本上乱合并；
# 2. 可以更自然地处理单词、数字、空白符、标点、换行等不同类型片段；
# 3. 更接近 GPT 系 tokenizer 的常见工程做法。
#
# 这里和 GPT-4 风格实现有一个细小差异：
# 数字片段使用的是 \p{N}{1,2}，而不是更常见的 \p{N}{1,3}。
# 作者的考虑是：在 32K 左右的小词表里，不希望在数字片段上花掉太多 token 容量。
# 从项目经验上看，2 是一个比较折中的设置。
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# 基于 HuggingFace `tokenizers` 的 GPT 风格 tokenizer 实现
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """对 HuggingFace Tokenizer 的一层封装，按项目规则统一接口。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # 从 HuggingFace 上的现成 tokenizer 初始化，例如 "gpt2"。
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # 从本地目录读取已经保存好的 tokenizer.json。
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 从一个文本迭代器训练 tokenizer。
        #
        # 这里的 text_iterator 会持续产出字符串，训练器边读边统计，
        # 不需要一次把所有训练文本都装进内存。

        # 1. 配置 HuggingFace tokenizer 的底层模型为 BPE。
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # 遇到未覆盖片段时可以退回到字节级表示，这是很关键的兜底能力。
            unk_token=None,
            fuse_unk=False,
        ))
        # 2. 不额外做 normalizer。
        #    也就是说，尽量保留原文本的原貌，不在训练前偷偷改大小写或做 Unicode 归一化。
        tokenizer.normalizer = None

        # 3. 配置 pre-tokenizer。
        #    先按 GPT 风格 regex 做一次“粗切分”，
        #    再进入 ByteLevel，把文本稳定映射到字节级可处理表示。
        #
        # 这里之所以先 Split 再 ByteLevel，是因为：
        # - Split 更像按语言结构做分段
        # - ByteLevel 更像把每段变成稳定可编码的底层单位
        gpt4_split_regex = Regex(SPLIT_PATTERN) # HuggingFace 这里要求显式包一层 Regex 对象。
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])

        # 4. Decoder 与 ByteLevel pre-tokenizer 配套。
        #    编码时怎么拆，解码时就怎么合回来。
        tokenizer.decoder = decoders.ByteLevel()

        # 5. 不额外做 post-processing。
        tokenizer.post_processor = None

        # 6. 配置 BPE 训练器。
        #    initial_alphabet 使用 ByteLevel 默认字母表，保证基础字节层单位都可表示。
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # 不设最小频次门槛，让训练器自己在 vocab_size 约束下学习。
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )

        # 7. 正式开始训练。
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # 编码单条字符串。
        #
        # prepend / append 可以传：
        # 1. 一个 special token 字符串，例如 "<|bos|>"
        # 2. 一个已经确定好的 token id
        #
        # num_threads 这里不会用到，它只是为了和项目另一套 tokenizer 接口保持一致。
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # 通过“精确匹配”编码一个 special token。
        # 注意 special token 不是普通文本切分出来的，而是保留控制符。
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # 不同 HuggingFace tokenizer 对“序列起始 token”的命名并不统一。
        # 所以这里做一个兼容层：
        # 1. 优先找 nanochat 风格的 <|bos|>
        bos = self.encode_special("<|bos|>")
        # 2. 如果没有，再尝试很多老模型常用的 <|endoftext|>
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3. 如果还找不到，宁可直接报错，也不要默默返回一个无效值。
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        # 为了统一项目接口，允许同时编码：
        # - 单个字符串
        # - 字符串列表
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        # skip_special_tokens=False 的意思是：
        # 如果 ids 里包含 <|bos|> 等特殊 token，也按原样解码出来。
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # 将 tokenizer 保存到磁盘，便于后续复用。
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# 基于 rustbpe + tiktoken 的 tokenizer 实现
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """
    项目默认使用的 tokenizer 封装。

    设计思路是：
    - 训练阶段：用 rustbpe 学分词规则
    - 推理/编码阶段：用 tiktoken 高效执行 encode/decode

    这样做的好处是：
    1. 训练规则时更灵活；
    2. 真正大量编码文本时速度更好；
    3. 整体接口仍然保持项目内统一。
    """

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1. 先用 rustbpe 训练分词规则。
        tokenizer = rustbpe.Tokenizer()
        # special token 不参与普通 BPE 训练，而是在后面单独插入词表。
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        # 2. 把 rustbpe 训练得到的 mergeable ranks 转换成 tiktoken 可用的 Encoding。
        #    也就是说：训练规则归训练规则，但真正执行编码/解码时走 tiktoken。
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # 映射：token 对应的字节串 -> merge 优先级 / rank
            special_tokens=special_tokens, # 映射：special token 字符串 -> token id
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # 从本地目录读取已经保存好的 tiktoken Encoding。
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # 从 tiktoken 的预置编码器加载，例如 "gpt2" / "cl100k_base"。
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken/很多早期 GPT tokenizer 会把这个特殊 token 命名为 "<|endoftext|>"。
        # 这名字看起来像“文本结束”，但工程上它又经常被放在序列开头，作为一个边界标记。
        #
        # 为了减少概念混乱，nanochat 统一把它理解成 "<|bos|>"：
        # beginning of sequence，序列开始标记。
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        # encode_single_token 会要求传入的必须是词表里真实存在的单个 special token。
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text 可以是：
        # - 单个字符串
        # - 字符串列表
        #
        # prepend / append 常用于显式加上 <|bos|> 之类的边界 token。

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            # encode_ordinary 表示把普通文本编码成 token ids，
            # 不会自动把 special token 从文本里解析出来。
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # 这里是原地插入，规模不大时完全可接受。
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            # 批量编码时走 tiktoken 的 batch 接口，能更高效地利用多线程。
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # 把 tiktoken Encoding 对象序列化到磁盘。
        # 后续 get_tokenizer() 读取的就是这里生成的 tokenizer.pkl。
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        把一条聊天对话渲染成训练可用的 token 序列。

        这里输入的不是普通纯文本，而是结构化 conversation，例如：
        - user 说了什么
        - assistant 回了什么
        - assistant 是否调用过 python 工具
        - python 输出是什么

        返回两个同长度列表：
        - ids:   对应整条对话的 token id 序列
        - mask:  监督掩码。mask=1 表示这个位置属于 assistant 的目标输出，需要算训练损失；
                 mask=0 表示这个位置只是上下文条件，不直接监督。

        这一步非常关键，因为聊天训练不是“对整段文本每个 token 都一视同仁”。
        项目真正想让模型学会的是 assistant 应该如何回答，
        所以通常只对 assistant 的输出部分计算 loss。
        """
        # ids 和 mask 会同步增长，保持一一对应。
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # 有些数据集的第一条消息可能是 system。
        # 当前项目这里选择了一个较简单直接的处理方式：
        # 把 system 内容并入第一条 user 消息中。
        #
        # 这样后面仍然可以保持严格的 user / assistant / user / assistant 交替结构。
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation) # 避免原地修改调用方传入的数据。
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # 先取出这次渲染需要用到的所有 special token id。
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # 开始真正渲染对话。
        #
        # 整体结构大致会长成：
        # <|bos|>
        # <|user_start|> ... <|user_end|>
        # <|assistant_start|> ... <|assistant_end|>
        # ...
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # 项目假定消息严格交替：
            # 偶数位应该是 user，奇数位应该是 assistant。
            # 这里显式检查，是为了尽早发现数据格式问题。
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content 可能是：
            # 1. 一个普通字符串
            # 2. 一个由多个 part 组成的列表（例如 text + python tool call + python_output）
            content = message["content"]

            if message["role"] == "user":
                # 当前实现里，user 消息被要求是普通字符串。
                # 它只作为上下文条件，因此 mask 全部记为 0。
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                # assistant 的内容是模型真正要学习生成的部分，
                # 因此绝大多数位置会打上 mask=1。
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # 最普通的情况：assistant 直接返回一段文本。
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # 普通文本 part：需要监督。
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # assistant 的 python 工具调用文本也属于它自己的输出，
                            # 因此照样监督。
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python_output 不是 assistant “预测出来”的内容，
                            # 而是工具真正返回的结果。
                            # 因此这些 token 会作为上下文喂给模型，但不参与监督。
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # 最后做截断，防止超长对话把显存/内存撑爆。
        # 这里是直接按 token 长度裁掉尾部。
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """调试小工具：把 render_conversation 的结果可视化出来。"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        在 RL / completion 场景下，把对话渲染成“等待 assistant 继续补全”的形式。

        和 Chat SFT 不同，这里不需要返回 mask。
        我们只需要准备一段前缀，让模型知道：
        “上下文已经给你了，接下来该 assistant 开口了。”
        """
        # 这里要求最后一条消息本来就是 assistant，
        # 因为我们要把它拿掉，转成“等待 assistant 继续生成”的前缀状态。
        conversation = copy.deepcopy(conversation) # 避免修改原数据
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # 原地删掉最后一个 assistant 回复

        # 先按普通对话前缀渲染
        ids, mask = self.render_conversation(conversation)

        # 最后补一个 assistant_start，告诉模型：
        # “现在轮到 assistant 继续往后写了”。
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat 项目级便捷函数

def get_tokenizer():
    # 从项目 base_dir 下读取默认 tokenizer。
    # 这就是 tok_train.py 训练并保存下来的那一份。
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    # 读取 token_bytes.pt。
    #
    # 这个文件并不是 tokenizer 本体，而是：
    # token_id -> 该 token 对应多少个 UTF-8 字节
    #
    # 它的核心用途是给 BPB(bits per byte) 评估做归一化。
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
