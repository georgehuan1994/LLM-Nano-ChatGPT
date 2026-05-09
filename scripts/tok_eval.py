"""
评估 tokenizer 的压缩效果。

这里的核心问题是：
同一段文本，被不同 tokenizer 编码后，会得到多少个 token？

如果一个 tokenizer 能用更少的 token 表示同样的文本，通常说明它的压缩率更高。
这往往意味着：
1. 在相同上下文窗口里能塞进更多原始文本；
2. 训练和推理时，每段文本对应的 token 序列更短；
3. 但也不代表它一定 “全面更好”，因为还要结合建模难度一起看。

这个脚本并不训练 tokenizer，也不评估语言模型本身。
它只回答一个更窄但很关键的问题：

“给定同一批文本，不同 tokenizer 会把它切成多长的 token 序列？”

所以可以把 tok_eval.py 理解为：
对 tok_train.py 产出的 tokenizer 做一次 “编码质量体检”。
"""

from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

# 一段英文新闻文本，用来测试英文自然语言的压缩效果。
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

# 一段韩文文本，用来测试非英文文本的压缩效果。
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 ‘정보의 균형’을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 ‘무엇이 쟁점인지’, ‘누가 무엇을 주장하는지’, ‘사실은 무엇인지’**를 명확히 전달하는 데 집중합니다.
""".strip()

# 一段代码文本，用来测试“程序代码”这种特殊分布文本的压缩效果。
# 代码里会有缩进、符号、关键字、变量名，分布和自然语言差异很大。
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # 原始 UTF-8 字节流
        ids = list(text_bytes) # 把字节流转成 0..255 的整数序列

        # 反复合并最常见的相邻字节对，逐步构造新 token
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # 统计每个相邻 pair 出现了多少次
            stats = get_stats(ids)
            # 找到出现次数最多的 pair
            pair = max(stats, key=stats.get)
            # 创建一个新 token，id 使用当前可用的下一个编号
            idx = 256 + i
            # 把序列里所有这个 pair 替换成新 token id
            ids = merge(ids, pair, idx)
            # 记录这次 merge 规则
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # 打印训练过程
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

# 一段数学 LaTeX 文本，用来测试公式、反斜杠、花括号等符号密集文本。
math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square’’—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere’s primary productivity.
""".strip()

# 再取一小部分训练集和验证集文本。
#
# 这样做有两个目的：
# 1. 看看 tokenizer 在“项目自己的真实数据”上压缩效果如何；
# 2. 对比它在见过的数据分布和没特别见过的数据分布上的表现差异。
#
# 注：这里的 train shard 来自训练语料，因此我们的 tokenizer 大概率已经 “见过类似分布”。
# 这里用 next(...) 只取一个 batch，是为了让评估脚本足够轻量：
# 它想做的是 “快速 sanity check”，不是一场大规模基准测试。
train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)

# 把所有待评估文本组织成一个列表，后面统一循环处理。
# 每个元素都是 (名字, 文本内容)。
# 这样后面无论是做 encode/decode 检查、算压缩率、打印表格，逻辑都能统一。
all_text = [
    ("news", news_text),
    ("korean", korean_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("fwe-train", train_text),
]
if val_text:
    all_text.append(("fwe-val", val_text))

# 分别评估三种 tokenizer：
# - gpt2: 经典 GPT-2 tokenizer
# - gpt4: 这里用 cl100k_base 近似代表 GPT-4 风格 tokenizer
# - ours: 当前项目自己训练出来的 tokenizer
#
# 这样可以回答一个很实际的问题：
# “我们自己训练的 tokenizer，在这些文本上压缩率到底比现成 tokenizer 好还是差？”
#
# 注意：
# 这里说的 “GPT-4 tokenizer” 其实不是某个闭源模型的完整内部实现，
# 而是用 OpenAI 生态里常见的 cl100k_base 编码器来做近似对比。
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # GPT-2 基础模型使用的 tokenizer
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # GPT-4 系常见的 cl100k_base tokenizer
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        # 先做 encode/decode 一致性检查，确保 tokenizer 至少是可逆的。
        # 如果一个 tokenizer 不能无损还原文本，那它在这个项目里就不合格。
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        # ratio = 原始字节数 / token 数
        # 这个值越大，表示 “每个 token 平均承载的原始字节越多”，
        # 通常可以理解为压缩率越好。
        #
        # 这里刻意使用 “字节数” 而不是 “字符数”：
        # 因为项目后面关心的是 BPB(bits per byte)，
        # 而 UTF-8 下中文、emoji、特殊符号的字节长度并不相同。
        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio
        }

# ANSI 颜色码，用于在终端里高亮显示谁的压缩效果更好。
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# 先打印词表大小。
# 词表大小会影响 tokenizer 的表达能力，也会影响模型输出层大小。
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """打印基线 tokenizer 与我们自己的 tokenizer 的对比表。"""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # 计算相对差异。
        # 这里用 token 数来比较：token 越少通常越好。
        #
        # relative_diff > 0: 我们的 tokenizer 用的 token 更少，压缩更好
        # relative_diff < 0: 我们的 tokenizer 用的 token 更多，压缩更差
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # 根据 ratio 判断谁压缩更好。
        # ratio 越高，表示每个 token 平均覆盖的字节越多。
        #
        # 这里同时打印 tokens 和 ratio，是为了避免只盯着单一数字：
        # - tokens 更直观，直接告诉你序列长度；
        # - ratio 更贴近“每个 token 装了多少原始信息”。
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")

# 分别和 GPT-2、GPT-4 风格 tokenizer 做对比。
print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

# 把评估结果整理成 markdown 表格，写入 report。
# 这样后续查看实验记录时，不需要重新跑脚本也能看到结果摘要。
#
# 注意这个 report 只是结果快照。
# 它保存的是评估统计，不会反向参与训练，也不会影响 tokenizer 本身。
from nanochat.report import get_report
lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace('-', '')
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results['ours']
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
    lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
        lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
    lines.append("")
report_markdown = "\n".join(lines)
get_report().log(section="Tokenizer evaluation", data=[
    report_markdown,
])
