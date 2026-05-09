# 手搓一个 ChatGPT

GPT 的全称是 **Generative Pre-trained Transformer**，生成式预训练 Transformer。

OpenAI 早期 GPT 论文的核心思路，就是先在大规模未标注文本上进行语言建模式预训练，再迁移到具体任务上。Transformer 架构则来自 2017 年的 *Attention Is All You Need*，其核心是用 self-attention 机制建模序列内部不同位置之间的关系，替代传统循环/卷积结构。

**如果我们要手搓一个 GPT，它到底是怎么一步一步学会说话的？**

## Tokenizer：模型看不到文字，它只看到数字

**我们怎样把人类的文本，翻译成机器能学习的形式？**

语言模型并不是直接看原始文本的，它看不到 "汉字"，也看不到 "英文单词"，更看不到代码里的 for、return、async、await。模型真正处理的是数字序列，也就是 token id。

所以在模型开始学 "下一个词是什么" 之前，我们必须先回答一个更基础的问题：

**一段文本，应该被切成多大的小块，模型才更容易学？**

这个 "分词器" 或者说 "切块器"，就是 Tokenizer。

### Tokenizer 到底在干什么？

如果把模型比作一个幼儿园小朋友，那么原始文本就是一本大部头著作。
问题在于，这个小孩一开始并不认识 "词"，它甚至连 "字" 也未必认识，它只会掰手指头数数，也就是处理离散编号。

如果我们是幼儿园老师，我们得先决定：

1. 是按单个字符教它？
2. 还是按常见词教它？
3. 还是按更灵活的 "常见片段" 教它？

不同的切法，会直接影响后面学习的难度。

举个最简单的英文例子：

```text
the cat sat on the mat
```

如果切得很碎，可能会变成：

```text
t | h | e |   | c | a | t | ...
```

这样模型要处理很多步。

如果切得更合理，可能会变成：

```text
the | cat | sat | on | the | mat
```

这样序列就短很多。

再换一个代码例子：

```python
for i in range(10):
```

如果 tokenizer 不熟悉代码，它可能切成很多碎片：

```text
f | o | r |   | i |   | i | n |   | r | a | n | g | e | ( | 1 | 0 | ) | :
```

但如果它见过很多代码，可能会更倾向于形成像下面这样的片段：

```text
for |  i |  in |  range | ( | 10 | ):
```

这里没有唯一标准答案，但直觉很清楚：

**高频片段如果能作为整体出现，模型通常会更省力。**

### 为什么要自己训练 Tokenizer？

现成 tokenizer 很多，比如 GPT-2 的 tokenizer、`cl100k_base`、Llama 系 tokenizer。  
很多时候直接拿来用完全没问题。

但 tokenizer 有一个很重要的特点：

**它不是 "通用越强越好"，而是 "越贴合语料越好"。**

比如：

- 如果我们的数据大部分是英文新闻，英文常见词切得好就很重要；
- 如果我们的数据里代码很多，那 `def`、`return`、`http`、`</div>` 这类片段能不能高效表示就很重要；
- 如果我们的数据里中文很多，那常见中文词组能不能不要被切得太碎就很重要；
- 如果我们的数据里数学公式很多，那 `\frac`、`\sum`、`^{2}` 这类片段能不能成块出现也很重要。

所以自己训练 tokenizer，本质上是在做一件事：

**让数据里最常见的模式尽量以更合适的颗粒度进入模型。**

### BPE 的核心思想：让高频片段慢慢长出来

这个项目的 tokenizer 训练，本质上是 BPE (Byte Pair Encoding) 的一个变体。

BPE 最好理解的方式，不是先看定义，而是直接看它 "怎么生长"。

假设训练文本里有很多下面这样的词：

```text
the
this
that
there
```

一开始，系统并不认识 `the` 这个整体。  
它只认识最基础的单位。对这个项目来说，可以先粗略理解成 "字节级别的小单位"。

然后训练过程会不断统计：

**哪些相邻片段经常一起出现？**

于是可能发生这样的合并：

#### 第一步

发现 `t` 和 `h` 经常相邻出现，于是合成：

```text
t + h -> th
```

#### 第二步

又发现 `th` 和 `e` 很常一起出现，于是继续合成：

```text
th + e -> the
```

#### 第三步

如果某些更长片段也足够常见，还可能继续合并。

比如在代码里，假设训练语料中到处都是：

```text
return
```

那么它也可能经历类似过程：

```text
r + e -> re
re + t -> ret
ret + urn -> return
```

实际内部过程当然会更复杂，不一定正好按这个顺序，但直觉就是这个直觉：

**BPE 不是人手工规定哪些词最重要，而是让高频片段自己从数据里长出来。**

### 为什么不是直接按 "单词" 切

既然最终目标是让文本别切太碎，那为什么不干脆按空格分词，或者按单词分词？

原因是现实文本远比英语单词列表复杂。

看几个例子：

#### 例子 1：新词和罕见词

```text
hyperparameterization
```

如果词表里没有这个词，按整词分就会尴尬。
但 BPE 可以退回到更小片段，比如：

```text
hyper | parameter | ization
```

哪怕还不行，也能继续退回到更小单位。

#### 例子 2：代码和符号

```text
user_id=42
```

这里既有字母，也有下划线、等号、数字。
按自然语言单词思路就不太好处理。

#### 例子 3：中文

中文本来就没有天然空格分词。
 "机器学习工程师" 到底该切成：

```text
机器 | 学习 | 工程师
```

还是：

```text
机器学习 | 工程师
```

甚至：

```text
机 | 器 | 学 | 习 | 工 | 程 | 师
```

并没有一个放之四海而皆准的固定答案。
BPE 的好处就是：它让数据自己决定哪些组合值得变成更稳定的 token。

### Tokenizer 的产出物，到底是什么

如果我们在项目里亲手训练一个 tokenizer，最后得到的并不是 "一段文本的切分结果"，而是一套以后可以反复复用的规则。

最核心的两个产物可以这样理解：

#### 1. `tokenizer.pkl`

这是分词器本体。

它里面包含的是：

- token 和 id 的映射
- merge 规则
- special token 配置

也就是说，它决定了：

**以后所有文本，到底该怎么被切成 token。**

#### 2. `token_bytes.pt`

这是另一张配套表：

```text
token_id -> 这个 token 对应多少个 UTF-8 字节
```

举个例子。假设某个 tokenizer 里有这些 token：

```text
token 101 -> "the"
token 102 -> "你"
token 103 -> "return"
```

那么它们对应的 UTF-8 字节数大概是：

```text
"the"     -> 3 bytes
"你"      -> 3 bytes
"return"  -> 6 bytes
```

注意这里说的是字节，不是字符数。

这张表的意义不是 "这个 token 长得漂不漂亮"，而是：

**这个 token 在原始文本里到底覆盖了多少底层内容。**

它是后面 BPB 评估的基础。

### BPB：为什么我们不能只看平均 loss

假设同一句话：

```text
I love deep learning
```

被两个 tokenizer 分成这样：

#### tokenizer A

```text
I | love | deep | learning
```

一共 4 个 token。

#### tokenizer B

```text
I |  l | ove |  de | ep |  learn | ing
```

一共 7 个 token。

这时如果只看 "每 token 的平均 loss"，其实并不公平。
因为两个 tokenizer 的颗粒度不一样。

更稳一点的问法是：

**为了描述同样一段原始文本，模型平均每个 byte 需要付出多少信息量？**

这就是 BPB (bits per byte) 的想法。

它就像物流计费：

- 按箱收费，不公平，因为箱子大小不同；
- 按公斤收费，更统一。

同理：

- 按每 token 算 loss，不同 tokenizer 下 "箱子大小" 不一样；
- 按每 byte 算，就更接近统一度量尺度。



---

embedding 是什么，隐藏维度是什么，它们之间是什么关系、fp8、fp16、注意力机制、多头注意力、Flash Attention、Chinchilla、AdamW、Muon、上下文长度和滑动窗口是类似于卷积核吗、lm_head、logits、cross-entropy、MLP、Transformer block 是什么、qkv、MHA、GQA、优化器、Linear 层是什么，为什么要这样设计、超参数、d12、学习率、残差连接、为什么神经网络要这样设计，是压缩算法吗？

---



## Embedding：将 Token 变成高维向量

Tokenizer 把原始文本翻译成了模型能处理的 token id。

比如一句话：

```text
I love deep learning
```

经过 tokenizer 之后，会变成：

```text
[40, 1842, 6732, 12930]
```

**但 token id 还不是神经网络真正擅长处理的对象。**

神经网络擅长处理的是向量，不是离散编号。

所以接下来的一步，是做 embedding。可以把 embedding 想成一本超大的 "数字词典"，每个 token 都是长度为 `n_embd` 的向量：

```text
40    -> [0.12, -0.33, 0.88, ...]
1842  -> [-0.51, 0.27, 0.04, ...]
6732  -> [0.91, -0.10, 0.22, ...]
```

### 初始 embedding 是怎么来的？

在一开始，模型并不知道 "猫" "狗" "苹果" "函数" "变量" 是什么意思。

embedding 词表，通常是随机初始化的。

假设 embedding 词表的大小 `vocab_size` 是：

```
vocab_size = 50000 (token)
```

向量的维度 `d_model` 是：

```
d_model = 4096
```

那 embedding 词表就是一个 50000 行 × 4096 列 的矩阵：

```
E ∈ R^(50000 × 4096)
```

每一行对应一个 token 的向量，也就是说，每个 token 都有一个 4096 维的可学习向量。

训练开始时，大概是这种状态：

```
token_id = 40    →  [0.003, -0.012, 0.008, ...]
token_id = 1842  →  [-0.010, 0.004, 0.017, ...]
token_id = 6732  →  [0.006, 0.011, -0.002, ...]
```

这些数字一开始没有语义，只是很小的随机数。

通过持续的训练，不断更新词表，embedding 词表才变得有意义。

### embedding 如何学出语义？

比如训练文本里经常出现：

```
猫 喜欢 吃 鱼
狗 喜欢 吃 骨头
猫 和 狗 都 是 动物
```

模型在预测时会遇到很多类似任务：

```
猫 喜欢 吃 ___
```

如果正确答案经常是 "鱼"，模型就会调整参数，让 "猫" 这个 token 的向量更有利于预测 "鱼" "动物" "宠物" 等相关 token。

这是通过大量预测任务训练出来的，所以 embedding 的语义来自训练语料中的共现关系。

### embedding 维度大小怎么决定？

向量的维度 `d_model` 是模型设计者设定的参数（也叫超参数，即预先定义好的常量）：

```
小模型：d_model = 512 / 768 / 1024
中模型：d_model = 2048 / 3072 / 4096
大模型：d_model = 5120 / 8192 / 12288 ...
```

维度越高，表达空间越宽，单个 token 能携带的信息容量越大。

维度越高，代价也越高：

- 模型参数量增加
- 显存增加
- 计算量增加
- 训练更贵
- 推理更慢

维度大小通常会和这些东西一起设计：

- 层数 depth
- 注意力头数 num_heads
- MLP 中间层维度
- 训练 token 数
- 算力预算
- 目标模型规模



## Attention Is All You Need

### 注意力机制是什么？

注意力机制解决的问题是：

> 当前 token 在理解自己时，应该重点参考前面的哪些 token？

比如句子：

```
小明把苹果放进书包，因为它很重。
```

"它" 到底指谁？模型需要看上下文。Attention 就是在计算：

```
当前 token 和其他 token 的相关性
```

核心公式是：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$
其中：

```
Q = Query：我现在想找什么信息？
K = Key：我这里有什么信息可以被别人匹配？
V = Value：如果别人关注我，我实际提供什么内容？
```

类比一下：

```
Q：学生的问题
K：每本书的目录标签
V：书里的正文内容
```

学生拿着问题 Q 去和所有书的目录 K 匹配，找到相关书后，再读取对应正文 V。

在 Transformer 里，每个 token 的向量 `x` 会通过三个 Linear 层变成 Q、K、V：

```
Q = xWq
K = xWk
V = xWv
```

也就是说：

```
同一个 token 向量
 → 变成 Query 版本
 → 变成 Key 版本
 → 变成 Value 版本
```

为什么要拆成 Q、K、V？

因为 "用来提问的信息"、"用来被别人匹配的信息"、"真正传递出去的信息" 是三种不同的表达方式。

### Attention 让 token 之间交流

对每个 token 来说，Attention 会问：

```
我应该从前文哪些 token 那里拿信息？
拿多少？
拿来的信息如何混合？
```

比如：

```
小明 把 苹果 放进 书包 因为 它 很 重
```

处理 "它" 的时候，Attention 可能会从 "小明"、"苹果"、"书包" 这些 token 里取信息。

Attention 输出的是：当前 token 融合上下文后的新信息。不是完整替换，而是生成一个 "增量"，例如 "它" 原本只是一个代词 embedding，经过 attention 后加上了 "可能指代苹果" 的上下文信息。
$$
\text{x}_1 = \text{x}_0 + \text{Attention}(\text{LayerNorm}(\text{x}_0)) = 它 + 它可能是苹果
$$

### Q、K、V 是怎么生成的？

→ tokenizer 把文本切成 token

→ 通过 embedding 词表查询 token 对应的高维向量

模型用三个 Linear 层生成 Q、K、V：

```
Q = XWq
K = XWk
V = XWv
```

其中：

```
Wq ∈ R^(d_model × d_model)
Wk ∈ R^(d_model × d_model)
Wv ∈ R^(d_model × d_model)
```

所以：

```
Q.shape = [seq_len, d_model]
K.shape = [seq_len, d_model]
V.shape = [seq_len, d_model]
```

如果是多头注意力，会再拆成多个 head。

例如：

```
d_model = 4096
num_heads = 32
head_dim = 128
```

于是：

```
Q.shape = [seq_len, 32, 128]
K.shape = [seq_len, 32, 128]
V.shape = [seq_len, 32, 128]
```

### 多头注意力：在高维空间中观察一个 token 向量

多头注意力会把 `d_model` 拆成多个 head。

关系通常是：

```
d_model = num_heads × head_dim
```

例如：

```
4096 = 32 × 128
```

每个 head 处理一个子空间。

```
num_heads 多：
  观察角度更多

head_dim 大：
  每个观察角度的表达容量更强
```

但总维度固定时，head 多了，每个 head 就变窄。









## LLM Pre Traning 大语言模型预训练





LLM 的本质流程是：把 token 变成向量，用 Transformer block 反复更新这些向量，最后把向量变回词表概率，并通过 cross-entropy 和优化器不断修正模型参数。

可以把 LLM 想成一个 **"把文字变成向量 → 在向量空间里反复加工 → 最后预测下一个 token"** 的系统。



```
 → tokenizer 切成 token
 → embedding 把 token 变成向量
 → 多个 Transformer block 反复处理向量
     → Attention 负责“看上下文”
     → MLP 负责“非线性加工/知识变换”
 → lm_head 把隐藏向量变成词表分数 logits
 → softmax 变成概率
 → cross-entropy 计算预测错了多少
 → optimizer 根据 loss 更新参数
```









Linear 层是什么？

**Linear 层就是矩阵乘法加偏置：**

```
y = xW + b
```

它的作用是把一个向量从一个空间投影到另一个空间。

比如输入是 4096 维，输出也是 4096 维：

```
[batch, seq_len, 4096] → Linear → [batch, seq_len, 4096]
```

或者从 4096 维扩展到 11008 维：

```
[batch, seq_len, 4096] → Linear → [batch, seq_len, 11008]
```

在 Transformer 里，Linear 到处都是：

```
生成 Q、K、V
Attention 输出投影
MLP 升维
MLP 降维
lm_head 输出词表 logits
```

你可以把 Linear 层理解成：模型学习出来的一组 "变换规则"，负责把信息重新组合。



多头注意力是什么？MHA 是什么？

MHA = Multi-Head Attention，多头注意力。

一个注意力头可以理解成一种观察角度，多个头就是多个观察角度并行工作。

比如理解一句话时：

```
头 1：关注语法关系
头 2：关注主谓宾
头 3：关注代词指代
头 4：关注时间顺序
头 5：关注代码括号匹配
```

实际不一定这么明确，但直觉上可以这样理解。

如果隐藏维度是 4096，有 32 个头，那么每个头大概处理：

```
4096 / 32 = 128 维
```

所以：

```
d_model = num_heads × head_dim
4096 = 32 × 128
```

多头注意力的作用是：让模型在同一层里，从多个子空间同时观察上下文关系。

GQA 是什么？

GQA = Grouped Query Attention，分组查询注意力。

它是 MHA 的一种省显存、省计算变体。

普通 MHA 里：

```
每个 Q head 都有自己独立的 K、V head
```

GQA 里：

```
多个 Q head 共享一组 K、V head
```

例如：

```
32 个 Q heads
8 个 KV heads
```

也就是每 4 个 Q head 共享一组 K/V。

为什么这样设计？

因为推理时，模型需要缓存历史 token 的 K/V，也就是常说的 **KV Cache**。上下文越长，KV Cache 越大。GQA 可以显著减少 KV Cache 的体积，同时保持较好的效果。

简单说：

```
MHA：效果强，但 KV Cache 大
MQA：所有 Q 共享一组 KV，最省，但可能损效果
GQA：折中方案，很多现代 LLM 都采用
```



Flash Attention 是什么？

**Flash Attention 不是新的注意力机制，而是更高效地计算标准 attention 的工程算法。**

普通 attention 最大的问题是要显式构造一个很大的注意力矩阵：

```
seq_len × seq_len
```

如果上下文长度是 8192，那么注意力矩阵规模是：

```
8192 × 8192
```

非常吃显存。

Flash Attention 的核心思想是：

> 不把完整 attention 矩阵全部写进显存，而是分块计算，尽量利用 GPU SRAM，减少显存读写。

所以它主要解决的是：

```
更省显存
更快
可以支持更长上下文
```

但它没有改变 attention 的数学定义。



上下文长度是什么？滑动窗口和卷积核类似吗？

**上下文长度 context length，就是模型一次最多能看到多少个 token。**

例如：

```
context length = 8192
```

表示模型最多能处理 8192 个 token 的输入。

滑动窗口注意力是：

```
每个 token 只看附近一段 token
```

比如窗口大小是 4096，那么第 8000 个 token 可能只看：

```
第 3904 到第 8000 个 token
```

它和卷积核有一点像，但不完全一样。

相似点：

```
都是局部窗口
都是限制当前点只能看附近区域
都可以降低计算量
```

不同点：

```
卷积核：固定权重，局部模式扫描
滑动窗口注意力：窗口内 token 两两动态计算相关性
```

所以更准确地说：

> 滑动窗口 attention 在 "局部性" 上像卷积核，但它的权重不是固定卷积核，而是根据输入动态算出来的。



Transformer block 是什么？

**Transformer block 是 LLM 的基本积木。**

一个典型 block 大概长这样：

```
输入 x
 → LayerNorm
 → Attention
 → 残差连接
 → LayerNorm
 → MLP
 → 残差连接
 → 输出 x
```

可以简化成：

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

一个大模型通常堆很多层 Transformer block。

比如：

```
12 层
24 层
32 层
80 层
```

每一层都在对 token 表示进行加工。

浅层可能偏词法、局部结构；深层可能偏语义、推理、任务模式。当然这只是粗略理解。



MLP 是什么？

**MLP = Multi-Layer Perceptron，多层感知机。**

在 Transformer 里，MLP 通常是：

```
Linear 升维
激活函数
Linear 降维
```

例如：

```
4096 → 11008 → 4096
```

Attention 负责 "从上下文拿信息"，MLP 负责 "对每个 token 自己的信息做非线性加工"。

可以这样理解：

```
Attention：横向交流，让 token 之间交换信息
MLP：纵向思考，对单个 token 的状态进行变换
```

为什么 MLP 要先升维再降维？

因为升维后，模型有更大的中间空间来组合特征。就像你在草稿纸上展开计算，最后再整理成原来的格式。



残差连接是什么？为什么需要？

残差连接就是：

```
输出 = 输入 + 变换结果
```

例如：

```
x = x + Attention(x)
x = x + MLP(x)
```

它的好处是：

1. **让深层网络更容易训练。**
    如果没有残差，几十层、上百层网络很容易梯度传不回去。
2. **保留原始信息。**
    每一层不是完全重写表示，而是在原表示上 "增量修改"。
3. **让模型可以学会 "不改"。**
    如果某一层暂时没必要处理某些信息，它可以让变换结果接近 0，直接把输入传下去。

你可以把它理解成：

```
不是每层都重新写一篇文章，
而是在上一版基础上做批注和修改。
```



lm_head 是什么？

lm_head 是语言模型最后一层，把隐藏向量映射到词表大小。

假设：

```
hidden_size = 4096
vocab_size = 50000
```

最后一个 token 的隐藏向量是：

```
[1, 4096]
```

lm_head 会把它变成：

```
[1, 50000]
```

这 50000 个数字分别对应词表里每个 token 的分数。

例如：

```
“猫”：12.3
“狗”：10.8
“车”：2.1
“的”：6.7
...
```

这些分数就是 logits。

**Logits 是还没有归一化的原始分数。**

它们不是概率，可以是负数，也不要求加起来等于 1。

例如：

```
猫：12.3
狗：10.8
车：2.1
```

经过 softmax 后才会变成概率：

```
猫：0.78
狗：0.17
车：0.01
...
```

所以：

```
lm_head 输出 logits
softmax 把 logits 转成概率
采样/argmax 根据概率选下一个 token
```



Cross-entropy 是什么？

**Cross-entropy 交叉熵，是训练语言模型最常用的 loss 计算方法。**

它衡量的是：

> 模型给正确答案的概率有多高。

比如正确下一个 token 是 "猫"。

模型预测：

```
猫：0.80
狗：0.10
车：0.01
```

loss 就低。

如果模型预测：

```
猫：0.01
狗：0.80
车：0.10
```

loss 就高。

公式可以粗略理解为：

```
loss = -log(正确 token 的概率)
```

正确 token 概率越大，loss 越小。



FP16、FP8 是什么？

它们是数字精度格式。

```
FP32：32 位浮点数，精度高，占显存大
FP16：16 位浮点数，精度较低，占显存小，训练常用
BF16：16 位，但动态范围更接近 FP32，LLM 训练常用
FP8：8 位浮点数，更省显存和带宽，但训练更难稳定
```

模型训练和推理里有大量矩阵乘法，数字格式越小：

```
显存占用越低
数据搬运越快
吞吐可能越高
```

但代价是：

```
数值误差更大
训练稳定性更难控制
```

所以 FP8 通常需要配套的 scaling、混合精度策略和硬件支持。



优化器是什么？

**优化器 optimizer，就是根据 loss 和梯度来更新模型参数的算法。**

训练过程是：

```
输入数据
 → 模型预测
 → 计算 loss
 → 反向传播得到梯度
 → 优化器更新参数
```

梯度告诉模型：

```
这个参数往哪个方向调，loss 会下降
```

优化器决定：

```
调多少？
怎么调？
是否考虑历史梯度？
是否加权重衰减？
```



AdamW 是什么？

**AdamW 是目前深度学习里非常常见的优化器。**

它大概做了几件事：

```
使用一阶动量：记住过去梯度方向
使用二阶动量：估计梯度波动大小
对不同参数自适应调整步长
使用 decoupled weight decay 控制参数规模
```

你可以把普通 SGD 理解成：

```
看当前坡度走一步
```

AdamW 更像：

```
看当前坡度 + 参考过去方向 + 根据路面崎岖程度调整步子
```

LLM 训练中，AdamW 长期是非常主流的选择。



Muon 是什么？

[KellerJordan/Muon: Muon is an optimizer for hidden layers in neural networks](https://github.com/KellerJordan/Muon)

这里的 Muon 不是物理里的 μ 子，而是一个比较新的神经网络优化器。它主要面向神经网络的隐藏层权重，核心特点是对二维权重矩阵的更新做矩阵正交化。Muon 原始项目也明确建议：隐藏层权重用 Muon，而 embedding、classifier head、bias 等参数仍可用 AdamW。

近年的论文和实现开始研究 Muon 在 LLM 训练中的扩展性，例如 2025 年的论文指出，加入 weight decay、调整 per-parameter update scale 是让 Muon 扩展到更大 LLM 训练的重要技巧。 PyTorch 文档里也已经有 `torch.optim.Muon` 的相关页面，并引用了 Muon 的原始说明和 LLM 扩展论文。

对初学者来说，可以先这样理解：

```
AdamW：非常成熟、通用、稳定
Muon：较新，试图利用矩阵结构改善隐藏层训练效率
```

你现在不需要马上深挖 Muon。先理解 AdamW、梯度、学习率、weight decay，会更重要。



Chinchilla 是什么？

Chinchilla 是 DeepMind 提出的一个重要结论，核心不是某个模型结构，而是一个训练配比思想：

> 在固定训练计算量下，模型参数量和训练 token 数需要平衡。很多早期大模型参数太多、训练 token 太少，并不 compute-optimal。

通俗说：

```
不是模型越大越好。
模型大小和数据量要匹配。
```

比如你有固定预算，可能不是训练一个超大但没看够数据的模型更好，而是训练一个稍小但看更多 token 的模型更好。

这影响了后面很多 LLM 的训练策略。



超参数是什么？

**超参数就是训练前由人设定的参数，不是模型自己学出来的参数。**

例如：

```
学习率 learning rate
batch size
层数 num_layers
隐藏维度 hidden_size
注意力头数 num_heads
上下文长度 context_length
weight decay
dropout
训练步数
warmup steps
```

模型参数是训练出来的：

```
embedding 表
Linear 层权重
Attention 权重
MLP 权重
lm_head 权重
```

超参数是你配置出来的：

```
模型多宽？
模型多深？
每次学多快？
一次喂多少数据？
上下文多长？
```



学习率是什么？

**学习率 learning rate 决定每次参数更新的步子有多大。**

```
学习率太大：容易震荡，甚至训练崩掉
学习率太小：训练很慢，可能学不动
```

类比下山：

```
学习率太大：一步跨太远，可能越过山谷
学习率太小：一步太短，走很久
```

LLM 训练通常还会配合：

```
warmup：一开始学习率从小慢慢升高
decay：后期学习率慢慢降低
```



为什么神经网络要这样设计？

因为 LLM 要解决的问题是：

```
给定前面的 token，预测下一个 token。
```

但这个任务背后需要很多能力：

```
理解词义
理解语法
理解上下文
处理长距离依赖
记住格式
学习世界知识
做简单推理
生成连贯文本
```

Transformer 的设计正好把这些需求拆开了：

```
Embedding：把离散 token 变成连续向量
Position encoding / RoPE：告诉模型 token 的位置
Attention：让 token 之间交换信息
MLP：对每个 token 的表示做复杂变换
残差连接：让深层网络容易训练
LayerNorm：稳定训练
lm_head：把隐藏状态转回词表预测
Cross-entropy：告诉模型预测错了多少
Optimizer：根据错误调整参数
```

所以它不是随便拼的，而是一套围绕 "序列建模" 和 "可训练性" 逐步演化出来的工程结构。



神经网络是压缩算法吗？

这个问题很关键。

严格说：

> 神经网络不是传统意义上的压缩算法，但训练后的模型可以被理解成对训练数据分布的一种有损压缩。

它不像 zip 那样：

```
输入原文件 → 压缩 → 可以完全还原
```

LLM 更像：

```
看过大量文本
 → 学到语言规律、知识模式、统计结构
 → 用参数保存这些规律
 → 生成时根据上下文重建“合理的后续文本”
```

所以它不是记住所有原文，也不是精确压缩整个互联网。

更准确的说法是：

```
LLM 学到的是数据分布的近似模型。
```

它压缩的不是 "每一篇文章的原文"，而是：

```
语言规律
概念关系
常见模式
知识关联
推理模板
代码结构
格式习惯
```

因此它有时能泛化，有时也会幻觉。



把所有概念串起来

可以把一次训练过程想成这样：

```
1. 输入文本：
   "猫坐在垫子上"

2. tokenizer 切 token：
   [猫, 坐, 在, 垫子, 上]

3. embedding：
   每个 token 变成 hidden_size 维向量

4. Transformer block 反复处理：
   Attention：每个 token 看上下文
   MLP：每个 token 内部加工
   残差连接：保留旧信息并叠加新信息

5. lm_head：
   把最后隐藏向量映射成 vocab_size 个 logits

6. softmax：
   logits 变成每个 token 的概率

7. cross-entropy：
   看正确 token 的概率有多高

8. optimizer：
   根据 loss 调整 embedding、attention、MLP、lm_head 等权重
```

推理时则少了训练部分：

```
输入文本
 → embedding
 → Transformer blocks
 → lm_head
 → logits
 → softmax / sampling
 → 生成下一个 token
 → 把新 token 接回上下文
 → 继续生成
```



最小知识地图

```
第一层：数据表示
token
embedding
hidden dimension
logits
lm_head

第二层：模型结构
Linear
Attention
QKV
MHA / GQA
MLP
Residual
Transformer block

第三层：训练
loss
cross-entropy
gradient
optimizer
AdamW
learning rate
hyperparameter

第四层：效率与规模
FP16 / BF16 / FP8
Flash Attention
context length
sliding window
KV Cache
Chinchilla
Muon
```







## 神经网络：小镇做题家

### 有了 token 之后，神经网络才真正开始工作

到这里，我们终于

---

### 训练语言模型时，`x` 和 `y` 到底是什么

GPT 训练时并不是把一段话扔进去，然后让模型 "自己悟"。

它做的是一个非常明确的监督任务：

**给前文，预测下一个 token。**

例如一句 token 序列是：

```text
[10, 20, 30, 40]
```

那么训练样本通常会组织成：

```text
x = [10, 20, 30]
y = [20, 30, 40]
```

意思是：

- 看到 `10`，预测 `20`
- 看到 `10,20`，预测 `30`
- 看到 `10,20,30`，预测 `40`

所以语言模型训练，本质上是在大量重复做一件事：

**让模型不断练习 "根据前文猜下一个 token" 这道题。**

## 注意力机制：为什么它是 GPT 的灵魂

embedding 之后，每个 token 都已经变成了向量。
但这些向量此时还只是 "查表得到的初始表示"，它们之间还没有充分交换上下文信息。

这时，注意力机制就登场了。

它最重要的作用，不是 "把整段话背下来"，而是：

**让当前位置决定：前文中哪些位置值得重点参考。**

举个最直观的例子：

```text
The capital of France is Paris.
```

当模型快要预测 `Paris` 时，前文里的每个词显然不是同等重要的。

例如：

- `The` 很重要吗？没那么重要。
- `capital` 有点重要。
- `France` 非常重要。

注意力机制做的事情，粗略说就是：

**给前文里的不同位置分配不同权重，然后把最相关的信息重点拿回来。**

所以 attention 不是 "平均看所有词"，而是 "有选择地看重点"。

---

### `q / k / v` 到底是什么

这三个字母第一次看特别抽象，但我们可以先用一句很粗的直觉记住它们：

- `q`(query)：我现在想找什么？
- `k`(key)：我这里有什么标签，供别人来匹配？
- `v`(value)：如果你真的关注我，我真正提供给你的内容是什么？

比如当前位置想预测下一个 token，它会生成自己的 `q`。  
前文每个位置则有自己的 `k` 和 `v`。

然后模型会比较：

```text
当前 q 和历史位置的 k 有多匹配？
```

匹配越高，那个位置的 `v` 就越该被拿来参考。

数学上常写成：

```text
attention_weights = softmax(q @ k^T)
output = attention_weights @ v
```

如果现在看这个公式还觉得抽象，也没关系。  
先抓住故事版理解就够了：

**当前位置拿着自己的问题(query)，去前文里找最相关的内容(value)。**

---

### 为什么叫 Causal Self-Attention

这个名字里有两个关键词：

#### Self

因为 query、key、value 都来自同一条序列本身。  
不是从别的句子借来的。

#### Causal

因为当前时刻只能看自己和前文，不能偷看未来。

比如在训练这句话：

```text
The capital of France is Paris.
```

当模型正在预测 `Paris` 时，它可以看：

```text
The capital of France is
```

但绝不能提前看见 `Paris`。  
否则这就不是预测，而是作弊。

所以 GPT 用的是 **causal self-attention**。

---

### 神经网络里的另一半：MLP 在做什么

很多人第一次学 Transformer 时，注意力机制太耀眼，反而忽略了 MLP。

但 MLP 其实非常重要。

如果说 attention 负责：

**让不同 token 之间交换信息**

那么 MLP 更像是在做：

**让每个 token 自己把吸收来的信息再深加工一遍**

一个非常形象的比喻是：

- attention：像开会，大家互相交流
- MLP：像会后每个人回办公室自己整理、思考、加工

所以一个 Transformer block，不是 "只有 attention"，而是：

```text
先交流一下（attention）
再自己加工一下（MLP）
```

而 GPT，就是把这样的 block 一层层堆起来。

---

### 残差连接：为什么深层网络不会一层层把旧信息洗掉

在 Transformer block 里，我们常看到这种结构：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

这里的 `x + ...` 就是残差连接。

它最直观的意义是：

**新的一层不是把旧表示推翻重来，而是在旧表示上做增量修正。**

为什么这样好？

因为如果深层网络每一层都从头重写，很容易训练不稳定，信息也不容易顺畅传下去。  
残差连接相当于给每层都留了一条 "保底通道"。

所以我们可以把残差理解成：

**旧知识别丢，新知识慢慢叠加。**

---

### `GPT.forward()`：整台机器终于转起来了

现在，我们终于可以把整条主线串起来了。

给定输入 token ids：

```text
idx
```

`GPT.forward()` 大致会做下面这些事：

#### 1. 取出位置编码信息

让模型知道 "谁在前，谁在后"。

#### 2. token embedding

把 token id 查表变成向量。

#### 3. 经过多层 Transformer block

每层都做：

- causal self-attention
- MLP
- 残差连接

#### 4. 经过 `lm_head`

把最后的隐藏向量投影回词表维度，得到 logits。

#### 5. 如果提供了 `targets`

就进一步计算交叉熵 loss。

所以 `GPT.forward()` 的本质可以写成：

```text
token ids
-> hidden vectors
-> contextualized hidden vectors
-> logits over vocab
-> loss
```

这条线一旦清楚，代码就不再像魔法。

---

### `lm_head` 到底是什么

这是最容易被忽略，但其实非常关键的一层。

前面多层 Transformer 算完之后，模型手里拿到的是每个位置的隐藏向量。  
但训练目标并不是 "这个向量漂不漂亮"，而是：

**下一个 token 到底应该是谁？**

所以必须再来一步，把隐藏向量变成：

```text
对词表中每个 token 的打分
```

这一步就是 `lm_head`。

如果词表大小是 32768，那么某个位置经过 `lm_head` 后，会得到一个长度 32768 的向量：

```text
[score(token_0), score(token_1), ..., score(token_32767)]
```

这个向量就是 logits。

注意：

- logits 不是概率；
- 它只是原始打分；
- 后面 softmax 才会把它变成概率分布。

---

### loss 为什么能从 logits 里算出来

有了 logits，训练就简单了。

因为现在模型已经对 "下一个 token 可能是谁" 给出了完整打分表，  
而我们手里正好有真实答案 `targets`。

所以接下来只要问一句：

**你给真实答案打的分高不高？**

如果真实 token 的概率被分得很高，loss 就低；  
如果真实 token 的概率被分得很低，loss 就高。

这就是交叉熵 loss 在语言模型里的最直观含义。

举个极简例子。

假设当前位置真实下一个 token 是 `Paris`。

模型给出的概率可能是：

```text
Paris   : 0.80
London  : 0.10
Berlin  : 0.05
other   : 0.05
```

那 loss 会比较低。

如果模型给的是：

```text
Paris   : 0.03
London  : 0.60
Berlin  : 0.20
other   : 0.17
```

那 loss 就会比较高。

所以 loss 本质上是在衡量：

**模型有没有把真实答案排到足够靠前的位置。**

---

### 回头看训练脚本，我们其实在干什么

到这里，我们再回头看 "手搓一个 GPT" 的训练流程，就会非常顺了。

我们真正做的事情其实是：

#### 第一步：准备 tokenizer

先决定文本怎么切成 token。

#### 第二步：把 token 变成训练样本

构造 `x / y`，让模型学 "根据前文猜下一个 token"。

#### 第三步：搭起神经网络

准备 embedding、attention、MLP、lm_head。

#### 第四步：前向传播

让数据在网络里一路流动，得到 logits 和 loss。

#### 第五步：反向传播和参数更新

把 "预测错了多少" 这个信号传回去，逐步调整参数。

所以训练 GPT 并不是神秘仪式。  
它只是在极大量的数据上，反复做同一件事：

**给前文，猜下一个 token，然后用错误来改进自己。**

---

### 这篇文章最想留下的几个认知

如果只总结最重要的几句话，我会写这几条。

#### 1. tokenizer 决定模型看到世界时的颗粒度

模型不是直接读文字，它先读 token。

#### 2. embedding 不是理解语义，而是把离散 id 变成可计算向量

这是神经网络真正开始工作的入口。

#### 3. attention 的核心不是 "记忆全文"，而是 "有选择地看前文" 

当前位置会重点参考和自己最相关的历史位置。

#### 4. MLP 不负责跨位置通信，它负责每个位置自己的深加工

attention 和 MLP 是分工合作，不是互相替代。

#### 5. `lm_head` 把隐藏表示重新翻译回 "全词表打分" 

所以模型才有可能回答 "下一个 token 是谁"。

#### 6. loss 不是玄学指标，它只是 "真实答案有没有被排在前面" 

而 BPB 则是在 tokenizer 粒度不同的前提下，让这种比较更公平。

---

**手搓一个 GPT，本质上不是 "搭一个神经网络" 这么简单，而是先决定文本该怎么切成 token，再把 token 变成向量，让注意力机制在上下文里找重点，让 MLP 深加工表示，最后通过 `lm_head` 输出下一个 token 的概率分布，并用交叉熵 loss 不断纠正模型；如果我们还想跨 tokenizer 公平比较训练效果，就要进一步引入 BPB 这种按 byte 归一化的指标。**

真正的 ChatGPT 还包含指令微调、RLHF / DPO、工具调用、安全对齐、推理服务、上下文管理等大量工程系统。