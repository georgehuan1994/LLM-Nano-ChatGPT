"""
这个模块提供与语言模型评估相关的辅助函数。

当前最核心的函数是 `evaluate_bpb(...)`，
它用来计算 BPB(bits per byte)。

为什么项目要单独算 BPB，而不是只看“平均 loss”？

因为平均 loss 默认是“每个 token 的平均预测损失”。
这在 tokenizer 固定时没问题，但一旦你更换 tokenizer，
这个指标就会混入“切分粒度不同”带来的偏差。

举个直觉例子：

同一句话：
    "I love deep learning"

如果 tokenizer A 切成 4 个 token，
tokenizer B 切成 7 个 token，
那么直接比较“每 token loss”就不完全公平。

更稳妥的比较方式是：
    为了描述同样长度的原始文本，模型平均每个字节需要多少 bit？

这就是 BPB 的意义：
它把 loss 从“按 token 计价”改成了“按 byte 计价”，
从而更适合跨 tokenizer 对比。
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    计算 BPB(bits per byte)。

    参数说明：
    - model:
        语言模型。要求支持 `model(x, y, loss_reduction='none')`，
        返回每个目标位置各自的 loss，而不是只返回一个平均值。
    - batches:
        一个批量迭代器。每次 next(...) 返回 `(x, y)`：
        - x: 输入 token ids
        - y: 目标 token ids
    - steps:
        要评估多少个 batch。
    - token_bytes:
        一维张量，形状为 `(vocab_size,)`。
        含义是：
            token_id -> 该 token 对应多少个 UTF-8 字节
        如果某个 token 不应该计入 BPB（例如 `<|bos|>` 这类 special token），
        则它的字节数会被记为 0。

    返回值：
    - 一个 float，表示 bits per byte，越低通常越好。

    -----------------------------
    这套计算的核心思路
    -----------------------------

    普通交叉熵 loss 往往可以理解为：
        “平均每个 token 需要多少信息量”

    而 BPB 想表达的是：
        “平均每个原始字节需要多少信息量”

    所以它的做法不是：
        先求 mean loss

    而是：
        1. 先把所有目标 token 的 loss 加总，得到 total_nats
        2. 再把这些目标 token 对应的字节数加总，得到 total_bytes
        3. 用 total_nats / total_bytes 做归一化
        4. 再把 nat 换算成 bit

    最终公式是：

        bpb = total_nats / (ln(2) * total_bytes)

    其中：
    - PyTorch 交叉熵默认以自然对数为底，所以单位是 nat
    - 1 nat = 1 / ln(2) bit

    -----------------------------
    为什么这里要额外处理两类 token
    -----------------------------

    1. special token 不应计入 BPB
       例如 `<|bos|>` 这种控制 token 并不对应真实原文内容，
       如果把它们也算进“文本字节长度”，会污染指标。

    2. ignore_index 位置不应计入 BPB
       训练/评估时有些位置本来就被 mask 掉了，常见做法是把目标 token 记成 -1。
       这些位置既不应该算 loss，也不应该算字节数。
    """
    # total_nats:
    #   所有“有效目标 token”的总损失，单位是 nat。
    # total_bytes:
    #   这些有效目标 token 一共覆盖了多少原始 UTF-8 字节。
    #
    # 注意这里刻意不直接累计“平均值”，而是累计“总量”。
    # 因为 BPB 的本质是：
    #   总信息量 / 总字节数
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)

        # model(..., loss_reduction='none') 返回每个位置单独的 loss，
        # 形状通常是 (B, T)。
        #
        # 这一步很关键：
        # 如果这里只拿到了一个平均 loss，就无法知道不同位置该如何对应不同字节数。
        loss2d = model(x, y, loss_reduction='none') # (B, T)

        # 展平成一维，只是为了让后面的逐位置统计更简单。
        loss2d = loss2d.view(-1)
        y = y.view(-1)

        # 如果 y 中存在负数，通常表示这些位置是 ignore_index，例如 -1。
        #
        # 这里先转成 int32 再比较，是因为某些后端（如 MPS）对 int64 的 < 0 支持不完整。
        if (y.int() < 0).any():
            # 较慢但更稳妥的分支：
            # 某些目标位置被 mask 掉了，因此不能直接拿 y 去索引 token_bytes，
            # 否则负数索引会出错。

            # valid=True 表示这个目标位置是真正参与评估的。
            valid = y >= 0

            # 对非法位置先临时塞一个 0，目的是保证索引时不越界。
            # 这里的 0 只是占位，后面会再用 valid 掩码把无效位置归零。
            y_safe = torch.where(valid, y, torch.zeros_like(y))

            # 把每个有效 token 映射到它对应的字节长度。
            # 无效位置（ignore_index）贡献 0 字节。
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )

            # 只统计真正有字节内容的 token 的 loss。
            #
            # 这里 `(num_bytes2d > 0)` 会自动把 special token 排除掉，
            # 因为 special token 在 token_bytes 里被预先记成了 0。
            total_nats += (loss2d * (num_bytes2d > 0)).sum()

            # 总字节数则直接累加这些 token 覆盖的字节长度。
            total_bytes += num_bytes2d.sum()
        else:
            # 快路径：
            # 如果没有 ignore_index，就可以直接安全地索引 token_bytes。
            num_bytes2d = token_bytes[y]

            # 同样只统计字节数 > 0 的 token 的 loss。
            # 这一步继续保证 special token 不会进入 BPB。
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()

    # 如果是分布式评估，不同 rank 只看到了自己那一份数据。
    # 所以这里要把所有卡上的 total_nats / total_bytes 汇总起来。
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    # 转回 Python 标量，便于最终计算和返回。
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    # 极端防御：
    # 如果一个有效字节都没有，BPB 没有定义。
    # 这里返回 inf，表示这次评估结果无意义/不可用。
    if total_bytes == 0:
        return float('inf')

    # PyTorch 交叉熵的单位是 nat，而我们要的是 bit。
    #
    # 换算关系：
    #   1 nat = 1 / ln(2) bit
    #
    # 所以：
    #   total_nats / ln(2) = total_bits
    #
    # 再除以 total_bytes，就得到每个字节平均需要多少 bit。
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
