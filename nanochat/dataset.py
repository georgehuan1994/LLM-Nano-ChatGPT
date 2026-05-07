"""
这个模块负责管理 nanochat 的基础预训练数据集。

这里的预训练数据以一组 parquet 文件的形式存储。你可以把它理解为：
1. 原始大语料被切分成很多个数据分片 shard。
2. 每个 shard 对应一个 parquet 文件，例如 shard_00000.parquet。
3. 训练时，代码会按顺序读取这些 parquet 文件中的文本字段 text。

这个文件主要提供两类能力：
1. 列出本地已经存在的 parquet 数据文件。
2. 在本地没有数据时，按需从远程下载指定 shard。

如果你想进一步了解这些 parquet 文件是如何准备出来的，可以查看 dev/repackage_data_reference.py
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# 当前预训练数据集的基本配置

# 远程数据集的基础下载地址，实际下载某个 shard 时，会在这个地址后面拼接具体文件名
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

# 最后一个 shard 的编号，对应 shard_06542.parquet
MAX_SHARD = 6542

# shard 文件名格式
index_to_filename = lambda index: f"shard_{index:05d}.parquet"

# 本地数据目录 .nanochat/base_data_climbmix
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# -----------------------------------------------------------------------------

def list_parquet_files(data_dir=None, warn_on_legacy=False) -> list[str]:
    """
    扫描数据目录，返回其中所有 parquet 文件的完整路径。

    参数说明：
    - data_dir:
        要扫描的目录。默认使用本模块定义的 DATA_DIR
    - warn_on_legacy:
        如果默认新目录不存在，是否打印一段旧数据集兼容提示

    返回值：
    - 按文件名排序后的 parquet 完整路径列表
    """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # 兼容旧版数据目录
    # 项目曾从 FinewebEdu-100B 切换到 ClimbMix-400B
    # 如果新目录不存在，这里会尝试回退到旧目录，避免老用户直接报错
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  警告：需要升级数据集")
            print("=" * 80)
            print()
            print(f"  未找到目录：{data_dir}")
            print()
            print("  nanochat 最近已从 FinewebEdu-100B 切换到 ClimbMix-400B。")
            print("  如果你在 2026-03-04 之后更新过代码，看到这条提示是正常的。")
            print("  如果你想升级到新的 ClimbMix-400B 数据集，可以运行下面两条命令：")
            print()
            print("    python -m nanochat.dataset -n 170     # 下载约 170 个 shard，足够做 GPT-2 量级实验，可按需调整")
            print("    python -m scripts.tok_train           # 基于新的 ClimbMix 数据重新训练 tokenizer")
            print()
            print("  当前会先回退到你旧的 FinewebEdu-100B 数据目录继续尝试运行。")
            print("=" * 80)
            print()
        # 回退到旧版数据目录
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    按批次遍历 parquet 数据集，逐批产出文本列表。

    这里的批次不是训练时的 batch，而是 parquet 文件内部的 row group。
    parquet 格式通常会把一个大文件再切成多个 row group，按 row group 读取会更高效。

    参数说明：
    - split:
        只能是 "train" 或 "val"。
        项目约定：最后一个 parquet 文件作为验证集，其余文件作为训练集。
    - start, step:
        主要给分布式训练 DDP 使用。
        例如多个进程并行时，可以让不同进程跳着读取不同的 row group，避免重复。
        常见用法是 start=rank, step=world_size。

    产出内容：
    - 每次 yield 一个 Python 列表，列表里是一批文本字符串。
    - 这个函数相当于数据读取器
    - 它不会一次性把所有数据读进内存，而是边读边产出，适合大规模语料。
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


def __download_single_file(index):
    """
    下载单个 shard 文件，并在失败时自动重试

    参数说明：
    - index:
        shard 编号，例如 0 会对应 shard_00000.parquet

    返回值：
    - True: 下载成功，或者文件本地已存在
    - False: 多次重试后仍然失败
    """

    # 先计算本地目标路径；如果文件已经存在，不再重复下载
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # 远程下载地址
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # 带重试的下载逻辑
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # 先写入临时文件，下载完整后再原子替换成正式文件
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 每次写入 1MB
                    if chunk:
                        f.write(chunk)
            # 下载完成，把临时文件更名为正式文件
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # 清理可能残留的临时文件或损坏文件
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # 指数退避重试：第 1 次失败等 2 秒，第 2 次等 4 秒，依此类推，直至第 5 次失败后放弃
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="下载 nanochat 预训练数据集的 parquet 分片文件"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="要下载的训练集 shard 数量。默认 -1 表示下载全部训练 shard；程序还会额外始终下载最后一个验证 shard。",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="并行下载的 worker 数量，默认 4。网络和磁盘较慢时不一定越大越快。",
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(DATA_DIR, exist_ok=True)

    # 用户通过 -n 指定要下载多少个训练 shard。
    # 验证 shard 总是会额外下载，固定使用最后一个 shard。
    # 也就是说：
    # - 如果 -n 8，那么会下载训练 shard 0~7，再加上最后一个验证 shard。
    # - 如果 -n -1，那么会下载全部训练 shard，再加上最后一个验证 shard。
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # 始终下载验证 shard

    # 开始并行下载
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(__download_single_file, ids_to_download)

    # 汇总下载结果
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
