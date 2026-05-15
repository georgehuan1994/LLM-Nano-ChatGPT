"""从 HuggingFace 下载 Karpathy 发布的预训练 nanochat 模型（含分词器与 base/chat 权重）。

默认下载 karpathy/nanochat-d34 的全部内容到 $NANOCHAT_BASE_DIR/pretrained/<repo_name>。
支持断点续传：HuggingFace Hub 默认会复用已下载的分块，重复执行同一命令即可继续。

示例：
    python -m scripts.download_pretrained
    python -m scripts.download_pretrained --repo-id karpathy/nanochat-d34
    python -m scripts.download_pretrained --include "tokenizer*" "*.json"
    python -m scripts.download_pretrained --target-dir /data/nanochat-d34
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-id",
        default="karpathy/nanochat-d34",
        help="HuggingFace 仓库名，默认 karpathy/nanochat-d34",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="本地保存目录；缺省时为 $NANOCHAT_BASE_DIR/pretrained/<repo_name>",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="只下载匹配这些 glob 的文件，例如 --include 'tokenizer*' '*.json'",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="忽略匹配这些 glob 的文件",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="指定分支/tag/commit；默认是 main",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="并发下载线程数，默认 4；网络好时可调大",
    )
    return parser.parse_args()


def resolve_target_dir(target_dir: str | None, repo_id: str) -> Path:
    if target_dir:
        return Path(target_dir).expanduser().resolve()
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        base = str(Path.home() / "autodl-fs" / ".nanochat")
    repo_name = repo_id.split("/")[-1]
    return Path(base).expanduser().resolve() / "pretrained" / repo_name


def main() -> None:
    args = parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_ENDPOINT", endpoint)

    target_dir = resolve_target_dir(args.target_dir, args.repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f" repo_id        : {args.repo_id}")
    print(f" revision       : {args.revision or 'main'}")
    print(f" target_dir     : {target_dir}")
    print(f" HF_ENDPOINT    : {endpoint}")
    print(f" include        : {args.include}")
    print(f" exclude        : {args.exclude}")
    print(f" max_workers    : {args.max_workers}")
    print("=" * 60)

    local_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(target_dir),
        allow_patterns=args.include,
        ignore_patterns=args.exclude,
        max_workers=args.max_workers,
    )

    print(f"\n下载完成，文件位于: {local_path}")


if __name__ == "__main__":
    main()
