#!/bin/bash

set -euo pipefail

# Inspect the global/cloud Python environment without creating a venv.
# Usage:
#   bash runs/check_cloud_env.sh
# Optional:
#   PYTHON=/path/to/python bash runs/check_cloud_env.sh

PYTHON="${PYTHON:-python}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python command not found: $PYTHON"
    echo "hint: use the cloud image's activated environment, or set PYTHON=/path/to/python"
    exit 1
fi

"$PYTHON" - <<'PY'
import importlib.util
import platform
import sys

print("Python executable:", sys.executable)
print("Python version:   ", sys.version.split()[0])
print("Platform:         ", platform.platform())

print("\nTorch/CUDA:")
try:
    import torch
    print("  torch:          ", torch.__version__)
    print("  torch cuda:     ", torch.version.cuda)
    print("  cuda available: ", torch.cuda.is_available())
    print("  cuda devices:   ", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"  gpu {idx}:         {props.name}, {props.total_memory / 1024**3:.1f} GiB, sm_{props.major}{props.minor}")
except Exception as exc:
    print("  error:          ", repr(exc))

print("\nCommands:")
import shutil
for command in ["python", "pip", "torchrun", "nvidia-smi"]:
    print(f"  {command:12s}", shutil.which(command) or "missing")

torchrun_module = importlib.util.find_spec("torch.distributed.run") is not None
print(f"  {'torchrun module':12s}", "ok" if torchrun_module else "missing")

required = [
    "torch",
    "nanochat",
    "datasets",
    "tokenizers",
    "tiktoken",
    "wandb",
    "filelock",
    "psutil",
    "setuptools",
]
optional = ["kernels", "rustbpe"]

print("\nRequired imports:")
missing = []
for name in required:
    ok = importlib.util.find_spec(name) is not None
    print(f"  {name:12s} {'ok' if ok else 'missing'}")
    if not ok:
        missing.append(name)

print("\nOptional imports:")
for name in optional:
    ok = importlib.util.find_spec(name) is not None
    print(f"  {name:12s} {'ok' if ok else 'missing'}")

print("\nRecommendation:")
version = sys.version_info
if version < (3, 12) or version >= (3, 13):
    print("  This repo is configured for Python >=3.12,<3.13 in pyproject.toml.")
    print("  Prefer the AutoDL PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 image.")
else:
    print("  Python version matches the project pin.")

try:
    import torch
    if not torch.__version__.split("+")[0].startswith("2.8.0"):
        print(f"  Expected PyTorch 2.8.0, found {torch.__version__}.")
    elif torch.version.cuda is None:
        print("  PyTorch 2.8.0 is installed as a CPU build. This is fine locally, but AutoDL A800 should report CUDA 12.x here.")
    elif not torch.cuda.is_available():
        print("  PyTorch is a CUDA build, but CUDA is not available in this container.")
        print("  Check that the AutoDL instance was started with GPU resources and that nvidia-smi works in the shell.")
except Exception:
    pass

if missing:
    print("  Missing packages were found. Install into the active cloud environment:")
    print("    python -m pip install -U setuptools wheel -i https://pypi.org/simple")
    print("    python -m pip install --no-build-isolation -e '.[gpu]'")
else:
    print("  Required imports are present.")
PY
