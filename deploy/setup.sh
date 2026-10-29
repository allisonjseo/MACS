#!/usr/bin/env bash
# Run once in the remote checkout, before submitting GPU jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
uv sync --locked --extra vision
.venv/bin/python -c 'import torch, timm; print({"torch": torch.__version__, "timm": timm.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()})'
