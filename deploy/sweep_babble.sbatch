#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with deploy/sweep_babble.sh}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="$PWD/../cache/huggingface"
export TORCH_HOME="$PWD/../cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PWD/../cache/inductor/sweep-$SLURM_JOB_ID"
export WANDB_CACHE_DIR="$PWD/../cache/wandb"
exec .venv/bin/python sweep.py run "$2" "$4" "$1" "$3"
