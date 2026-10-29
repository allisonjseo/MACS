#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with deploy/babble.sh}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec "${PYTHON:-.venv/bin/python}" deploy/babble_job.py "$@"
