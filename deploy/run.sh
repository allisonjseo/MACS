#!/usr/bin/env bash
# Shared foreground launcher: run on a GPU workstation or inside an allocation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 3 )); then
    echo 'Usage: bash deploy/run.sh MODEL MANIFEST OUTPUT [train.py fit options]' >&2
    exit 2
fi
model=$1
manifest=$2
output=$3
shift 3
case "$model" in resnet|swin|dino) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
python=${PYTHON:-.venv/bin/python}
export PYTHONUNBUFFERED=1
printf 'Host: %s; CUDA_VISIBLE_DEVICES: %s; SLURM_JOB_ID: %s\n' "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}" "${SLURM_JOB_ID:-none}"
command=("$python" train.py fit --config "configs/$model.toml" --manifest "$manifest" --output "$output" "$@")
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
