#!/usr/bin/env bash
set -euo pipefail
# All resource settings are supplied and printed by bridges2.sh.
: "${SLURM_ARRAY_TASK_ID:?Submit using deploy/bridges2.sh}"
: "${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"
models=(resnet swin dino)
model=${models[$SLURM_ARRAY_TASK_ID]}
manifest=$1
output_root=$2
shift 2
bash deploy/run.sh "$model" "$manifest" "$output_root/$SLURM_ARRAY_JOB_ID-$model" "$@"
