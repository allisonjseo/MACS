#!/usr/bin/env bash
# Submit from the checkout on bridges2 (SSH alias: csharma2@bridges2.psc.edu).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 2 )); then
    echo 'Usage: PSC_ACCOUNT=allocation bash deploy/bridges2.sh MANIFEST OUTPUT_ROOT [train.py fit options]' >&2
    exit 2
fi
: "${PSC_ACCOUNT:?Set PSC_ACCOUNT to your Bridges-2 GPU allocation}"
manifest=$1
output_root=$2
shift 2
mkdir -p "$output_root/logs"
options=(
    --account="$PSC_ACCOUNT"
    --partition="${PARTITION:-GPU-shared}"
    --gpus="${GPU_TYPE:-h100-80}:1"
    --ntasks=1
    --cpus-per-task="${CPUS:-8}"
    --mem="${MEMORY:-32G}"
    --time="${WALLTIME:-04:00:00}"
    --array=0-2
    --job-name=macs-vision
    --chdir="$PWD"
    --output="$output_root/logs/%A_%a.out"
    --error="$output_root/logs/%A_%a.err"
)
printf 'Scheduler configuration:'
printf ' %q' "${options[@]}"
printf '\n'
# Resolve and print each actual training config before submitting any job.
python=${PYTHON:-.venv/bin/python}
for model in resnet swin dino; do
    "$python" train.py fit --config "configs/$model.toml" --manifest "$manifest" \
        --output "$output_root/SLURM_JOB_ID-$model" "$@" --dry-run
done
sbatch "${options[@]}" deploy/bridges2.sbatch "$manifest" "$output_root" "$@"
