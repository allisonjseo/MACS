#!/usr/bin/env bash
# Submit on Babel: cyrils@login.babel.cs.cmu.edu.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 2 )); then
    echo 'Usage: bash deploy/babble.sh MANIFEST OUTPUT_ROOT [train.py fit options]' >&2
    exit 2
fi
manifest=$1
output_root=$2
shift 2
mkdir -p "$output_root/logs"
python=${PYTHON:-.venv/bin/python}
config_suffix=${MACS_CONFIG_SUFFIX:-}
case "$config_suffix" in ''|-gaussian) ;; *) echo 'Invalid MACS_CONFIG_SUFFIX' >&2; exit 2 ;; esac
options=(
    --parsable --account="${BABEL_ACCOUNT:-agu}"
    --partition="${PARTITION:-preempt}" --qos="${QOS:-preempt_qos}"
    --gres=gpu:1 --constraint="${GPU_CONSTRAINT:-H200}"
    --ntasks=1 --cpus-per-task="${CPUS:-4}" --mem="${MEMORY:-32G}"
    --time="${WALLTIME:-04:00:00}" --no-requeue --job-name=macs-vision
    --chdir="$PWD" --output="$output_root/logs/%j.out" --error="$output_root/logs/%j.err"
)
printf 'Scheduler configuration:'
printf ' %q' "${options[@]}"
printf '\n'
read -r -a models <<< "${MACS_MODELS:-resnet swin dino}"
for model in "${models[@]}"; do
    case "$model" in resnet|swin|dino) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
    "$python" train.py fit --config "configs/$model$config_suffix.toml" --manifest "$manifest" \
        --output "$output_root/SLURM_JOB_ID-$model" "$@" --dry-run
done
sbatch "${options[@]}" deploy/babble.sbatch "$manifest" "$output_root" "$@"
