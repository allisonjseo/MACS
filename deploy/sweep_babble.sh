#!/usr/bin/env bash
# Three independent Optuna studies, one allocated GPU per frozen encoder.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# != 3 )); then
    echo 'Usage: bash deploy/sweep_babble.sh MANIFEST REVIEWED_PLAN OUTPUT_ROOT' >&2
    exit 2
fi
manifest=$1
plan=$2
output=$3
mkdir -p "$output/logs"
for backbone in resnet swin dino; do
    sbatch --parsable --account=agu --partition=preempt --qos=preempt_qos \
        --gres=gpu:1 --constraint=L40S --ntasks=1 --cpus-per-task=4 --mem=16G \
        --time=02:00:00 --requeue --open-mode=append --job-name="macs-optuna-$backbone" \
        --chdir="$PWD" --output="$output/logs/%j.out" --error="$output/logs/%j.err" \
        deploy/sweep_babble.sbatch "$manifest" "$plan" "$output/$backbone" "$backbone"
done
