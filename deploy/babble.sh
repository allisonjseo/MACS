#!/usr/bin/env bash
# Foreground, single-GPU launcher; no scheduler/hostname assumptions about Babble.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 2 )); then
    echo 'Usage: bash deploy/babble.sh MANIFEST OUTPUT_ROOT [train.py fit options]' >&2
    echo 'Run on Babble, inside a GPU allocation if that system requires one.' >&2
    exit 2
fi
manifest=$1
output_root=$2
shift 2
run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
# Run serially on one visible GPU; each model receives its own run directory.
for model in resnet swin dino; do
    bash deploy/run.sh "$model" "$manifest" "$output_root/$run_id-$model" "$@"
done
