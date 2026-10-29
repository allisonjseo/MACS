#!/usr/bin/env bash
# Fetch timing/metrics/W&B records and sync using credentials on this machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
host=${BABEL_HOST:-cyrils@login.babel.cs.cmu.edu}
remote=${BABEL_OUTPUT:-/home/cyrils/macs/runs}
local_output=${1:-runs/babel}
identity=${BABEL_IDENTITY:-$HOME/.ssh/id_ed25519}
mkdir -p "$local_output"
printf -v remote_shell '%q ' ssh -i "$identity" -o BatchMode=yes
rsync -az -e "$remote_shell" --exclude='*.pt' "$host:$remote/" "$local_output/"
.venv/bin/python deploy/sync_babble.py "$local_output"
