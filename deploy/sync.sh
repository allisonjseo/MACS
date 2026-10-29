#!/usr/bin/env bash
# Copy code only. Transfer data separately or prepare it on the target machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# != 2 )); then
    echo 'Usage: bash deploy/sync.sh SSH_ALIAS /absolute/remote/project/path' >&2
    exit 2
fi
host=$1
remote=$2
# These values also pass through the remote shell; accept simple paths/aliases.
[[ $host =~ ^[a-zA-Z0-9][a-zA-Z0-9._@-]*$ ]] || { echo 'Invalid SSH alias' >&2; exit 2; }
[[ $remote =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Use an absolute remote path without spaces' >&2; exit 2; }
ssh "$host" "mkdir -p '$remote'"
rsync -az --exclude='.git/' --exclude='.venv/' --exclude='data' --exclude='runs' \
    --exclude='__pycache__/' --exclude='.*_cache/' --exclude='.DS_Store' \
    --exclude='.env*' --exclude='.vscode/' --exclude='.idea/' ./ "$host:$remote/"
