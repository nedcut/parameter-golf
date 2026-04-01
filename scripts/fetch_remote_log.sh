#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${1:-.env.runpod}"
RUN_ID="${2:-}"
if [[ -z "$RUN_ID" ]]; then
  echo "usage: $0 [env-file] <run-id>" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
: "${REMOTE_HOST:?REMOTE_HOST required}"
: "${REMOTE_DIR:?REMOTE_DIR required}"

mkdir -p remote-logs
scp "$REMOTE_HOST:$REMOTE_DIR/logs/$RUN_ID.txt" "remote-logs/$RUN_ID.txt"
echo "Saved remote-logs/$RUN_ID.txt"
