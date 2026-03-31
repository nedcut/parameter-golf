#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${1:-.env.runpod}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  echo "Copy .env.runpod.example to .env.runpod and edit it." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${REMOTE_HOST:?REMOTE_HOST required}"
: "${REMOTE_DIR:?REMOTE_DIR required}"
: "${RUN_ID:?RUN_ID required}"
: "${DATA_PATH:?DATA_PATH required}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH required}"
: "${VOCAB_SIZE:?VOCAB_SIZE required}"
: "${NPROC_PER_NODE:?NPROC_PER_NODE required}"

REMOTE_CMD=$(cat <<EOF
set -e
cd "$REMOTE_DIR"
mkdir -p logs
${EXTRA_ENV:-} \
RUN_ID="$RUN_ID" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE="$VOCAB_SIZE" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py
EOF
)

ssh "$REMOTE_HOST" "$REMOTE_CMD"
