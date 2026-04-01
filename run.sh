#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Run setup first." >&2
  exit 1
fi

source .venv/bin/activate

cmd="${1:-help}"
shift || true

case "$cmd" in
  smoke-download)
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${1:-1}"
    ;;
  smoke-train)
    RUN_ID="${RUN_ID:-smoke_sp1024}" \
    DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}" \
    TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
    VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
    ITERATIONS="${ITERATIONS:-20}" \
    WARMUP_STEPS="${WARMUP_STEPS:-2}" \
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}" \
    TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
    VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-10}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}" \
    TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}" \
    USE_COMPILE="${USE_COMPILE:-0}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train_gpt.py
    ;;
  train-1gpu)
    RUN_ID="${RUN_ID:-baseline_sp1024}" \
    DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}" \
    TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
    VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train_gpt.py
    ;;
  local-baseline)
    RUN_ID="${RUN_ID:-local_baseline_sp1024}" \
    DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}" \
    TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
    VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
    USE_COMPILE="${USE_COMPILE:-0}" \
    ITERATIONS="${ITERATIONS:-1000}" \
    WARMUP_STEPS="${WARMUP_STEPS:-2}" \
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}" \
    TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
    VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}" \
    TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train_gpt.py
    ;;
  parse-log)
    python3 scripts/parse_train_log.py "$@"
    ;;
  python)
    python3 "$@"
    ;;
  shell)
    exec "$SHELL"
    ;;
  help|--help|-h)
    cat <<'EOF'
Usage: ./run.sh <command> [args]

Commands:
  smoke-download [train_shards]   Download cached FineWeb subset (default: 1)
  smoke-train                     Run a small CUDA smoke training job
  train-1gpu                      Run the 1-GPU torch baseline
  local-baseline                  Run a more serious local 1-GPU baseline-ish job
  parse-log <logfile>             Summarize a training log as JSON
  python <args...>                Run python3 inside the venv
  shell                           Open a shell with the venv activated
EOF
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    exit 1
    ;;
esac
