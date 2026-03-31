#!/usr/bin/env bash
#SBATCH --job-name=pg-smoke
#SBATCH --partition=gpu-short
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
BASE_STORAGE="${STORAGE:-${SCRATCH:-$HOME}}"
DATA_ROOT="${DATA_ROOT:-$BASE_STORAGE/parameter-golf-data}"
RUN_ROOT="${RUN_ROOT:-$BASE_STORAGE/parameter-golf-runs}"
RUN_ID="${RUN_ID:-smoke-${SLURM_JOB_ID}}"
VARIANT="${VARIANT:-sp1024}"

DATA_PATH="${DATA_PATH:-$DATA_ROOT/datasets/fineweb10B_${VARIANT}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DATA_ROOT/tokenizers/fineweb_1024_bpe.model}"
RUN_DIR="$RUN_ROOT/$RUN_ID"

cd "$REPO_ROOT"

if [[ -x ".conda/bin/python" ]]; then
    # Prefer the repo-local conda env on cluster nodes because it carries Python headers.
    PYTHON="$REPO_ROOT/.conda/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
else
    PYTHON="$(command -v python)"
fi
export PATH="$(dirname "$PYTHON"):$PATH"

mkdir -p "$RUN_DIR"
if [[ -n "${SLURM_JOB_SCRIPT:-}" && -f "${SLURM_JOB_SCRIPT}" ]]; then
    cp "${SLURM_JOB_SCRIPT}" "$RUN_DIR/"
fi

cd "$RUN_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export RUN_ID
export DATA_PATH
export TOKENIZER_PATH
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export ITERATIONS="${ITERATIONS:-200}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

echo "Repo root: $REPO_ROOT"
echo "Run dir: $RUN_DIR"
echo "Data path: $DATA_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "Starting: $(date)"

"$PYTHON" "$REPO_ROOT/train_gpt.py"

echo "Finished: $(date)"
