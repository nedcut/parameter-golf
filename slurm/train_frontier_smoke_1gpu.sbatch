#!/usr/bin/env bash
#SBATCH --job-name=pg-frontier-smoke
#SBATCH --partition=gpu-short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=slurm/output/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
BASE_STORAGE="${STORAGE:-${SCRATCH:-$HOME}}"
DATA_ROOT="${DATA_ROOT:-$BASE_STORAGE/parameter-golf-data}"
RUN_ROOT="${RUN_ROOT:-$BASE_STORAGE/parameter-golf-runs}"
RUN_ID="${RUN_ID:-frontier-smoke-${SLURM_JOB_ID}}"
VARIANT="${VARIANT:-sp1024}"
FRONTIER_REL_DIR="records/track_10min_16mb/2026-03-26_11L_PreTTT_Frontier_Int4QAT"

DATA_PATH="${DATA_PATH:-$DATA_ROOT/datasets/fineweb10B_${VARIANT}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DATA_ROOT/tokenizers/fineweb_1024_bpe.model}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
SCRIPT_PATH="$REPO_ROOT/$FRONTIER_REL_DIR/train_gpt.py"

cd "$REPO_ROOT"

if [[ -x ".conda/bin/python" && -x ".conda/bin/torchrun" ]]; then
    # Prefer the repo-local conda env on cluster nodes because it carries Python headers.
    PYTHON="$REPO_ROOT/.conda/bin/python"
    TORCHRUN="$REPO_ROOT/.conda/bin/torchrun"
elif [[ -x ".venv/bin/python" && -x ".venv/bin/torchrun" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
    TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"
else
    PYTHON="$(command -v python)"
    TORCHRUN="$(command -v torchrun)"
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
export SEED="${SEED:-1337}"
export ITERATIONS="${ITERATIONS:-200}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export EVAL_STRIDE="${EVAL_STRIDE:-0}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-$ITERATIONS}"
export QAT_BITS="${QAT_BITS:-0}"
export QAT_ENABLED="${QAT_ENABLED:-0}"
export QAT_BLOCK_SIZE="${QAT_BLOCK_SIZE:-128}"
export QAT_ONSET_SCALE="${QAT_ONSET_SCALE:-0.15}"
export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0}"
export FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC="${FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC:-1}"
export FRONTIER_PRE_EMA_SNAPSHOT_PATH="${FRONTIER_PRE_EMA_SNAPSHOT_PATH:-final_model_pre_ema.pt}"

echo "Repo root: $REPO_ROOT"
echo "Run dir: $RUN_DIR"
echo "Frontier dir: $FRONTIER_REL_DIR"
echo "Train script: $SCRIPT_PATH"
echo "Data path: $DATA_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "EVAL_STRIDE: $EVAL_STRIDE"
echo "ITERATIONS: $ITERATIONS"
echo "WARMDOWN_ITERS: $WARMDOWN_ITERS"
echo "QAT_BITS: $QAT_BITS"
echo "QAT_ENABLED: $QAT_ENABLED"
echo "QAT_ONSET_SCALE: $QAT_ONSET_SCALE"
echo "LATE_QAT_THRESHOLD: $LATE_QAT_THRESHOLD"
echo "QAT_BLOCK_SIZE: $QAT_BLOCK_SIZE"
echo "FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC: $FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC"
echo "FRONTIER_PRE_EMA_SNAPSHOT_PATH: $FRONTIER_PRE_EMA_SNAPSHOT_PATH"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "Torchrun: $TORCHRUN"
echo "Starting: $(date)"

"$TORCHRUN" \
    --standalone \
    --nproc_per_node=1 \
    "$SCRIPT_PATH"

echo "Finished: $(date)"
