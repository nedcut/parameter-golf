#!/usr/bin/env bash
#SBATCH --job-name=pg-4gpu
#SBATCH --partition=gpu-standard
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurm/output/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
BASE_STORAGE="${STORAGE:-${SCRATCH:-$HOME}}"
DATA_ROOT="${DATA_ROOT:-$BASE_STORAGE/parameter-golf-data}"
RUN_ROOT="${RUN_ROOT:-$BASE_STORAGE/parameter-golf-runs}"
RUN_ID="${RUN_ID:-train4-${SLURM_JOB_ID}}"
VARIANT="${VARIANT:-sp1024}"

DATA_PATH="${DATA_PATH:-$DATA_ROOT/datasets/fineweb10B_${VARIANT}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DATA_ROOT/tokenizers/fineweb_1024_bpe.model}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
RUN_DIR="$RUN_ROOT/$RUN_ID"

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
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Repo root: $REPO_ROOT"
echo "Run dir: $RUN_DIR"
echo "Data path: $DATA_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "Torchrun: $TORCHRUN"
echo "Starting: $(date)"

"$TORCHRUN" \
    --standalone \
    --nproc_per_node="$GPUS_PER_NODE" \
    "$REPO_ROOT/train_gpt.py"

echo "Finished: $(date)"
