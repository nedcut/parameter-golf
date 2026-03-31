#!/usr/bin/env bash
#SBATCH --job-name=pg-data
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
BASE_STORAGE="${STORAGE:-${SCRATCH:-$HOME}}"
DATA_ROOT="${DATA_ROOT:-$BASE_STORAGE/parameter-golf-data}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
VARIANT="${VARIANT:-sp1024}"

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

mkdir -p "$DATA_ROOT/datasets"
mkdir -p "$DATA_ROOT/tokenizers"

link_into_data() {
    local target="$1"
    local link_path="$2"
    if [[ -L "$link_path" ]]; then
        local current_target
        current_target="$(readlink "$link_path")"
        if [[ "$current_target" == "$target" ]]; then
            return
        fi
        rm "$link_path"
    elif [[ -e "$link_path" ]]; then
        echo "Refusing to replace non-symlink path: $link_path" >&2
        exit 1
    fi
    ln -s "$target" "$link_path"
}

link_into_data "$DATA_ROOT/datasets" data/datasets
link_into_data "$DATA_ROOT/tokenizers" data/tokenizers

export HF_HOME="${HF_HOME:-$DATA_ROOT/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$DATA_ROOT/hf-cache}"

echo "Repo root: $REPO_ROOT"
echo "Data root: $DATA_ROOT"
echo "Train shards: $TRAIN_SHARDS"
echo "Variant: $VARIANT"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "Starting: $(date)"

"$PYTHON" data/cached_challenge_fineweb.py \
    --variant "$VARIANT" \
    --train-shards "$TRAIN_SHARDS"

echo "Finished: $(date)"
