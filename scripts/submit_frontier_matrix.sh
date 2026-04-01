#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="${TARGET:-smoke}"
PROFILE="${PROFILE:-default}"
MATRIX="${MATRIX:-baseline}"
SEEDS=(${SEEDS:-1337 2025 42})
INT4_ONSETS=(${INT4_ONSETS:-0.10 0.15 0.20 0.30})
SHORTLIST_INT4_ONSET="${SHORTLIST_INT4_ONSET:-0.30}"
RUN_GROUP="${RUN_GROUP:-frontier-${TARGET}-$(date +%Y%m%d-%H%M%S)}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DRY_RUN="${DRY_RUN:-0}"

case "$TARGET" in
  smoke)
    JOB_SCRIPT="slurm/train_frontier_smoke_1gpu.sbatch"
    case "$PROFILE" in
      default|quick)
        DEFAULT_ITERATIONS="${ITERATIONS:-200}"
        DEFAULT_WARMDOWN_ITERS="${WARMDOWN_ITERS:-$DEFAULT_ITERATIONS}"
        DEFAULT_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-100}"
        DEFAULT_TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
        ;;
      mid)
        DEFAULT_ITERATIONS="${ITERATIONS:-1000}"
        DEFAULT_WARMDOWN_ITERS="${WARMDOWN_ITERS:-$DEFAULT_ITERATIONS}"
        DEFAULT_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-250}"
        DEFAULT_TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
        ;;
      *)
        echo "Unsupported PROFILE=$PROFILE for TARGET=smoke (expected default, quick, or mid)" >&2
        exit 2
        ;;
    esac
    DEFAULT_PRE_EMA="${FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC:-1}"
    DEFAULT_EVAL_STRIDE="${EVAL_STRIDE:-0}"
    ;;
  full)
    JOB_SCRIPT="slurm/train_frontier_4gpu.sbatch"
    DEFAULT_ITERATIONS="${ITERATIONS:-20000}"
    DEFAULT_WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
    DEFAULT_VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-4000}"
    DEFAULT_TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
    DEFAULT_PRE_EMA="${FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC:-0}"
    DEFAULT_EVAL_STRIDE="${EVAL_STRIDE:-64}"
    ;;
  *)
    echo "Unsupported TARGET=$TARGET (expected smoke or full)" >&2
    exit 2
    ;;
esac

submit_job() {
  local run_id="$1"
  shift
  local exports=(
    "RUN_ID=$run_id"
    "ITERATIONS=$DEFAULT_ITERATIONS"
    "WARMDOWN_ITERS=$DEFAULT_WARMDOWN_ITERS"
    "VAL_LOSS_EVERY=$DEFAULT_VAL_LOSS_EVERY"
    "TRAIN_LOG_EVERY=$DEFAULT_TRAIN_LOG_EVERY"
    "EVAL_STRIDE=$DEFAULT_EVAL_STRIDE"
    "FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC=$DEFAULT_PRE_EMA"
  )
  while (($#)); do
    exports+=("$1")
    shift
  done
  local export_csv
  export_csv="$(IFS=,; echo "${exports[*]}")"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN $SBATCH_BIN --parsable --export=ALL,$export_csv $JOB_SCRIPT"
    return
  fi
  local job_id
  job_id="$("$SBATCH_BIN" --parsable --export="ALL,$export_csv" "$JOB_SCRIPT")"
  printf '%s\t%s\n' "$job_id" "$run_id"
}

submit_baseline() {
  local seed
  for seed in "${SEEDS[@]}"; do
    submit_job "${RUN_GROUP}-noqat-s${seed}" \
      "SEED=$seed" \
      "QAT_BITS=0" \
      "QAT_ENABLED=0" \
      "LATE_QAT_THRESHOLD=0"
    submit_job "${RUN_GROUP}-legacy-int6-s${seed}" \
      "SEED=$seed" \
      "QAT_BITS=0" \
      "QAT_ENABLED=0" \
      "LATE_QAT_THRESHOLD=0.15"
    submit_job "${RUN_GROUP}-int4-o015-s${seed}" \
      "SEED=$seed" \
      "QAT_BITS=4" \
      "QAT_ENABLED=0" \
      "QAT_ONSET_SCALE=0.15" \
      "LATE_QAT_THRESHOLD=0"
  done
}

submit_onset_sweep() {
  local seed onset onset_tag
  for seed in "${SEEDS[@]}"; do
    for onset in "${INT4_ONSETS[@]}"; do
      onset_tag="${onset/./}"
      submit_job "${RUN_GROUP}-int4-o${onset_tag}-s${seed}" \
        "SEED=$seed" \
        "QAT_BITS=4" \
        "QAT_ENABLED=0" \
        "QAT_ONSET_SCALE=$onset" \
        "LATE_QAT_THRESHOLD=0"
    done
  done
}

submit_shortlist() {
  local seed onset_tag
  onset_tag="${SHORTLIST_INT4_ONSET/./}"
  for seed in "${SEEDS[@]}"; do
    submit_job "${RUN_GROUP}-noqat-s${seed}" \
      "SEED=$seed" \
      "QAT_BITS=0" \
      "QAT_ENABLED=0" \
      "LATE_QAT_THRESHOLD=0"
    submit_job "${RUN_GROUP}-legacy-int6-s${seed}" \
      "SEED=$seed" \
      "QAT_BITS=0" \
      "QAT_ENABLED=0" \
      "LATE_QAT_THRESHOLD=0.15"
    submit_job "${RUN_GROUP}-int4-o${onset_tag}-s${seed}" \
      "SEED=$seed" \
      "QAT_BITS=4" \
      "QAT_ENABLED=0" \
      "QAT_ONSET_SCALE=$SHORTLIST_INT4_ONSET" \
      "LATE_QAT_THRESHOLD=0"
  done
}

echo -e "job_id\trun_id"
case "$MATRIX" in
  baseline)
    submit_baseline
    ;;
  shortlist)
    submit_shortlist
    ;;
  onset)
    submit_onset_sweep
    ;;
  full)
    submit_baseline
    submit_onset_sweep
    ;;
  *)
    echo "Unsupported MATRIX=$MATRIX (expected baseline, shortlist, onset, or full)" >&2
    exit 2
    ;;
esac
