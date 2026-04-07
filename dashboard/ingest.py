"""Data ingestion: wraps existing parse_log() and adds time-series extraction."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

from .config import CACHE_TTL, MLX_GLOB, MLX_LOG_DIR, PROJECT_ROOT, SLURM_GLOB, SLURM_LOG_DIR
from .data_model import RunData, RunEvent, TrainStep, ValCheckpoint

# Import parse_log and helpers from existing scripts
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from summarize_frontier_logs import parse_log, parse_pairs  # noqa: E402

STEP_PAT = re.compile(r"^step:(\d+)/(\d+)\s")
MODEL_PARAMS_PAT = re.compile(r"^model_params:(\d+)")
_UNIT_SUFFIX = re.compile(r"[a-zA-Z%]+$")


def _to_float(val: object, default: float = 0.0) -> float:
    """Convert a value to float, stripping unit suffixes like 'ms', 'MB', '%'.

    Returns *default* for empty/unparseable strings.  Propagates nan/inf
    rather than silently converting them to 0.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("nan", "inf", "-inf", "+inf"):
            return float(low)
        cleaned = _UNIT_SUFFIX.sub("", val)
        try:
            return float(cleaned) if cleaned else default
        except ValueError:
            return default
    return default


def _extract_time_series(path: Path) -> tuple[list[TrainStep], list[ValCheckpoint], list[RunEvent], int | None, str]:
    """Second pass over raw log to extract step-by-step data, events, and run status."""
    train_steps: list[TrainStep] = []
    val_checkpoints: list[ValCheckpoint] = []
    events: list[RunEvent] = []
    model_params: int | None = None
    has_finished = False
    has_traceback = False

    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()

        # Step lines (training loss or validation)
        m = STEP_PAT.match(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            kv = parse_pairs(line)

            if "train_loss" in kv and "val_bpb" not in kv:
                train_steps.append(TrainStep(
                    step=step,
                    total_steps=total,
                    train_loss=_to_float(kv["train_loss"]),
                    train_time_ms=_to_float(kv.get("train_time", 0)),
                    step_avg_ms=_to_float(kv.get("step_avg", 0)),
                ))
            if "val_bpb" in kv:
                val_checkpoints.append(ValCheckpoint(
                    step=step,
                    total_steps=total,
                    val_loss=_to_float(kv.get("val_loss", 0)),
                    val_bpb=_to_float(kv["val_bpb"]),
                    train_time_ms=_to_float(kv.get("train_time", 0)),
                ))
            continue

        # Model params
        mp = MODEL_PARAMS_PAT.match(line)
        if mp:
            model_params = int(mp.group(1))
            continue

        # Events
        if line.startswith("swa:start"):
            step_m = re.search(r"step:(\d+)", line)
            events.append(RunEvent(step=int(step_m.group(1)) if step_m else 0, label="SWA start"))
        elif line.startswith("qat:enabled") or line.startswith("late_qat:enabled"):
            step_m = re.search(r"step:(\d+)", line) or re.search(r"step[=:](\d+)", line)
            events.append(RunEvent(step=int(step_m.group(1)) if step_m else 0, label="QAT enabled"))
        elif line.startswith("ema:applying"):
            events.append(RunEvent(step=train_steps[-1].step if train_steps else 0, label="EMA applied"))

        # Status detection
        if line.startswith("Finished:"):
            has_finished = True
        if "Traceback" in line or "CANCELLED" in line:
            has_traceback = True

    if has_finished:
        status = "complete"
    elif has_traceback:
        status = "crashed"
    else:
        status = "incomplete"

    return train_steps, val_checkpoints, events, model_params, status


def parse_run(path: Path) -> RunData:
    """Parse a single log file into a RunData object."""
    summary = parse_log(path)
    train_steps, val_checkpoints, events, model_params, status = _extract_time_series(path)

    return RunData(
        path=summary["path"],
        run_id=summary.get("run_id"),
        mode=summary.get("mode", "unknown"),
        seed=summary.get("seed"),
        status=status,
        iterations=summary.get("iterations"),
        warmdown_iters=summary.get("warmdown_iters"),
        model_params=model_params or summary.get("model_params"),
        qat_bits=summary.get("qat_bits"),
        qat_onset_scale=summary.get("qat_onset_scale"),
        step200_val_bpb=summary.get("step200_val_bpb"),
        post_ema_val_bpb=summary.get("post_ema_val_bpb"),
        final_float_fixed_bpb=summary.get("final_float_fixed_bpb"),
        final_float_sliding_bpb=summary.get("final_float_sliding_bpb"),
        final_quant_fixed_bpb=summary.get("final_quant_fixed_bpb"),
        final_quant_sliding_bpb=summary.get("final_quant_sliding_bpb"),
        final_int6_bpb=summary.get("final_int6_bpb"),
        pre_ema_int6_bpb=summary.get("pre_ema_int6_bpb"),
        submission_size_bytes=summary.get("submission_size_bytes"),
        train_steps=train_steps,
        val_checkpoints=val_checkpoints,
        events=events,
        raw_summary=summary,
    )


def _discover_logs() -> list[Path]:
    """Find all log files across known directories."""
    paths: list[Path] = []
    if SLURM_LOG_DIR.is_dir():
        paths.extend(sorted(SLURM_LOG_DIR.glob(SLURM_GLOB)))
    if MLX_LOG_DIR.is_dir():
        paths.extend(sorted(MLX_LOG_DIR.glob(MLX_GLOB)))
    return paths


def _assign_display_labels(runs: list[RunData]) -> None:
    """Disambiguate duplicate run labels so selectors map to a single run."""
    counts = Counter(run.base_name for run in runs)
    for run in runs:
        if counts[run.base_name] == 1:
            run.label = run.base_name
            continue
        run.label = f"{run.base_name} [{Path(run.path).name}]"


@st.cache_data(ttl=CACHE_TTL)
def load_all_runs() -> list[RunData]:
    """Load and parse all training runs."""
    paths = _discover_logs()
    runs = []
    for p in paths:
        try:
            runs.append(parse_run(p))
        except Exception as e:
            st.warning(f"Failed to parse {p.name}: {e}")
    _assign_display_labels(runs)
    return runs
