"""Data structures for parsed training runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TrainStep:
    step: int
    total_steps: int
    train_loss: float
    train_time_ms: float
    step_avg_ms: float


@dataclass
class ValCheckpoint:
    step: int
    total_steps: int
    val_loss: float
    val_bpb: float
    train_time_ms: float


@dataclass
class RunEvent:
    step: int
    label: str  # e.g. "swa:start", "qat:enabled", "ema:applying"


@dataclass
class RunData:
    # Identity
    path: str
    run_id: str | None
    mode: str  # from infer_mode()
    seed: int | None
    status: Literal["complete", "crashed", "incomplete"]

    # Config
    iterations: int | None
    warmdown_iters: int | None
    model_params: int | None
    qat_bits: int | None
    qat_onset_scale: str | None

    # Summary metrics (from parse_log)
    step200_val_bpb: float | None
    post_ema_val_bpb: float | None
    final_float_fixed_bpb: float | None
    final_float_sliding_bpb: float | None
    final_quant_fixed_bpb: float | None
    final_quant_sliding_bpb: float | None
    final_int6_bpb: float | None
    pre_ema_int6_bpb: float | None
    submission_size_bytes: int | None
    label: str | None = None

    # Time-series
    train_steps: list[TrainStep] = field(default_factory=list)
    val_checkpoints: list[ValCheckpoint] = field(default_factory=list)
    events: list[RunEvent] = field(default_factory=list)

    # Raw summary dict for anything we didn't explicitly extract
    raw_summary: dict = field(default_factory=dict)

    @property
    def quant_degradation(self) -> float | None:
        """BPB gap between float sliding and quantized sliding (positive = worse after quant)."""
        if self.final_float_sliding_bpb is not None and self.final_quant_sliding_bpb is not None:
            return self.final_quant_sliding_bpb - self.final_float_sliding_bpb
        return None

    @property
    def base_name(self) -> str:
        """Canonical run label before any deduplication suffix is added."""
        if self.run_id:
            return self.run_id
        return Path(self.path).stem

    @property
    def size_pct(self) -> float | None:
        """Submission size as percentage of 16MB limit."""
        from .config import SIZE_LIMIT_BYTES
        if self.submission_size_bytes is not None:
            return self.submission_size_bytes / SIZE_LIMIT_BYTES * 100
        return None

    @property
    def wallclock_seconds(self) -> float | None:
        """Total training wallclock from last train_time_ms."""
        if self.train_steps:
            return self.train_steps[-1].train_time_ms / 1000
        return None

    @property
    def early_val_bpb(self) -> float | None:
        """Earliest meaningful validation BPB: prefers step200 summary, then first non-zero checkpoint."""
        if self.step200_val_bpb is not None:
            return self.step200_val_bpb
        for checkpoint in self.val_checkpoints:
            if checkpoint.step > 0:
                return checkpoint.val_bpb
        if self.val_checkpoints:
            return self.val_checkpoints[0].val_bpb
        return None

    @property
    def progress_pct(self) -> float | None:
        """Approximate training progress from the last logged step."""
        if self.iterations and self.train_steps:
            return min(100.0, self.train_steps[-1].step / self.iterations * 100)
        if self.status == "complete" and self.iterations is not None:
            return 100.0
        return None

    @property
    def display_name(self) -> str:
        """Short display name for charts."""
        return self.label or self.base_name
