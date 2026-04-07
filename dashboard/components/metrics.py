"""Reusable metric display components."""
from __future__ import annotations

import streamlit as st

from ..config import SOTA_BPB


def metric_card(label: str, value: float | int | None, fmt: str = ".4f", reference: float | None = None):
    """Display a metric with optional delta from a reference value."""
    if value is None:
        st.metric(label, "N/A")
        return
    formatted = f"{value:{fmt}}"
    if reference is not None:
        delta = value - reference
        st.metric(label, formatted, delta=f"{delta:+.4f}", delta_color="inverse")
    else:
        st.metric(label, formatted)


def bpb_metric(label: str, bpb: float | None):
    """Display a BPB metric with delta from SOTA."""
    metric_card(label, bpb, fmt=".4f", reference=SOTA_BPB)


def status_badge(status: str) -> str:
    """Return a colored status indicator string."""
    colors = {"complete": "🟢", "crashed": "🔴", "incomplete": "🟡"}
    return f"{colors.get(status, '⚪')} {status}"


def kpi_row(runs: list):
    """Display top-level KPI cards across columns."""
    from ..data_model import RunData
    complete = [r for r in runs if r.status == "complete"]

    # Best BPB (quant sliding, the competition metric)
    bpb_values = [r.final_quant_sliding_bpb for r in runs if r.final_quant_sliding_bpb is not None]
    best_bpb = min(bpb_values) if bpb_values else None

    # Runs under size limit
    from ..config import SIZE_LIMIT_BYTES
    under_budget = sum(1 for r in runs if r.submission_size_bytes and r.submission_size_bytes <= SIZE_LIMIT_BYTES)

    cols = st.columns(4)
    with cols[0]:
        bpb_metric("Best BPB (quant sliding)", best_bpb)
    with cols[1]:
        mean_bpb = sum(bpb_values) / len(bpb_values) if bpb_values else None
        metric_card("Mean BPB", mean_bpb, fmt=".4f")
    with cols[2]:
        metric_card("Under 16MB", under_budget, fmt="d")
    with cols[3]:
        metric_card("Complete runs", len(complete), fmt="d")
