"""Shared sidebar filter widgets."""
from __future__ import annotations

import streamlit as st

from ..config import SIZE_LIMIT_BYTES
from ..data_model import RunData


def apply_sidebar_filters(runs: list[RunData]) -> list[RunData]:
    """Render sidebar filters and return the filtered run list."""
    st.sidebar.markdown("### Filters")

    query = st.sidebar.text_input("Search run name", placeholder="Run ID, filename, mode...")

    # QAT mode filter
    all_modes = sorted({r.mode for r in runs})
    selected_modes = st.sidebar.multiselect("QAT mode", all_modes, default=all_modes)

    # Status filter
    all_statuses = sorted({r.status for r in runs})
    selected_statuses = st.sidebar.multiselect("Status", all_statuses, default=all_statuses)

    # Iteration range — separate smoke tests from full runs
    iter_values = [r.iterations for r in runs if r.iterations is not None]
    if iter_values:
        min_iter, max_iter = min(iter_values), max(iter_values)
        if min_iter < max_iter:
            iter_range = st.sidebar.slider(
                "Iterations range",
                min_value=min_iter,
                max_value=max_iter,
                value=(min_iter, max_iter),
            )
        else:
            iter_range = (min_iter, max_iter)
    else:
        iter_range = None

    under_budget_only = st.sidebar.checkbox("Only under 16MB", value=False)
    require_quant_metric = st.sidebar.checkbox("Only with quant sliding BPB", value=False)

    # Apply filters
    filtered = runs
    if query:
        query_lower = query.lower()
        filtered = [
            r for r in filtered
            if query_lower in r.display_name.lower()
            or query_lower in r.mode.lower()
            or query_lower in r.path.lower()
        ]
    filtered = [r for r in filtered if r.mode in selected_modes]
    filtered = [r for r in filtered if r.status in selected_statuses]
    if iter_range:
        filtered = [
            r for r in filtered
            if r.iterations is None or iter_range[0] <= r.iterations <= iter_range[1]
        ]
    if under_budget_only:
        filtered = [
            r for r in filtered
            if r.submission_size_bytes is not None and r.submission_size_bytes <= SIZE_LIMIT_BYTES
        ]
    if require_quant_metric:
        filtered = [r for r in filtered if r.final_quant_sliding_bpb is not None]

    st.sidebar.markdown(f"**{len(filtered)}** / {len(runs)} runs")
    return filtered
