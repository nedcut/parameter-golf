"""Overview page: sortable table of all runs with KPI cards."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.filters import apply_sidebar_filters
from dashboard.components.metrics import kpi_row, status_badge
from dashboard.config import SIZE_LIMIT_BYTES, SOTA_BPB
from dashboard.data_model import RunData


def render(runs: list[RunData]):
    st.header("Run Overview")
    filtered = apply_sidebar_filters(runs)

    kpi_row(filtered)
    st.markdown("---")

    if not filtered:
        st.info("No runs match the current filters.")
        return

    # Build DataFrame
    rows = []
    for r in filtered:
        rows.append({
            "Status": status_badge(r.status),
            "Run ID": r.display_name,
            "Mode": r.mode,
            "Seed": r.seed,
            "Iters": r.iterations,
            "Float sliding": r.final_float_sliding_bpb,
            "Quant sliding": r.final_quant_sliding_bpb,
            "Quant gap": r.quant_degradation,
            "Int6 BPB": r.final_int6_bpb,
            "Size (MB)": round(r.submission_size_bytes / 1e6, 2) if r.submission_size_bytes is not None else None,
            "Size %": r.size_pct,
            "Params (M)": round(r.model_params / 1e6, 1) if r.model_params is not None else None,
            "Progress %": r.progress_pct,
        })

    df = pd.DataFrame(rows).sort_values(
        by=["Quant sliding", "Float sliding", "Run ID"],
        ascending=[True, True, True],
        na_position="last",
    )

    # Highlight styling
    def style_row(row):
        styles = [""] * len(row)
        # Red if over budget
        size_idx = df.columns.get_loc("Size (MB)")
        if row["Size (MB)"] is not None and row["Size (MB)"] > SIZE_LIMIT_BYTES / 1e6:
            styles[size_idx] = "background-color: rgba(255, 80, 80, 0.3)"
        # Green tint if quant sliding beats SOTA
        qs_idx = df.columns.get_loc("Quant sliding")
        if row["Quant sliding"] is not None and row["Quant sliding"] < SOTA_BPB:
            styles[qs_idx] = "background-color: rgba(0, 204, 150, 0.3)"
        return styles

    styled = df.style.apply(style_row, axis=1)
    st.dataframe(styled, use_container_width=True, height=min(700, 45 + 35 * len(df)))

    st.markdown("#### Open a run")
    selected_run = st.selectbox(
        "Jump to Run Detail",
        [r.display_name for r in filtered],
        index=None,
        placeholder="Select a run...",
        key="overview_run_picker",
    )
    if st.button("Open Run Detail", disabled=selected_run is None):
        st.session_state["selected_run"] = selected_run
        st.session_state["page_name"] = "Run Detail"
        st.rerun()
