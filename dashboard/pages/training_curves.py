"""Training curves page: multi-run loss overlay with val checkpoints."""
from __future__ import annotations

import streamlit as st

from dashboard.components.charts import loss_curve
from dashboard.components.filters import apply_sidebar_filters
from dashboard.data_model import RunData


def render(runs: list[RunData]):
    st.header("Training Curves")
    filtered = apply_sidebar_filters(runs)

    # Only runs with time-series data
    with_data = [r for r in filtered if r.train_steps]
    if not with_data:
        st.info("No runs with training step data match the current filters.")
        return

    # Run selector — default to top 5 by final quant sliding BPB
    sorted_runs = sorted(
        with_data,
        key=lambda r: r.final_quant_sliding_bpb if r.final_quant_sliding_bpb is not None else 999,
    )
    default_names = [r.display_name for r in sorted_runs[:5]]
    all_names = [r.display_name for r in sorted_runs]

    selected = st.multiselect(
        "Select runs to plot",
        all_names,
        default=default_names,
    )
    selected_runs = [r for r in sorted_runs if r.display_name in selected]

    if not selected_runs:
        st.info("Select at least one run to plot.")
        return

    # Controls
    col1, col2, col3 = st.columns(3)
    show_val = col1.checkbox("Show val_bpb checkpoints", value=True)
    log_y = col2.checkbox("Log scale Y-axis", value=False)

    # Tabs for step vs wallclock
    tab_step, tab_time = st.tabs(["By Step", "By Wallclock"])

    with tab_step:
        fig = loss_curve(selected_runs, by_time=False, show_val=show_val, log_y=log_y)
        st.plotly_chart(fig, use_container_width=True)

    with tab_time:
        fig = loss_curve(selected_runs, by_time=True, show_val=show_val, log_y=log_y)
        st.plotly_chart(fig, use_container_width=True)

    # Summary table for selected runs
    st.markdown("#### Selected runs summary")
    summary_data = []
    for r in selected_runs:
        summary_data.append({
            "Run": r.display_name,
            "Mode": r.mode,
            "Steps": f"{len(r.train_steps)} logged",
            "Final train loss": f"{r.train_steps[-1].train_loss:.4f}" if r.train_steps else "N/A",
            "Final val BPB": f"{r.val_checkpoints[-1].val_bpb:.4f}" if r.val_checkpoints else "N/A",
            "Wallclock": f"{r.wallclock_seconds:.0f}s" if r.wallclock_seconds else "N/A",
        })
    st.dataframe(summary_data, use_container_width=True)
