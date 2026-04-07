"""Size budget page: artifact size vs 16MB limit visualization."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import size_scatter
from dashboard.components.filters import apply_sidebar_filters
from dashboard.config import SIZE_LIMIT_BYTES
from dashboard.data_model import RunData


def render(runs: list[RunData]):
    st.header("Size Budget")
    filtered = apply_sidebar_filters(runs)

    eligible = [r for r in filtered if r.submission_size_bytes is not None]
    if not eligible:
        st.info("No runs with submission size data match the current filters.")
        return

    # Main scatter plot
    st.subheader("Size vs BPB Tradeoff")
    st.caption("Ideal runs are in the bottom-left quadrant (under budget, low BPB).")
    scatter_eligible = [r for r in eligible if r.final_quant_sliding_bpb is not None]
    if scatter_eligible:
        fig = size_scatter(scatter_eligible)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Headroom table
    st.subheader("Size Headroom")
    st.caption("How much room remains before hitting the 16MB limit.")
    rows = []
    for r in sorted(eligible, key=lambda r: r.submission_size_bytes, reverse=True):
        headroom = SIZE_LIMIT_BYTES - r.submission_size_bytes
        rows.append({
            "Run": r.display_name,
            "Mode": r.mode,
            "Size (MB)": round(r.submission_size_bytes / 1e6, 2),
            "Budget used": f"{r.size_pct:.1f}%",
            "Headroom (KB)": round(headroom / 1024, 1),
            "Over budget": "YES" if headroom < 0 else "",
        })

    df = pd.DataFrame(rows)

    def style_headroom(row):
        styles = [""] * len(row)
        if row["Over budget"] == "YES":
            return ["background-color: rgba(255, 80, 80, 0.3)"] * len(row)
        headroom_kb = row["Headroom (KB)"]
        if headroom_kb < 500:
            styles[df.columns.get_loc("Headroom (KB)")] = "background-color: rgba(255, 165, 0, 0.3)"
        return styles

    styled = df.style.apply(style_headroom, axis=1)
    st.dataframe(styled, use_container_width=True)
