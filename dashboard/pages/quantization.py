"""Quantization analysis page: float vs quant gaps, QAT mode effectiveness."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import mode_boxplot, quant_gap_bar
from dashboard.components.filters import apply_sidebar_filters
from dashboard.config import MODE_COLORS, DEFAULT_COLOR
from dashboard.data_model import RunData


def render(runs: list[RunData]):
    st.header("Quantization Analysis")
    filtered = apply_sidebar_filters(runs)

    if not filtered:
        st.info("No runs match the current filters.")
        return

    # Section 1: Float vs Quant gap chart
    st.subheader("Float vs Quantized BPB Gap")
    st.caption("Sorted by degradation (largest gap first). Smaller gap = better quantization robustness.")
    eligible = [
        r for r in filtered
        if r.final_float_sliding_bpb is not None and r.final_quant_sliding_bpb is not None
    ]
    if eligible:
        fig = quant_gap_bar(eligible)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No runs have both float and quantized sliding metrics.")

    st.markdown("---")

    # Section 2: QAT mode box plots
    st.subheader("BPB Distribution by QAT Mode")
    st.caption("How does each QAT strategy perform across seeds? Lower is better.")
    metric = st.selectbox("Metric", [
        "final_quant_sliding_bpb",
        "final_float_sliding_bpb",
        "final_int6_bpb",
        "post_ema_val_bpb",
    ], format_func=lambda x: x.replace("_", " ").replace("final ", "").title())

    fig = mode_boxplot(filtered, metric=metric)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Section 3: Onset sweep analysis
    st.subheader("QAT Onset Scale vs Degradation")
    st.caption("For int4 runs: does later onset (higher scale) reduce quant degradation?")
    onset_runs = [
        r for r in filtered
        if r.qat_onset_scale is not None and r.quant_degradation is not None
    ]
    if onset_runs:
        fig = go.Figure()
        for r in onset_runs:
            try:
                onset_val = float(r.qat_onset_scale)
            except (ValueError, TypeError):
                continue
            fig.add_trace(go.Scatter(
                x=[onset_val],
                y=[r.quant_degradation],
                mode="markers+text",
                text=[r.display_name],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    color=MODE_COLORS.get(r.mode, DEFAULT_COLOR),
                    size=12,
                ),
                name=r.display_name,
                showlegend=False,
            ))
        fig.update_layout(
            xaxis_title="QAT onset scale",
            yaxis_title="Quant degradation (BPB)",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No runs with both QAT onset scale and quantization metrics.")
