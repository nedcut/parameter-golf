"""Run detail page: single-run drill-down with config, curves, and final metrics."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import loss_curve
from dashboard.components.metrics import bpb_metric, metric_card
from dashboard.config import MODE_COLORS, DEFAULT_COLOR, SIZE_LIMIT_BYTES
from dashboard.data_model import RunData


def render(runs: list[RunData]):
    st.header("Run Detail")

    if not runs:
        st.info("No runs available.")
        return

    names = [r.display_name for r in runs]
    default_idx = 0

    # If navigated from overview via session_state
    if "selected_run" in st.session_state and st.session_state["selected_run"] in names:
        default_idx = names.index(st.session_state["selected_run"])

    selected_name = st.selectbox("Select run", names, index=default_idx)
    run = next(r for r in runs if r.display_name == selected_name)

    # Status and identity
    status_colors = {"complete": "green", "crashed": "red", "incomplete": "orange"}
    st.markdown(f"**Status:** :{status_colors.get(run.status, 'gray')}[{run.status}] | "
                f"**Mode:** {run.mode} | **Seed:** {run.seed}")

    tab_config, tab_training, tab_results = st.tabs(["Config", "Training", "Results"])

    # --- Config tab ---
    with tab_config:
        st.subheader("Run Configuration")
        summary = run.raw_summary
        config_keys = [
            "run_id", "run_dir", "path", "mode", "seed", "iterations",
            "warmdown_iters", "qat_bits", "qat_enabled", "qat_onset_scale",
            "late_qat_threshold",
        ]
        config_data = {k: summary.get(k, "N/A") for k in config_keys}
        if run.model_params is not None:
            config_data["model_params"] = f"{run.model_params:,}"
        if run.progress_pct is not None:
            config_data["progress_pct"] = f"{run.progress_pct:.1f}%"

        col1, col2 = st.columns(2)
        items = list(config_data.items())
        mid = len(items) // 2
        with col1:
            for k, v in items[:mid]:
                st.text(f"{k}: {v}")
        with col2:
            for k, v in items[mid:]:
                st.text(f"{k}: {v}")

    # --- Training tab ---
    with tab_training:
        st.subheader("Training Dynamics")
        if run.train_steps:
            fig = loss_curve([run], by_time=False, show_val=True)

            # Add event lines
            for event in run.events:
                fig.add_vline(
                    x=event.step, line_dash="dot", line_color="white",
                    annotation_text=event.label, annotation_position="top",
                    annotation_font_size=10,
                )

            fig.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Throughput plot
            st.subheader("Training Throughput")
            steps = run.train_steps
            fig_tp = go.Figure()
            fig_tp.add_trace(go.Scatter(
                x=[s.step for s in steps],
                y=[s.step_avg_ms for s in steps],
                mode="lines",
                line=dict(color=MODE_COLORS.get(run.mode, DEFAULT_COLOR)),
                name="ms/step",
            ))
            fig_tp.update_layout(
                xaxis_title="Step",
                yaxis_title="ms / step",
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig_tp, use_container_width=True)
        else:
            st.info("No step-by-step training data available for this run.")

    # --- Results tab ---
    with tab_results:
        st.subheader("Final Evaluation Metrics")

        # BPB metrics row
        cols = st.columns(4)
        with cols[0]:
            bpb_metric("Float fixed", run.final_float_fixed_bpb)
        with cols[1]:
            bpb_metric("Float sliding", run.final_float_sliding_bpb)
        with cols[2]:
            bpb_metric("Quant fixed", run.final_quant_fixed_bpb)
        with cols[3]:
            bpb_metric("Quant sliding", run.final_quant_sliding_bpb)

        cols2 = st.columns(4)
        with cols2[0]:
            bpb_metric("Post-EMA", run.post_ema_val_bpb)
        with cols2[1]:
            bpb_metric("Int6 roundtrip", run.final_int6_bpb)
        with cols2[2]:
            bpb_metric("Pre-EMA Int6", run.pre_ema_int6_bpb)
        with cols2[3]:
            metric_card("Quant gap", run.quant_degradation, fmt="+.4f")

        st.markdown("---")

        # Size and resource info
        st.subheader("Size & Resources")
        cols3 = st.columns(3)
        with cols3[0]:
            if run.submission_size_bytes is not None:
                size_mb = run.submission_size_bytes / 1e6
                metric_card("Submission size", size_mb, fmt=".2f")
                pct = run.size_pct
                if pct is not None and pct > 100:
                    st.error(f"Over budget by {(run.submission_size_bytes - SIZE_LIMIT_BYTES) / 1024:.1f} KB")
                elif pct is not None:
                    st.success(f"Under budget: {(SIZE_LIMIT_BYTES - run.submission_size_bytes) / 1024:.1f} KB headroom")
            else:
                st.text("Size: N/A")
        with cols3[1]:
            if run.model_params is not None:
                metric_card("Parameters", run.model_params / 1e6, fmt=".1f")
        with cols3[2]:
            if run.wallclock_seconds is not None:
                metric_card("Wallclock", run.wallclock_seconds, fmt=".0f")

        # Raw notes
        notes = run.raw_summary.get("notes", [])
        enable_event = run.raw_summary.get("enable_event")
        if notes or enable_event:
            st.markdown("---")
            st.subheader("Notes & Events")
            if enable_event:
                st.code(enable_event)
            for note in notes:
                st.code(note)
