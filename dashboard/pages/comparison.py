"""Run comparison page: side-by-side detailed comparison."""
from __future__ import annotations

import streamlit as st

from dashboard.components.charts import loss_curve, radar_chart
from dashboard.data_model import RunData


def _metrics_table(run: RunData) -> dict:
    """Extract key metrics as a display dict."""
    return {
        "Mode": run.mode,
        "Seed": run.seed,
        "Iterations": run.iterations,
        "Model params": f"{run.model_params / 1e6:.1f}M" if run.model_params is not None else "N/A",
        "Float fixed BPB": f"{run.final_float_fixed_bpb:.4f}" if run.final_float_fixed_bpb is not None else "N/A",
        "Float sliding BPB": f"{run.final_float_sliding_bpb:.4f}" if run.final_float_sliding_bpb is not None else "N/A",
        "Quant fixed BPB": f"{run.final_quant_fixed_bpb:.4f}" if run.final_quant_fixed_bpb is not None else "N/A",
        "Quant sliding BPB": f"{run.final_quant_sliding_bpb:.4f}" if run.final_quant_sliding_bpb is not None else "N/A",
        "Quant degradation": f"{run.quant_degradation:+.4f}" if run.quant_degradation is not None else "N/A",
        "Int6 BPB": f"{run.final_int6_bpb:.4f}" if run.final_int6_bpb is not None else "N/A",
        "Post-EMA BPB": f"{run.post_ema_val_bpb:.4f}" if run.post_ema_val_bpb is not None else "N/A",
        "Size": f"{run.submission_size_bytes / 1e6:.2f} MB" if run.submission_size_bytes is not None else "N/A",
        "Size budget": f"{run.size_pct:.1f}%" if run.size_pct is not None else "N/A",
        "Wallclock": f"{run.wallclock_seconds:.0f}s" if run.wallclock_seconds is not None else "N/A",
        "Status": run.status,
    }


def render(runs: list[RunData]):
    st.header("Run Comparison")

    if not runs:
        st.info("No runs available.")
        return

    names = [r.display_name for r in runs]

    # Select 2-3 runs
    col1, col2, col3 = st.columns(3)
    sel1 = col1.selectbox("Run A", names, index=0)
    sel2 = col2.selectbox("Run B", names, index=min(1, len(names) - 1))
    sel3 = col3.selectbox("Run C (optional)", ["(none)"] + names, index=0)

    chosen_names: list[str] = []
    for name in (sel1, sel2):
        if name not in chosen_names:
            chosen_names.append(name)
    if sel3 != "(none)" and sel3 not in chosen_names:
        chosen_names.append(sel3)
    name_to_run = {r.display_name: r for r in runs}
    selected = [name_to_run[n] for n in chosen_names if n in name_to_run]

    if len(selected) < 2:
        st.warning("Select at least two *different* runs to compare.")
        return

    # Side-by-side metrics
    st.subheader("Metrics Comparison")
    cols = st.columns(len(selected))
    tables = [_metrics_table(r) for r in selected]
    for i, (col, run, table) in enumerate(zip(cols, selected, tables)):
        with col:
            st.markdown(f"**{run.display_name}**")
            for k, v in table.items():
                st.text(f"{k}: {v}")

    st.markdown("---")

    # Diff highlights
    st.subheader("Key Differences")
    numeric_keys = [
        ("final_float_sliding_bpb", "Float sliding BPB"),
        ("final_quant_sliding_bpb", "Quant sliding BPB"),
        ("submission_size_bytes", "Size (bytes)"),
    ]
    diff_rows = []
    for attr, label in numeric_keys:
        vals = [getattr(r, attr) for r in selected]
        if all(v is not None for v in vals):
            diff_rows.append({
                "Metric": label,
                **{r.display_name: f"{v:.4f}" if isinstance(v, float) else str(v) for r, v in zip(selected, vals)},
                "Delta": f"{max(vals) - min(vals):.4f}" if isinstance(vals[0], float) else str(max(vals) - min(vals)),
            })
    if diff_rows:
        st.dataframe(diff_rows, use_container_width=True)

    st.markdown("---")

    # Mini loss curves
    st.subheader("Training Curves")
    curves_runs = [r for r in selected if r.train_steps]
    if curves_runs:
        fig = loss_curve(curves_runs, by_time=False, show_val=True)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No step-by-step data available for selected runs.")

    st.markdown("---")

    # Radar chart
    st.subheader("Multi-Dimension Comparison")
    radar_runs = [
        r for r in selected
        if r.final_quant_sliding_bpb is not None
        and r.submission_size_bytes is not None
        and r.quant_degradation is not None
        and r.early_val_bpb is not None
    ]
    if len(radar_runs) >= 2:
        st.caption("Scores are normalized within the selected runs. Higher is better on every axis.")
        fig = radar_chart(radar_runs)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Radar comparison needs at least two runs with size, float/quant, and validation metrics.")
