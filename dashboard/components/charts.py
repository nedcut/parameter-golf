"""Plotly chart factory functions."""
from __future__ import annotations

import plotly.graph_objects as go

from ..config import DEFAULT_COLOR, MODE_COLORS, SIZE_LIMIT_BYTES, SOTA_BPB
from ..data_model import RunData


def _color_for(mode: str) -> str:
    normalized = mode.split("@", 1)[0]
    return MODE_COLORS.get(normalized, DEFAULT_COLOR)


def _normalize_values(values: list[float | None], *, lower_is_better: bool) -> list[float]:
    """Normalize a metric onto a 0-100 scale for cross-run comparison charts."""
    present = [value for value in values if value is not None]
    if not present:
        return [0.0] * len(values)
    low, high = min(present), max(present)
    if low == high:
        return [100.0 if value is not None else 0.0 for value in values]

    normalized: list[float] = []
    for value in values:
        if value is None:
            normalized.append(0.0)
            continue
        if lower_is_better:
            score = (high - value) / (high - low)
        else:
            score = (value - low) / (high - low)
        normalized.append(max(0.0, min(100.0, score * 100)))
    return normalized


def loss_curve(runs: list[RunData], by_time: bool = False, show_val: bool = True, log_y: bool = False) -> go.Figure:
    """Multi-run training loss overlay."""
    fig = go.Figure()

    for run in runs:
        if not run.train_steps:
            continue
        name = run.display_name
        color = _color_for(run.mode)

        x = [s.train_time_ms / 1000 if by_time else s.step for s in run.train_steps]
        y = [s.train_loss for s in run.train_steps]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"{name}<br>Step: %{{customdata}}<br>Loss: %{{y:.4f}}<extra></extra>",
            customdata=[s.step for s in run.train_steps],
        ))

        if show_val and run.val_checkpoints:
            vx = [v.train_time_ms / 1000 if by_time else v.step for v in run.val_checkpoints]
            vy = [v.val_bpb for v in run.val_checkpoints]
            fig.add_trace(go.Scatter(
                x=vx, y=vy, mode="markers", name=f"{name} (val_bpb)",
                marker=dict(color=color, size=8, symbol="diamond"),
                hovertemplate=f"{name}<br>Step: %{{customdata}}<br>Val BPB: %{{y:.4f}}<extra></extra>",
                customdata=[v.step for v in run.val_checkpoints],
            ))

    # SOTA reference
    fig.add_hline(y=SOTA_BPB, line_dash="dash", line_color="gold",
                  annotation_text=f"SOTA {SOTA_BPB}", annotation_position="top right")

    x_title = "Wallclock (seconds)" if by_time else "Step"
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Loss / BPB",
        yaxis_type="log" if log_y else "linear",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def quant_gap_bar(runs: list[RunData]) -> go.Figure:
    """Grouped bar: float_sliding vs quant_sliding BPB per run."""
    eligible = [
        r for r in runs
        if r.final_float_sliding_bpb is not None and r.final_quant_sliding_bpb is not None
    ]
    eligible.sort(key=lambda r: (r.quant_degradation or 0), reverse=True)

    names = [r.display_name for r in eligible]
    float_vals = [r.final_float_sliding_bpb for r in eligible]
    quant_vals = [r.final_quant_sliding_bpb for r in eligible]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Float sliding", x=names, y=float_vals, marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="Quant sliding", x=names, y=quant_vals, marker_color="#EF553B"))

    fig.update_layout(
        barmode="group",
        yaxis_title="BPB",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=20, t=30, b=80),
    )
    return fig


def size_scatter(runs: list[RunData]) -> go.Figure:
    """Scatter: submission size vs BPB with budget line."""
    eligible = [
        r for r in runs
        if r.submission_size_bytes is not None and r.final_quant_sliding_bpb is not None
    ]

    fig = go.Figure()
    for run in eligible:
        fig.add_trace(go.Scatter(
            x=[run.submission_size_bytes / 1e6],
            y=[run.final_quant_sliding_bpb],
            mode="markers+text",
            text=[run.display_name],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(color=_color_for(run.mode), size=12),
            name=run.display_name,
            showlegend=False,
            hovertemplate=(
                f"{run.display_name}<br>"
                f"Size: %{{x:.2f}} MB<br>"
                f"BPB: %{{y:.4f}}<extra></extra>"
            ),
        ))

    # 16MB budget line
    fig.add_vline(x=SIZE_LIMIT_BYTES / 1e6, line_dash="dash", line_color="red",
                  annotation_text="16 MB limit", annotation_position="top left")
    # SOTA line
    fig.add_hline(y=SOTA_BPB, line_dash="dash", line_color="gold",
                  annotation_text=f"SOTA {SOTA_BPB}", annotation_position="top right")

    fig.update_layout(
        xaxis_title="Submission size (MB)",
        yaxis_title="Quant sliding BPB",
        template="plotly_dark",
        height=500,
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def radar_chart(runs: list[RunData]) -> go.Figure:
    """Radar chart comparing runs across normalized dimensions."""
    if not runs:
        return go.Figure()

    categories = ["Quant BPB", "Size headroom", "Quant robustness", "Early convergence"]
    bpb_scores = _normalize_values(
        [r.final_quant_sliding_bpb for r in runs],
        lower_is_better=True,
    )
    headroom_scores = _normalize_values(
        [
            max(0.0, 100.0 - (r.size_pct or 100.0))
            if r.submission_size_bytes is not None else None
            for r in runs
        ],
        lower_is_better=False,
    )
    robustness_scores = _normalize_values(
        [r.quant_degradation for r in runs],
        lower_is_better=True,
    )
    convergence_scores = _normalize_values(
        [r.early_val_bpb for r in runs],
        lower_is_better=True,
    )

    fig = go.Figure()
    for run, bpb, headroom, robustness, convergence in zip(
        runs,
        bpb_scores,
        headroom_scores,
        robustness_scores,
        convergence_scores,
    ):
        values = [bpb, headroom, robustness, convergence]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            name=run.display_name,
            line=dict(color=_color_for(run.mode)),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


def mode_boxplot(runs: list[RunData], metric: str = "final_quant_sliding_bpb") -> go.Figure:
    """Box plot of a metric grouped by QAT mode."""
    fig = go.Figure()

    modes = sorted({r.mode for r in runs})
    for mode in modes:
        values = [getattr(r, metric) for r in runs if r.mode == mode and getattr(r, metric) is not None]
        if values:
            fig.add_trace(go.Box(
                y=values, name=mode,
                marker_color=_color_for(mode),
                boxpoints="all", jitter=0.3,
            ))

    fig.add_hline(y=SOTA_BPB, line_dash="dash", line_color="gold",
                  annotation_text=f"SOTA {SOTA_BPB}")

    fig.update_layout(
        yaxis_title="BPB",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig
