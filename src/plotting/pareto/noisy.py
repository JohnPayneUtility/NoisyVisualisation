"""
Noisy vs clean fitness comparison plot.

This module provides visualization comparing noisy and clean fitness values
on Pareto fronts.
"""

import plotly.graph_objects as go

from ..base import (
    DEFAULT_TEMPLATE,
    PARETO_COLORSCALE,
    generation_color,
    get_series_info,
    get_run_entries,
    compute_generation_range,
    create_empty_figure,
)


def plot_noisy(frontdata, series_labels):
    """
    Plot true (clean) fitness vs noisy fitness for Pareto front solutions.

    Creates a scatter plot showing:
    - Clean fitness values as circles
    - Noisy fitness values as triangles
    - Dotted lines connecting corresponding clean/noisy pairs
    - Colors indicating generation number

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series

    Returns:
        go.Figure: Plotly figure comparing noisy and clean fitness
    """
    fig = go.Figure()

    if not frontdata:
        fig.update_layout(title="No multi-objective data")
        return fig

    group_idx = 0
    run_idx = 0
    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    gen_entries = get_run_entries(runs_full, run_idx)

    if not gen_entries:
        fig.update_layout(title=f"No generations for {series_name}")
        return fig

    gmin, gmax = compute_generation_range(gen_entries)

    clean_x, clean_y, clean_c = [], [], []
    noisy_x, noisy_y, noisy_c = [], [], []

    for entry in gen_entries:
        g = entry.get("gen_idx", 0)
        clean_pts = entry.get("algo_front_clean_fitnesses") or []
        noisy_pts = entry.get("algo_front_noisy_fitnesses") or []
        col = generation_color(g, gmin, gmax)

        for c_fit, n_fit in zip(clean_pts, noisy_pts):
            # Draw dotted connector
            fig.add_trace(go.Scatter(
                x=[float(c_fit[0]), float(n_fit[0])],
                y=[float(c_fit[1]), float(n_fit[1])],
                mode="lines",
                line=dict(dash="dot", width=0.4, color=col),
                hoverinfo="skip",
                showlegend=False
            ))
            clean_x.append(float(c_fit[0]))
            clean_y.append(float(c_fit[1]))
            clean_c.append(g)
            noisy_x.append(float(n_fit[0]))
            noisy_y.append(float(n_fit[1]))
            noisy_c.append(g)

    # Clean points (circles) with colorbar
    fig.add_trace(go.Scatter(
        x=clean_x, y=clean_y, mode="markers", name="Clean fitness",
        marker=dict(
            symbol="circle", size=8, color=clean_c,
            colorscale=PARETO_COLORSCALE, colorbar=dict(title="Generation"),
            cmin=gmin, cmax=gmax
        )
    ))

    # Noisy points (triangles) same scale
    fig.add_trace(go.Scatter(
        x=noisy_x, y=noisy_y, mode="markers", name="Noisy fitness",
        marker=dict(
            symbol="triangle-up", size=4, color=noisy_c,
            colorscale=PARETO_COLORSCALE, showscale=False,
            cmin=gmin, cmax=gmax
        )
    ))

    fig.update_layout(
        title=f"True vs Noisy Fitness - {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template=DEFAULT_TEMPLATE,
        legend_title="Type"
    )

    return fig
