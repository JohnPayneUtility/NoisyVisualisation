"""
Subplot-based Pareto front plots.

This module provides functions for displaying Pareto fronts in grid layouts,
with each generation shown in its own subplot.
"""

import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..base import (
    DEFAULT_TEMPLATE,
    get_series_info,
    get_run_entries,
    create_empty_figure,
    front_distance,
)


def plot_subplots(frontdata, series_labels):
    """
    Plot each generation's Pareto front in its own subplot.

    Creates a grid of subplots where each subplot shows the Pareto front
    at a particular generation.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series

    Returns:
        go.Figure: Plotly figure with subplot grid
    """
    if not frontdata:
        return create_empty_figure("No multi-objective data")

    # Pick first series and first run
    group_idx = 0
    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        return create_empty_figure(f"No runs for {series_name}")

    run_idx = 2
    gen_entries = get_run_entries(runs_full, run_idx)

    if not gen_entries:
        return create_empty_figure(f"No generations for {series_name}, Run {run_idx}")

    # Determine grid layout
    n_gens = len(gen_entries)
    cols = math.ceil(math.sqrt(n_gens))
    rows = math.ceil(n_gens / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"G{entry.get('gen_idx', i)}" for i, entry in enumerate(gen_entries)],
        shared_xaxes=True,
        shared_yaxes=True
    )

    for i, entry in enumerate(gen_entries):
        pts = entry.get('algo_front_noisy_fitnesses') or []
        if not pts:
            continue

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                showlegend=False
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title=f"True Fitness of Noisy PF - {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template=DEFAULT_TEMPLATE,
        height=300 * rows,
        margin=dict(l=40, r=20, t=80, b=40)
    )

    return fig


def plot_subplots_multi(frontdata, series_labels, nruns=1):
    """
    Plot multiple runs' Pareto fronts in subplots, with all runs overlaid per generation.

    Creates a grid where each subplot shows a generation, with multiple runs
    differentiated by color.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        nruns: Number of runs to display (default: 1)

    Returns:
        go.Figure: Plotly figure with subplot grid and multiple runs
    """
    if not frontdata:
        return create_empty_figure("No multi-objective data")

    group_idx = 0
    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        return create_empty_figure(f"No runs for {series_name}")

    # Decide which runs to plot
    max_runs_available = len(runs_full)
    num_runs = min(nruns, max_runs_available)
    run_indices = list(range(num_runs))

    # Determine number of generations (use max across runs)
    n_gens = max(len(runs_full[r]) for r in run_indices)

    if n_gens == 0:
        return create_empty_figure(f"No generations for {series_name}")

    cols = math.ceil(math.sqrt(n_gens))
    rows = math.ceil(n_gens / cols)

    # Build subplot titles
    subplot_titles = []
    for gi in range(n_gens):
        gen_label = gi
        for r in run_indices:
            if gi < len(runs_full[r]):
                gen_label = runs_full[r][gi].get("gen_idx", gi)
                break
        subplot_titles.append(f"G{gen_label}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True
    )

    colors = px.colors.qualitative.Plotly

    for gi in range(n_gens):
        row = gi // cols + 1
        col = gi % cols + 1

        for r_idx, run_id in enumerate(run_indices):
            run = runs_full[run_id]
            if gi >= len(run):
                continue

            entry = run[gi]
            pts = entry.get("algo_front_noisy_fitnesses") or []
            if not pts:
                continue

            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            color = colors[r_idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"Run {run_id}",
                    showlegend=(gi == 0),
                    line=dict(color=color),
                    marker=dict(color=color)
                ),
                row=row,
                col=col
            )

    fig.update_layout(
        title=f"True Fitness of Noisy PF - {series_name}, first {num_runs} runs",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template=DEFAULT_TEMPLATE,
        height=300 * rows,
        margin=dict(l=40, r=20, t=80, b=40),
    )

    return fig


def plot_subplots_highlighted(
    frontdata,
    series_labels,
    solutions_key="algo_front_solutions",
    fits_key="algo_front_noisy_fitnesses",
    dist_decimals=3,
):
    """
    Plot Pareto fronts in subplots with previous generation shown in grey.

    Highlights new solutions (not present in previous generation) with star markers.
    Adds front distance to each subplot title.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        solutions_key: Key for solution data in entries
        fits_key: Key for fitness data in entries
        dist_decimals: Decimal places for distance display

    Returns:
        go.Figure: Plotly figure with highlighted changes
    """
    if not frontdata:
        return create_empty_figure("No multi-objective data")

    group_idx = 0
    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        return create_empty_figure(f"No runs for {series_name}")

    run_idx = 0
    gen_entries = get_run_entries(runs_full, run_idx)

    if not gen_entries:
        return create_empty_figure(f"No generations for {series_name}, Run {run_idx}")

    # Build subplot titles including distance-to-previous
    subplot_titles = []
    for i, entry in enumerate(gen_entries):
        g = entry.get("gen_idx", i)

        if i == 0:
            subplot_titles.append(f"G{g} (d=-)")
            continue

        prev_entry = gen_entries[i - 1]
        cur_front_sols = entry.get(solutions_key) or []
        prev_front_sols = prev_entry.get(solutions_key) or []

        if not cur_front_sols or not prev_front_sols:
            d_txt = "-"
        else:
            try:
                d = front_distance(cur_front_sols, prev_front_sols)
                d_txt = f"{d:.{dist_decimals}f}"
            except Exception:
                d_txt = "-"

        subplot_titles.append(f"G{g} (d={d_txt})")

    # Determine grid layout
    n_gens = len(gen_entries)
    cols = min(5, n_gens)
    rows = math.ceil(n_gens / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for i, entry in enumerate(gen_entries):
        pts = entry.get(fits_key) or []
        if not pts:
            continue

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        row = i // cols + 1
        col = i % cols + 1

        # Previous generation points (grey)
        if i > 0:
            prev_entry = gen_entries[i - 1]
            prev_pts = prev_entry.get(fits_key) or []

            if prev_pts:
                prev_xs = [float(p[0]) for p in prev_pts]
                prev_ys = [float(p[1]) for p in prev_pts]

                fig.add_trace(
                    go.Scatter(
                        x=prev_xs,
                        y=prev_ys,
                        mode="markers",
                        marker=dict(color="lightgrey"),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            prev_set = {(float(p[0]), float(p[1])) for p in prev_pts}
        else:
            prev_set = set()

        # Current generation points with highlighting
        marker_symbols = []
        for x_val, y_val in zip(xs, ys):
            marker_symbols.append("circle" if (x_val, y_val) in prev_set else "star")

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                marker=dict(symbol=marker_symbols),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"True Fitness of Noisy PF - {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template=DEFAULT_TEMPLATE,
        height=300 * rows,
        margin=dict(l=40, r=20, t=80, b=40),
    )

    return fig
