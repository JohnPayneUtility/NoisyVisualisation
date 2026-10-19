"""
Two-front comparison plot: a noisy front against a clean or corrected front.

This module draws, generation by generation, a noisy front's observed objective values next to a second
front's true objective values (the Pareto tab's three "Noisy (A vs B)" plot types).
"""

import plotly.graph_objects as go

from ....common import sol_tuple_ints
from ..base import (
    DEFAULT_TEMPLATE,
    PARETO_COLORSCALE,
    generation_color,
    get_series_info,
    get_run_entries,
    compute_generation_range,
)


def plot_noisy(frontdata, series_labels, labels=("noisy population", "clean population")):
    """
    Plot a noisy front (A) against a clean or corrected front (B) for one run, at every generation where
    either front changes.

    Each entry holds a_solutions/a_fitnesses (A's members under their observed y) and b_solutions/b_fitnesses
    (B's members under f(x)). The figure shows:
    - A's points as triangles and B's as circles, coloured by generation on one shared scale
    - a dotted line from each A point to the B point of the same genotype, when that genotype is on B

    Args:
        frontdata: Data for the front pairs (list of groups, each containing runs)
        series_labels: Labels for each series
        labels: (A's label, B's label)

    Returns:
        go.Figure: Plotly figure comparing the two fronts
    """
    fig = go.Figure()
    label_a, label_b = labels

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

    a_x, a_y, a_c = [], [], []
    b_x, b_y, b_c = [], [], []

    for entry in gen_entries:
        g = entry.get("gen_idx", 0)
        col = generation_color(g, gmin, gmax)
        on_b = {sol_tuple_ints(s): p for s, p in zip(entry.get("b_solutions") or [], entry.get("b_fitnesses") or [])}

        for sol, a_fit in zip(entry.get("a_solutions") or [], entry.get("a_fitnesses") or []):
            b_fit = on_b.get(sol_tuple_ints(sol))
            if b_fit is not None:
                # Draw dotted connector to the same genotype on front B
                fig.add_trace(go.Scatter(
                    x=[float(a_fit[0]), float(b_fit[0])],
                    y=[float(a_fit[1]), float(b_fit[1])],
                    mode="lines",
                    line=dict(dash="dot", width=0.4, color=col),
                    hoverinfo="skip",
                    showlegend=False
                ))
            a_x.append(float(a_fit[0]))
            a_y.append(float(a_fit[1]))
            a_c.append(g)

        for b_fit in entry.get("b_fitnesses") or []:
            b_x.append(float(b_fit[0]))
            b_y.append(float(b_fit[1]))
            b_c.append(g)

    # Front B (circles) with colorbar
    fig.add_trace(go.Scatter(
        x=b_x, y=b_y, mode="markers", name=label_b,
        marker=dict(
            symbol="circle", size=8, color=b_c,
            colorscale=PARETO_COLORSCALE, colorbar=dict(title="Generation"),
            cmin=gmin, cmax=gmax
        )
    ))

    # Front A (triangles) same scale
    fig.add_trace(go.Scatter(
        x=a_x, y=a_y, mode="markers", name=label_a,
        marker=dict(
            symbol="triangle-up", size=4, color=a_c,
            colorscale=PARETO_COLORSCALE, showscale=False,
            cmin=gmin, cmax=gmax
        )
    ))

    fig.update_layout(
        title=f"{label_a} vs {label_b} - {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template=DEFAULT_TEMPLATE,
        legend_title="Front"
    )

    return fig
