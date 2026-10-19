"""
Basic Pareto front plot.

This module provides the basic single Pareto front visualization.
"""

import plotly.graph_objects as go

from ..base import (
    DEFAULT_TEMPLATE,
    DEFAULT_COLORSCALE,
    get_series_info,
    get_run_entries,
    compute_generation_range,
    create_empty_figure,
    FRONT_KEYS,
    UNSUPPORTED_METRIC_TITLE,
    front_line,
    selected_front,
)


def plot_basic(frontdata, series_labels, continuous_colorscale=True,
               colorscale='Viridis', IndVsDist_IndType="noisy_population", attainment=False,
               gridlines=None):
    """
    Generate a basic Pareto front plot showing all generations.

    Creates a scatter plot with all Pareto front generations overlaid,
    with colors indicating generation number.

    Args:
        frontdata: Data for the Pareto fronts (list of groups, each containing runs)
        series_labels: Labels for each series
        continuous_colorscale: If True, use continuous color scale instead of discrete colors
        colorscale: The colorscale to use when continuous_colorscale is True
                   (e.g., 'Viridis', 'Plasma', 'Cividis')
        IndVsDist_IndType: the front type (a FRONT_KEYS key): its members, objective values and stored HV
        attainment: If True, join each front by its attainment staircase instead of straight segments
        gridlines: None, or an (x, y) pair: a grey vertical line at x and a grey horizontal line at y
                   (either may be None to omit that line), e.g. the bounds of the valid objective space

    Returns:
        go.Figure: Plotly figure with the Pareto front visualization
    """
    fig = go.Figure()

    front = selected_front(IndVsDist_IndType)
    if front is None:
        return create_empty_figure(UNSUPPORTED_METRIC_TITLE)
    _, fits_key, _, front_label = FRONT_KEYS[front]

    if not frontdata:
        fig.update_layout(title="No multi-objective data")
        return fig

    # Pick first series and first run for a simple baseline
    group_idx = 0
    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    run_idx = 0
    gen_entries = get_run_entries(runs_full, run_idx)

    if not gen_entries:
        fig.update_layout(title=f"No generations for {series_name}, Run {run_idx}")
        return fig

    if continuous_colorscale:
        min_gen, max_gen = compute_generation_range(
            [e for e in gen_entries if e.get(fits_key)]
        )

    first_trace = True
    for entry in gen_entries:
        pts = entry.get(fits_key) or []
        g = entry.get('gen_idx', None)
        if not pts:
            continue

        xs, ys, line_shape = front_line(pts, entry.get('opt_weights'), attainment)

        if continuous_colorscale:
            # Use continuous color scale with colorbar (only show colorbar for first trace)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=f"G{g}",
                    marker=dict(
                        color=[g] * len(xs),
                        colorscale=colorscale,
                        cmin=min_gen,
                        cmax=max_gen,
                        colorbar=dict(
                            title="Generation",
                            thickness=15,
                            len=0.7
                        ) if first_trace else None,
                        showscale=first_trace
                    ),
                    line=dict(
                        color=f'rgba({int(255 * (g - min_gen) / max(1, max_gen - min_gen))}, '
                              f'{int(100 + 155 * (g - min_gen) / max(1, max_gen - min_gen))}, '
                              f'{int(255 - 155 * (g - min_gen) / max(1, max_gen - min_gen))}, 0.5)'
                    ),
                    showlegend=False,
                    line_shape=line_shape
                )
            )
            first_trace = False
        else:
            # Use discrete colors (original behavior)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name=f"G{g}",
                    line_shape=line_shape
                )
            )

    if gridlines is not None:
        gx, gy = gridlines
        if gx is not None:
            fig.add_vline(x=gx, line=dict(color='grey', width=1))
        if gy is not None:
            fig.add_hline(y=gy, line=dict(color='grey', width=1))

    fig.update_layout(
        title=f"{front_label} PF - {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template=DEFAULT_TEMPLATE,
        legend_title="Generation" if not continuous_colorscale else None
    )
    return fig
