"""
Animated Pareto front plot.

This module provides animated visualization of Pareto front evolution
across generations.
"""

import plotly.graph_objects as go

from ..base import (
    DEFAULT_TEMPLATE,
    get_series_info,
    get_run_entries,
    create_empty_figure,
)


def plot_animation(frontdata, series_labels):
    """
    Create an animated plot showing Pareto front evolution.

    Each frame shows one generation's Pareto front with play/pause controls
    and a slider for generation selection.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series

    Returns:
        go.Figure: Animated Plotly figure
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

    # Collect data and axis ranges across all generations
    all_x, all_y = [], []
    gens_data = []  # list of (gen_label, xs, ys)

    for i, entry in enumerate(gen_entries):
        pts = entry.get('algo_front_noisy_fitnesses') or []
        g = entry.get('gen_idx', i)

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        gens_data.append((g, xs, ys))
        all_x.extend(xs)
        all_y.extend(ys)

    if not all_x or not all_y:
        return create_empty_figure(f"No points to plot for {series_name}, Run {run_idx}")

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Initial frame: first generation with points
    init_idx = next((idx for idx, (_, xs, ys) in enumerate(gens_data) if xs and ys), 0)
    init_gen, init_xs, init_ys = gens_data[init_idx]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=init_xs,
                y=init_ys,
                mode="lines+markers",
                name=f"G{init_gen}",
                showlegend=False,
            )
        ]
    )

    # Build frames
    frames = []
    for g, xs, ys in gens_data:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        showlegend=False,
                    )
                ],
                name=str(g),
                layout=go.Layout(
                    title_text=f"True Fitness of Noisy PF - {series_name}, Run {run_idx}, Gen {g}"
                )
            )
        )

    fig.frames = frames

    # Animation controls (Play/Pause + slider)
    steps = []
    for g, _, _ in gens_data:
        steps.append(
            dict(
                method="animate",
                label=f"G{g}",
                args=[
                    [str(g)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=300, redraw=True),
                        transition=dict(duration=0)
                    ),
                ],
            )
        )

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Generation: "},
            pad={"t": 30},
            steps=steps,
        )
    ]

    fig.update_layout(
        title=f"True Fitness of Noisy PF - {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template=DEFAULT_TEMPLATE,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.15,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        sliders=sliders,
        height=600,
    )

    return fig
