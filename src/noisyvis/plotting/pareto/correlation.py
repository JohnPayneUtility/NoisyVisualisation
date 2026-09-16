"""
Correlation and scatter plots for Pareto front analysis.

This module provides visualizations for analyzing the relationship between
decision space movement and objective space changes.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import (
    DEFAULT_TEMPLATE,
    front_distance,
    symmetric_range,
    get_series_info,
    create_empty_figure,
)


def plot_movement_correlation(
    frontdata,
    series_labels,
    group_idx=0,
    run_idx=0,
    solution_key="algo_front_solutions",
    IndVsDist_IndType="NoisyHV",
    window=25,
    corr_method="pearson",
    show_deltas=True,
):
    """
    Plot sliding-window correlation between decision movement and objective progress.

    Shows how the correlation between decision space movement (dD) and
    objective progress (dO) changes over generations.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        group_idx: Index of the group to use
        run_idx: Index of the run to use
        solution_key: Key for solution data
        IndVsDist_IndType: 'NoisyHV' or 'CleanHV' for hypervolume source
        window: Sliding window size for correlation
        corr_method: 'pearson' or 'spearman'
        show_deltas: Whether to show delta traces

    Returns:
        go.Figure: Plotly figure with correlation over time
    """
    fig = go.Figure()

    if not frontdata:
        fig.update_layout(title="No MO_data provided")
        return fig

    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    if len(runs_full) <= run_idx:
        fig.update_layout(title=f"Run {run_idx} not available for {series_name}")
        return fig

    gen_entries = runs_full[run_idx]
    if not gen_entries or len(gen_entries) < 2:
        fig.update_layout(title=f"Not enough generations for {series_name} (run {run_idx})")
        return fig

    # Determine HV key based on type
    if IndVsDist_IndType == 'NoisyHV':
        hv_key = 'algo_front_noisy_hypervolume'
    else:
        hv_key = 'algo_front_clean_hypervolume'

    # Compute per-gen series
    gens = []
    delta_D = []
    delta_O = []

    sols = [e.get(solution_key) or [] for e in gen_entries]
    hvs = [e.get(hv_key, None) for e in gen_entries]
    gen_ids = [e.get("gen_idx", i) for i, e in enumerate(gen_entries)]

    for t in range(1, len(gen_entries)):
        prev_s, cur_s = sols[t-1], sols[t]
        dD = float(front_distance(prev_s, cur_s))

        prev_hv = hvs[t-1]
        cur_hv = hvs[t]
        if prev_hv is None or cur_hv is None:
            dO = 0.0
        else:
            dO = float(cur_hv - prev_hv)

        gens.append(int(gen_ids[t]))
        delta_D.append(dD)
        delta_O.append(dO)

    gens = np.asarray(gens)
    delta_D = np.asarray(delta_D, dtype=float)
    delta_O = np.asarray(delta_O, dtype=float)

    corr_range = [-1.1, 1.1]
    dD_range = symmetric_range(delta_D)
    dO_range = symmetric_range(delta_O)

    # Sliding correlation
    def _corr(x, y, method="spearman"):
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return np.nan

        if method == "pearson":
            return float(np.corrcoef(x, y)[0, 1])

        # spearman: rank-transform then pearson
        rx = x.argsort().argsort().astype(float)
        ry = y.argsort().argsort().astype(float)
        if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
            return np.nan
        return float(np.corrcoef(rx, ry)[0, 1])

    corr = np.full_like(delta_D, fill_value=np.nan, dtype=float)

    w = int(window)
    if w < 2:
        w = 2

    for i in range(len(delta_D)):
        if i < w - 1:
            continue
        xs = delta_D[i-w+1:i+1]
        ys = delta_O[i-w+1:i+1]
        corr[i] = _corr(xs, ys, method=corr_method)

    # Plot
    fig.add_trace(go.Scatter(
        x=gens, y=corr, mode="lines+markers",
        name=f"{corr_method.title()} corr(dD, dO), window={w}"
    ))

    if show_deltas:
        fig.add_trace(go.Scatter(
            x=gens, y=delta_D, mode="lines",
            name="dD (decision change)", yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=gens, y=delta_O, mode="lines",
            name="dO (HV change)", yaxis="y3"
        ))

    fig.update_layout(
        title=f"Movement correlation - {series_name} (run {run_idx})",
        xaxis=dict(title="Generation"),
        yaxis=dict(
            title="Correlation (-1 to 1)",
            range=corr_range,
            zeroline=True,
            zerolinewidth=2
        ),
        yaxis2=dict(
            title="dD",
            overlaying="y",
            side="right",
            range=dD_range,
            zeroline=True,
            zerolinewidth=2,
            showgrid=False
        ),
        yaxis3=dict(
            title="dO",
            overlaying="y",
            side="right",
            anchor="free",
            position=1.0,
            range=dO_range,
            zeroline=True,
            zerolinewidth=2,
            showgrid=False
        ),
        legend=dict(orientation="h"),
    )

    return fig


def plot_move_delta_histograms(
    frontdata,
    series_labels,
    group_idx=0,
    solution_key="algo_front_solutions",
    IndVsDist_IndType="NoisyHV",
    bins_decision=50,
    bins_objective=50,
    include_zero_moves=False,
):
    """
    Plot histograms of decision and objective space changes.

    Pools per-generation move deltas across all runs for one series,
    then plots histograms of dD and dO.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        group_idx: Index of the group to use
        solution_key: Key for solution data
        IndVsDist_IndType: 'NoisyHV' or 'CleanHV' for hypervolume source
        bins_decision: Number of bins for decision space histogram
        bins_objective: Number of bins for objective space histogram
        include_zero_moves: Whether to include generations with no change

    Returns:
        go.Figure: Plotly figure with side-by-side histograms
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("dD (decision-space front distance)", "dO (objective change)")
    )

    if IndVsDist_IndType == 'NoisyHV':
        hv_key = 'algo_front_noisy_hypervolume'
    else:
        hv_key = 'algo_front_clean_hypervolume'

    if not frontdata:
        fig.update_layout(title="No data provided")
        return fig

    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    # Pool deltas across runs
    all_delta_D = []
    all_delta_O = []

    for run_idx, gen_entries in enumerate(runs_full):
        if not gen_entries or len(gen_entries) < 2:
            continue

        sols = [e.get(solution_key) or [] for e in gen_entries]
        hvs = [e.get(hv_key, None) for e in gen_entries]

        for t in range(1, len(gen_entries)):
            prev_s, cur_s = sols[t-1], sols[t]
            dD = float(front_distance(prev_s, cur_s))

            prev_hv = hvs[t-1]
            cur_hv = hvs[t]
            if prev_hv is None or cur_hv is None:
                dO = 0.0
            else:
                dO = float(cur_hv - prev_hv)

            if include_zero_moves or (dD != 0.0 or dO != 0.0):
                all_delta_D.append(dD)
                all_delta_O.append(dO)

    all_delta_D = np.asarray(all_delta_D, dtype=float)
    all_delta_O = np.asarray(all_delta_O, dtype=float)

    # Plot histograms
    fig.add_trace(
        go.Histogram(x=all_delta_D, nbinsx=bins_decision, name="dD"),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=all_delta_O, nbinsx=bins_objective, name="dO"),
        row=1, col=2
    )

    fig.update_layout(
        title=f"Move delta histograms (pooled across runs) - {series_name}",
        bargap=0.05,
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="dD", row=1, col=1)
    fig.update_xaxes(title_text="dO", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    return fig


def plot_objective_vs_decision(
    frontdata,
    series_labels,
    group_idx=0,
    run_idx=None,
    solution_key="algo_front_solutions",
    IndVsDist_IndType="NoisyHV",
    include_zero_moves=True,
    color_by="gen",
    marker_size=6,
):
    """
    Scatter plot of objective change vs decision space movement.

    Shows the relationship between how much the front moved in decision space
    and how much the objective values changed.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        group_idx: Index of the group to use
        run_idx: Index of the run to use (None = pool all runs)
        solution_key: Key for solution data
        IndVsDist_IndType: 'NoisyHV' or 'CleanHV' for hypervolume source
        include_zero_moves: Whether to include generations with no change
        color_by: 'gen' (generation) or 'run' (run index) or None
        marker_size: Size of scatter markers

    Returns:
        go.Figure: Plotly scatter figure
    """
    fig = go.Figure()

    if IndVsDist_IndType == 'NoisyHV':
        hv_key = 'algo_front_noisy_hypervolume'
    else:
        hv_key = 'algo_front_clean_hypervolume'

    if not frontdata:
        fig.update_layout(title="No data provided")
        return fig

    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    # Decide which runs to use
    if run_idx is None:
        run_indices = list(range(len(runs_full)))
    else:
        if run_idx >= len(runs_full):
            fig.update_layout(title=f"Run {run_idx} not available for {series_name}")
            return fig
        run_indices = [run_idx]

    all_dD, all_dO, all_gen, all_run = [], [], [], []

    for r in run_indices:
        gen_entries = runs_full[r]
        if not gen_entries or len(gen_entries) < 2:
            continue

        sols = [e.get(solution_key) or [] for e in gen_entries]
        hvs = [e.get(hv_key, None) for e in gen_entries]
        gen_ids = [e.get("gen_idx", i) for i, e in enumerate(gen_entries)]

        for t in range(1, len(gen_entries)):
            prev_s, cur_s = sols[t-1], sols[t]
            dD = float(front_distance(prev_s, cur_s))

            prev_hv = hvs[t-1]
            cur_hv = hvs[t]
            dO = 0.0 if (prev_hv is None or cur_hv is None) else float(cur_hv - prev_hv)

            if include_zero_moves or (dD != 0.0 or dO != 0.0):
                all_dD.append(dD)
                all_dO.append(dO)
                all_gen.append(int(gen_ids[t]))
                all_run.append(int(r))

    dD = np.asarray(all_dD, dtype=float)
    dO = np.asarray(all_dO, dtype=float)
    gen = np.asarray(all_gen, dtype=int) if all_gen else np.array([], dtype=int)
    run = np.asarray(all_run, dtype=int) if all_run else np.array([], dtype=int)

    if len(dD) == 0:
        fig.update_layout(title=f"No moves to plot - {series_name}")
        return fig

    # Choose coloring
    if color_by is None:
        marker = dict(size=marker_size, opacity=0.75)
        fig.add_trace(go.Scatter(
            x=dD, y=dO, mode="markers",
            marker=marker,
            name="Moves"
        ))
    else:
        if (run_idx is None and color_by == "gen"):
            color_by = "run"

        c = gen if color_by == "gen" else run
        color_title = "Generation" if color_by == "gen" else "Run"

        fig.add_trace(go.Scatter(
            x=dD, y=dO, mode="markers",
            marker=dict(
                size=marker_size,
                opacity=0.75,
                color=c,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_title)
            ),
            name="Moves"
        ))

    # Add reference lines
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=0, line_dash="dash")

    title_suffix = "" if run_idx is None else f" (run {run_idx})"
    fig.update_layout(
        title=f"Objective vs decision movement - {series_name}{title_suffix}",
        xaxis_title="dD (decision-space front distance)",
        yaxis_title=f"dO (objective change: {hv_key})",
        legend=dict(orientation="h"),
    )

    return fig
