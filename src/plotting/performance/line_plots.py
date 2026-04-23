"""
Line plots for algorithm performance comparison.

This module provides line plots with error bars for comparing algorithm
performance across different noise levels.
"""

import plotly.graph_objects as go
import plotly.express as px

from ..base import DEFAULT_TEMPLATE, create_empty_figure


def _viridis_colors(n, colorscale='Viridis'):
    if n == 1:
        return [px.colors.sample_colorscale(colorscale, [0.5])[0]]
    positions = [i / (n - 1) for i in range(n)]
    return px.colors.sample_colorscale(colorscale, positions)


def plot_line(dataframe, fitness_mode='best', problem_goal='maximise', xaxis_title=None, colorscale='Viridis'):
    """
    Create a line plot comparing algorithm performance across noise levels.

    Shows mean performance with standard deviation error bars for each algorithm.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - max_fit: Best fitness value recorded across the run (maximisation)
            - min_fit: Best fitness value recorded across the run (minimisation)
            - final_fit: Final fitness value at end of run
        fitness_mode: 'best' to plot best fit, 'final' to plot final_fit
        problem_goal: 'maximise' or 'minimise' — determines which column is 'best'

    Returns:
        go.Figure: Line plot with error bars
    """
    df = dataframe.copy()

    if fitness_mode == 'final':
        fit_col = 'final_fit'
        yaxis_label = 'Final solution found'
    elif problem_goal == 'minimise':
        fit_col = 'min_fit'
        yaxis_label = 'Best solution found'
    else:
        fit_col = 'max_fit'
        yaxis_label = 'Best solution found'

    df = df[['algo_name', 'noise', fit_col]]
    stats = df.groupby(['algo_name', 'noise'])[fit_col].agg(['mean', 'std']).reset_index()

    algos = list(stats['algo_name'].unique())
    colors = _viridis_colors(len(algos), colorscale)

    fig = go.Figure()
    for algo, color in zip(algos, colors):
        subset = stats[stats['algo_name'] == algo]
        fig.add_trace(go.Scatter(
            x=subset['noise'],
            y=subset['mean'],
            error_y=dict(
                type='data',
                array=subset['std'],
                visible=True,
                thickness=1.5,
                width=5
            ),
            mode='lines+markers',
            name=algo,
            line=dict(color=color),
            marker=dict(color=color)
        ))

    fig.update_layout(
        title='title',
        xaxis_title=xaxis_title or r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)',
        yaxis_title=yaxis_label,
        legend_title='Algo Name',
        template=DEFAULT_TEMPLATE
    )

    return fig


def plot_line_evals(dataframe, fitness_mode='final', show_std=True, xaxis_title=None, colorscale='Viridis'):
    """
    Create a line plot comparing algorithm runtime (evaluations) across noise levels.

    Shows mean evaluations, optionally with standard deviation error bars, for each algorithm.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - n_evals: Total fitness evaluations used by the algorithm
            - evals_to_best: Evaluations consumed until best fitness was found (optional)
        fitness_mode: 'best' to plot evals_to_best, 'final' to plot n_evals
        show_std: Whether to show symmetric std error bars (default True)

    Returns:
        go.Figure: Line plot with optional error bars
    """
    df = dataframe.copy()

    if fitness_mode == 'best' and 'evals_to_best' in df.columns:
        eval_col = 'evals_to_best'
        yaxis_label = 'Evaluations to best found fitness'
    else:
        eval_col = 'n_evals'
        yaxis_label = 'Evaluations to final fitness'

    df = df[['algo_name', 'noise', eval_col]].dropna(subset=[eval_col])

    stats = df.groupby(['algo_name', 'noise'])[eval_col].agg(['mean', 'std']).reset_index()

    algos = list(stats['algo_name'].unique())
    colors = _viridis_colors(len(algos), colorscale)

    fig = go.Figure()
    for algo, color in zip(algos, colors):
        subset = stats[stats['algo_name'] == algo]
        error_y = dict(
            type='data',
            array=subset['std'],
            visible=True,
            thickness=1.5,
            width=5
        ) if show_std else None
        fig.add_trace(go.Scatter(
            x=subset['noise'],
            y=subset['mean'],
            error_y=error_y,
            mode='lines+markers',
            name=algo,
            line=dict(color=color),
            marker=dict(color=color)
        ))

    fig.update_layout(
        title='title',
        xaxis_title=xaxis_title or r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)',
        yaxis_title=yaxis_label,
        legend_title='Algo Name',
        template=DEFAULT_TEMPLATE
    )

    return fig


def plot_line_mo(dataframe, colorscale='Viridis'):
    """
    Create a line plot for multi-objective performance using hypervolume.

    Shows mean hypervolume with standard deviation error bars for each algorithm.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - final_true_hv: Final true hypervolume value

    Returns:
        go.Figure: Line plot with error bars
    """
    df = dataframe.copy()

    # Check if we have MO data
    if 'final_true_hv' not in df.columns:
        return create_empty_figure('No multi-objective data available')

    # Remove rows with None values
    df = df.dropna(subset=['final_true_hv'])

    if df.empty:
        return create_empty_figure('No multi-objective data available')

    df = df[['algo_name', 'noise', 'final_true_hv']]

    # Calculate statistics
    stats = df.groupby(['algo_name', 'noise'])['final_true_hv'].agg(['mean', 'std']).reset_index()

    algos = list(stats['algo_name'].unique())
    colors = _viridis_colors(len(algos), colorscale)

    fig = go.Figure()
    for algo, color in zip(algos, colors):
        subset = stats[stats['algo_name'] == algo]
        fig.add_trace(go.Scatter(
            x=subset['noise'],
            y=subset['mean'],
            error_y=dict(
                type='data',
                array=subset['std'],
                visible=True,
                thickness=1.5,
                width=5
            ),
            mode='lines+markers',
            name=algo,
            line=dict(color=color),
            marker=dict(color=color)
        ))

    fig.update_layout(
        title='Multi-Objective Performance (Hypervolume)',
        xaxis_title=r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)',
        yaxis_title='Final Hypervolume',
        legend_title='Algo Name',
        template=DEFAULT_TEMPLATE
    )

    return fig
