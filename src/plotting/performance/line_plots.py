"""
Line plots for algorithm performance comparison.

This module provides line plots with error bars for comparing algorithm
performance across different noise levels.
"""

import plotly.graph_objects as go

from ..base import DEFAULT_TEMPLATE, create_empty_figure


def plot_line(dataframe, value='final'):
    """
    Create a line plot comparing algorithm performance across noise levels.

    Shows mean performance with standard deviation error bars for each algorithm.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - final_fit: Final fitness value
        value: Which value to plot (currently only 'final' supported)

    Returns:
        go.Figure: Line plot with error bars
    """
    df = dataframe.copy()
    df = df[['algo_name', 'noise', 'final_fit']]

    if value == 'final':
        stats = df.groupby(['algo_name', 'noise'])['final_fit'].agg(['mean', 'std']).reset_index()

    fig = go.Figure()
    for algo in stats['algo_name'].unique():
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
            name=algo
        ))

    fig.update_layout(
        title='title',
        xaxis_title=r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)',
        yaxis_title='Best solution found',
        legend_title='Algo Name',
        template=DEFAULT_TEMPLATE
    )

    return fig


def plot_line_mo(dataframe):
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

    fig = go.Figure()
    for algo in stats['algo_name'].unique():
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
            name=algo
        ))

    fig.update_layout(
        title='Multi-Objective Performance (Hypervolume)',
        xaxis_title=r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)',
        yaxis_title='Final Hypervolume',
        legend_title='Algo Name',
        template=DEFAULT_TEMPLATE
    )

    return fig
