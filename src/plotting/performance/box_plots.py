"""
Box plots for algorithm performance comparison.

This module provides box plots for comparing algorithm performance
distributions across different noise levels.
"""

import plotly.graph_objects as go
import plotly.express as px

from ..base import DEFAULT_TEMPLATE, create_empty_figure


def plot_box(dataframe, value='final'):
    """
    Create a box plot comparing algorithm performance across noise levels.

    Shows the distribution of performance values for each algorithm at each
    noise level.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - final_fit: Final fitness value
            - max_fit: Maximum fitness value
        value: Which value to plot (currently only 'final' supported)

    Returns:
        go.Figure: Box plot comparing algorithms
    """
    df = dataframe.copy()
    df = df[['algo_name', 'noise', 'final_fit', 'max_fit']]

    noise_levels = sorted(df['noise'].unique())

    fig = px.box(
        df,
        x="noise",
        y="max_fit",
        color="algo_name",
        category_orders={"noise": noise_levels},
        points=False
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="d, where d x mean(W) is s.d. of noise",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        yaxis=dict(
            title=dict(
                text="Best solution found",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        legend=dict(
            title=dict(font=dict(size=24, color="black")),
            font=dict(size=20, color="black")
        ),
        boxmode="group",
        template=DEFAULT_TEMPLATE
    )

    return fig


def plot_box_evals(dataframe):
    """
    Create a box plot comparing algorithm runtime (evaluations) across noise levels.

    Shows the distribution of n_evals for each algorithm at each noise level.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - n_evals: Total fitness evaluations used by the algorithm

    Returns:
        go.Figure: Box plot comparing algorithms
    """
    df = dataframe.copy()
    df = df[['algo_name', 'noise', 'n_evals']]

    noise_levels = sorted(df['noise'].unique())

    fig = px.box(
        df,
        x="noise",
        y="n_evals",
        color="algo_name",
        category_orders={"noise": noise_levels},
        points=False
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="d, where d x mean(W) is s.d. of noise",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        yaxis=dict(
            title=dict(
                text="Runtime (evaluations)",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        legend=dict(
            title=dict(font=dict(size=24, color="black")),
            font=dict(size=20, color="black")
        ),
        boxmode="group",
        template=DEFAULT_TEMPLATE
    )

    return fig


def plot_box_mo(dataframe):
    """
    Create a box plot for multi-objective performance using hypervolume.

    Shows the distribution of hypervolume values for each algorithm at each
    noise level.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - final_true_hv: Final true hypervolume value

    Returns:
        go.Figure: Box plot comparing algorithms
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

    noise_levels = sorted(df['noise'].unique())

    fig = px.box(
        df,
        x="noise",
        y="final_true_hv",
        color="algo_name",
        category_orders={"noise": noise_levels},
        points=False
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="d, where d x mean(W) is s.d. of noise",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        yaxis=dict(
            title=dict(
                text="Final Hypervolume",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        legend=dict(
            title=dict(font=dict(size=24, color="black")),
            font=dict(size=20, color="black")
        ),
        boxmode="group",
        template=DEFAULT_TEMPLATE
    )

    return fig
