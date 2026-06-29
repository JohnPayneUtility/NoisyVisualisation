"""
Box plots for algorithm performance comparison.

This module provides box plots for comparing algorithm performance
distributions across different noise levels.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ..base import DEFAULT_TEMPLATE, create_empty_figure


def _viridis_colors(n, colorscale='Viridis'):
    if n == 1:
        return [px.colors.sample_colorscale(colorscale, [0.5])[0]]
    positions = [i / (n - 1) for i in range(n)]
    return px.colors.sample_colorscale(colorscale, positions)


def plot_box(dataframe, fitness_mode='best', problem_goal='maximise', xaxis_title=None, colorscale='Viridis'):
    """
    Create a box plot comparing algorithm performance across noise levels.

    Shows the distribution of performance values for each algorithm at each
    noise level.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - max_fit: Best fitness value recorded across the run (maximisation)
            - min_fit: Best fitness value recorded across the run (minimisation)
            - final_fit: Final fitness value at end of run
            - max_fit_noisy / min_fit_noisy / final_fit_noisy: noisy counterparts
        fitness_mode: 'best'/'final' to plot true fitness, 'best_noisy'/'final_noisy'
            to plot noisy fitness
        problem_goal: 'maximise' or 'minimise' — determines which column is 'best'

    Returns:
        go.Figure: Box plot comparing algorithms
    """
    df = dataframe.copy()

    if fitness_mode == 'final':
        fit_col = 'final_fit'
        yaxis_label = 'Final solution found'
    elif fitness_mode == 'final_noisy':
        fit_col = 'final_fit_noisy'
        yaxis_label = 'Final noisy solution found'
    elif fitness_mode == 'best_noisy':
        fit_col = 'min_fit_noisy' if problem_goal == 'minimise' else 'max_fit_noisy'
        yaxis_label = 'Best noisy solution found'
    elif problem_goal == 'minimise':
        fit_col = 'min_fit'
        yaxis_label = 'Best solution found'
    else:
        fit_col = 'max_fit'
        yaxis_label = 'Best solution found'

    df = df[['algo_name', 'noise', fit_col]]

    noise_levels = sorted(df['noise'].unique())
    algos = sorted(df['algo_name'].unique())
    colors = _viridis_colors(len(algos), colorscale)

    fig = px.box(
        df,
        x="noise",
        y=fit_col,
        color="algo_name",
        category_orders={"noise": noise_levels, "algo_name": algos},
        color_discrete_sequence=colors,
        points=False
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=xaxis_title or "d, where d x mean(W) is s.d. of noise",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        yaxis=dict(
            title=dict(
                text=yaxis_label,
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


def plot_box_evals(dataframe, fitness_mode='final', xaxis_title=None, colorscale='Viridis'):
    """
    Create a box plot comparing algorithm runtime (evaluations) across noise levels.

    Shows the distribution of evaluations for each algorithm at each noise level.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - n_evals: Total fitness evaluations used by the algorithm
            - evals_to_best: Evaluations consumed until best fitness was found (optional)
            - evals_to_final: Evaluations consumed until final fitness was found (optional)
            - evals_to_best_noisy / evals_to_final_noisy: noisy counterparts (optional)
        fitness_mode: 'best'/'final' to plot true-fitness evals, 'best_noisy'/'final_noisy'
            to plot noisy-fitness evals

    Returns:
        go.Figure: Box plot comparing algorithms
    """
    df = dataframe.copy()

    if fitness_mode == 'best' and 'evals_to_best' in df.columns:
        eval_col = 'evals_to_best'
        yaxis_label = 'Evaluations to best found fitness'
    elif fitness_mode == 'final' and 'evals_to_final' in df.columns:
        eval_col = 'evals_to_final'
        yaxis_label = 'Evaluations to final found fitness'
    elif fitness_mode == 'best_noisy' and 'evals_to_best_noisy' in df.columns:
        eval_col = 'evals_to_best_noisy'
        yaxis_label = 'Evaluations to best found noisy fitness'
    elif fitness_mode == 'final_noisy' and 'evals_to_final_noisy' in df.columns:
        eval_col = 'evals_to_final_noisy'
        yaxis_label = 'Evaluations to final found noisy fitness'
    else:
        eval_col = 'n_evals'
        yaxis_label = 'Evaluations to final fitness'

    df = df[['algo_name', 'noise', eval_col]].dropna(subset=[eval_col])

    noise_levels = sorted(df['noise'].unique())
    algos = sorted(df['algo_name'].unique())
    colors = _viridis_colors(len(algos), colorscale)

    fig = px.box(
        df,
        x="noise",
        y=eval_col,
        color="algo_name",
        category_orders={"noise": noise_levels, "algo_name": algos},
        color_discrete_sequence=colors,
        points=False
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=xaxis_title or "d, where d x mean(W) is s.d. of noise",
                font=dict(size=24, color="black")
            ),
            tickfont=dict(size=20, color="black")
        ),
        yaxis=dict(
            title=dict(
                text=yaxis_label,
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


def plot_box_misjudgements_so(dataframe, xaxis_title=None, colorscale='Viridis'):
    """
    Create a box plot showing the number of misjudgements per run for each
    algorithm at each noise level (single-objective problems only).

    A misjudgement is any step where the representative solution's true fitness
    moved in the wrong direction relative to the previous step.

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - n_misjudgements: Count of misjudgements for this run
        xaxis_title: Label for the x axis (noise parameter description)

    Returns:
        go.Figure: Grouped box plot, one series per algorithm
    """
    df = dataframe.copy()

    if 'n_misjudgements' not in df.columns:
        return create_empty_figure('No misjudgement data available')

    df = df[['algo_name', 'noise', 'n_misjudgements']].dropna(subset=['n_misjudgements'])

    if df.empty:
        return create_empty_figure('No misjudgement data available')

    noise_levels = sorted(df['noise'].unique())
    algos = sorted(df['algo_name'].unique())
    colors = _viridis_colors(len(algos), colorscale)

    fig = px.box(
        df,
        x='noise',
        y='n_misjudgements',
        color='algo_name',
        category_orders={'noise': noise_levels, 'algo_name': algos},
        color_discrete_sequence=colors,
        points=False,
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=xaxis_title or 'Noise level',
                font=dict(size=24, color='black'),
            ),
            tickfont=dict(size=20, color='black'),
        ),
        yaxis=dict(
            title=dict(
                text='No. misjudgements',
                font=dict(size=24, color='black'),
            ),
            tickfont=dict(size=20, color='black'),
        ),
        legend=dict(
            title=dict(font=dict(size=24, color='black')),
            font=dict(size=20, color='black'),
        ),
        boxmode='group',
        template=DEFAULT_TEMPLATE,
    )

    return fig


def plot_box_advanced_misjudgements_so(dataframe, algo_name, xaxis_title=None, colorscale='Viridis'):
    """
    Create a box plot breaking down misjudgement types for a single algorithm
    (single-objective problems only).

    Unlike plot_box_misjudgements_so (one series per algorithm), this plot
    fixes the algorithm and shows one series per mistake type:
        - 'increasing noise': steps where the noise magnitude
          |true_fit - noisy_fit| grew relative to the previous step.
        - 'comparison': steps where the true fitness got worse but the noisy
          fitness got better relative to the previous step (the noisy signal
          would have favoured a solution that was actually worse).
        - 'constraint': visits where the true fitness is negative (a constraint
          violation).

    Args:
        dataframe: DataFrame with columns:
            - algo_name: Algorithm identifier
            - noise: Noise level
            - n_increasing_noise: Count of increasing-noise steps for this run
            - n_comparison_misjudgements: Count of comparison-misjudgement steps for this run
            - n_constraint_misjudgements: Count of constraint-violation visits for this run
        algo_name: The single algorithm to show on this plot
        xaxis_title: Label for the x axis (noise parameter description)

    Returns:
        go.Figure: Grouped box plot, one series per mistake type
    """
    df = dataframe.copy()

    mistake_columns = {
        'n_increasing_noise': 'increasing noise',
        'n_comparison_misjudgements': 'comparison',
        'n_constraint_misjudgements': 'constraint',
    }
    available_columns = {col: label for col, label in mistake_columns.items() if col in df.columns}

    if not available_columns or 'algo_name' not in df.columns:
        return create_empty_figure('No misjudgement data available')

    df = df[df['algo_name'] == algo_name]

    if df.empty:
        return create_empty_figure('No misjudgement data available')

    df = pd.concat([
        df[['noise', col]].rename(columns={col: 'count'}).assign(mistake_type=label).dropna(subset=['count'])
        for col, label in available_columns.items()
    ], ignore_index=True)

    if df.empty:
        return create_empty_figure('No misjudgement data available')

    noise_levels = sorted(df['noise'].unique())
    mistake_types = sorted(df['mistake_type'].unique())
    colors = _viridis_colors(len(mistake_types), colorscale)

    fig = px.box(
        df,
        x='noise',
        y='count',
        color='mistake_type',
        category_orders={'noise': noise_levels, 'mistake_type': mistake_types},
        color_discrete_sequence=colors,
        points=False,
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=xaxis_title or 'Noise level',
                font=dict(size=24, color='black'),
            ),
            tickfont=dict(size=20, color='black'),
        ),
        yaxis=dict(
            title=dict(
                text='No. misjudgements',
                font=dict(size=24, color='black'),
            ),
            tickfont=dict(size=20, color='black'),
        ),
        legend=dict(
            title=dict(font=dict(size=24, color='black')),
            font=dict(size=20, color='black'),
        ),
        boxmode='group',
        template=DEFAULT_TEMPLATE,
    )

    return fig


def plot_box_mo(dataframe, colorscale='Viridis'):
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
    algos = sorted(df['algo_name'].unique())
    colors = _viridis_colors(len(algos), colorscale)

    fig = px.box(
        df,
        x="noise",
        y="final_true_hv",
        color="algo_name",
        category_orders={"noise": noise_levels, "algo_name": algos},
        color_discrete_sequence=colors,
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
