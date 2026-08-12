"""
LON Stats plots: relate LON node properties (e.g. neighbourhood feasibility)
to the sampled noisy-fitness distributions shown as box/IQR plots.
"""

from typing import Any, Dict, List

import plotly.graph_objects as go
from dash import dash_table

# Quantities selectable for the LON stats scatter plot axes.
# Value = key into a compute_node_feasibility_error() stats dict.
AXIS_OPTIONS = [
    ('neigh_feas', 'Neighbourhood Feasibility'),
    ('error', 'Sampling Error'),
    ('abs_error', 'Absolute Error'),
    ('iqr', 'Sample Range (Q3-Q1)'),
    ('fitness', 'Fitness'),
    ('median', 'Median Sampled Fitness'),
]
AXIS_LABELS = dict(AXIS_OPTIONS)
DEFAULT_X_AXIS = 'neigh_feas'
DEFAULT_Y_AXIS = 'error'

# Plot styles for the LON node scatter/violin chart.
PLOT_STYLE_OPTIONS = [
    ('scatter', 'Scatter'),
    ('violin', 'Violin'),
]
DEFAULT_PLOT_STYLE = 'scatter'


def plot_lon_scatter(node_stats: List[Dict[str, Any]], x_key: str, y_key: str) -> go.Figure:
    """
    Scatter plot of one LON node quantity against another, for the nodes
    currently drawn in the main NLon_box/NLon_IQR plot.

    Args:
        node_stats: Output of compute_node_feasibility_error
        x_key: Key into each node_stats dict to use as x (see AXIS_OPTIONS)
        y_key: Key into each node_stats dict to use as y (see AXIS_OPTIONS)

    Returns:
        go.Figure: Plotly scatter figure
    """
    fig = go.Figure()

    x_label = AXIS_LABELS.get(x_key, x_key)
    y_label = AXIS_LABELS.get(y_key, y_key)

    if not node_stats:
        fig.update_layout(title="No data - select NLon_box or NLon_IQR with a loaded LON")
        return fig

    x = [s[x_key] for s in node_stats]
    y = [s[y_key] for s in node_stats]
    hover_text = [
        f"{s['node']}<br>Fitness: {s['fitness']:.3f}<br>Median sample: {s['median']:.3f}"
        f"<br>Error: {s['error']:.3f}<br>Sample range: {s['iqr']:.3f}"
        for s in node_stats
    ]

    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=8, opacity=0.75),
        text=hover_text,
        hoverinfo='text',
        name='LON nodes',
    ))

    # A zero reference line is only meaningful for the signed error axis
    if y_key == 'error':
        fig.add_hline(y=0, line_dash="dash")
    if x_key == 'error':
        fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        title=f"{x_label} vs. {y_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )

    return fig


def plot_lon_violin(node_stats: List[Dict[str, Any]], x_key: str, y_key: str) -> go.Figure:
    """
    Violin plot of y grouped by each unique x value, for the nodes currently
    drawn in the main NLon_box/NLon_IQR plot.

    Intended for x quantities that only take a handful of distinct values
    (e.g. neighbourhood feasibility for a fixed neighbourhood size) - each
    unique x value becomes its own violin showing the spread of y across
    the nodes sharing that value. No binning is performed, so x quantities
    with mostly-unique values will produce one thin violin per node.

    Args:
        node_stats: Output of compute_node_feasibility_error
        x_key: Key into each node_stats dict to use as the grouping x axis
        y_key: Key into each node_stats dict to use as y

    Returns:
        go.Figure: Plotly violin figure
    """
    fig = go.Figure()

    x_label = AXIS_LABELS.get(x_key, x_key)
    y_label = AXIS_LABELS.get(y_key, y_key)

    if not node_stats:
        fig.update_layout(title="No data - select NLon_box or NLon_IQR with a loaded LON")
        return fig

    x = [s[x_key] for s in node_stats]
    y = [s[y_key] for s in node_stats]

    fig.add_trace(go.Violin(
        x=x, y=y,
        box_visible=True,
        meanline_visible=True,
        points='all',
        name='LON nodes',
    ))

    if y_key == 'error':
        fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title=f"{x_label} vs. {y_label} (grouped by unique {x_label} values)",
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )

    return fig


def plot_lon_stats(node_stats: List[Dict[str, Any]], x_key: str, y_key: str, plot_style: str = DEFAULT_PLOT_STYLE) -> go.Figure:
    """
    Dispatch to the scatter or violin LON node plot based on plot_style.

    Args:
        node_stats: Output of compute_node_feasibility_error
        x_key: Key into each node_stats dict to use as x (see AXIS_OPTIONS)
        y_key: Key into each node_stats dict to use as y (see AXIS_OPTIONS)
        plot_style: 'scatter' or 'violin'

    Returns:
        go.Figure
    """
    if plot_style == 'violin':
        return plot_lon_violin(node_stats, x_key, y_key)
    return plot_lon_scatter(node_stats, x_key, y_key)


def build_selected_correlation_display(correlation: Dict[str, Any], x_label: str, y_label: str) -> dash_table.DataTable:
    """
    Build a one-row DataTable showing the Pearson/Spearman correlation for
    the quantities currently selected on the scatter plot's axes.

    Args:
        correlation: Output of compute_correlation_pair
        x_label: Display label for the x axis quantity
        y_label: Display label for the y axis quantity

    Returns:
        dash_table.DataTable
    """
    pearson = correlation['pearson']
    spearman = correlation['spearman']

    return dash_table.DataTable(
        columns=[
            {'name': 'X', 'id': 'x'},
            {'name': 'Y', 'id': 'y'},
            {'name': 'Pearson r', 'id': 'pearson'},
            {'name': 'Spearman r', 'id': 'spearman'},
        ],
        data=[{
            'x': x_label,
            'y': y_label,
            'pearson': f"{pearson:.3f}" if pearson is not None else 'N/A',
            'spearman': f"{spearman:.3f}" if spearman is not None else 'N/A',
        }],
        style_table={'width': '700px'},
        style_cell={'textAlign': 'center', 'padding': '8px'},
        style_header={'fontWeight': 'bold'},
    )


def build_correlation_table(correlations: List[Dict[str, Any]]) -> dash_table.DataTable:
    """
    Build a small DataTable summarizing Pearson and Spearman correlations
    between neighbourhood feasibility and sampled-fitness-distribution
    quantities.

    Args:
        correlations: Output of compute_pairwise_correlations

    Returns:
        dash_table.DataTable
    """
    data = [{
        'pair': c['label'],
        'pearson': f"{c['pearson']:.3f}" if c['pearson'] is not None else 'N/A',
        'spearman': f"{c['spearman']:.3f}" if c['spearman'] is not None else 'N/A',
    } for c in correlations]

    return dash_table.DataTable(
        columns=[
            {'name': 'Pair', 'id': 'pair'},
            {'name': 'Pearson r', 'id': 'pearson'},
            {'name': 'Spearman r', 'id': 'spearman'},
        ],
        data=data,
        style_table={'width': '700px'},
        style_cell={'textAlign': 'center', 'padding': '8px'},
        style_header={'fontWeight': 'bold'},
    )
