"""Dash components built from graph-derived statistics.

The two `dash_table` builders that used to live in `visualization/lon_stats_plots.py`. They return
Dash components, so they belong to the dashboard layer rather than the visualisation layer; moving
them here is what lets `noisyvis.viz` stay free of Dash (plan §8, layering rule 2).
"""

from typing import Any, Dict, List
from dash import dash_table


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
