"""
Plot registry for dynamic plot type dispatch.

This module provides a registry pattern for looking up plot functions
by name, eliminating long if/elif chains in callbacks.

Usage:
    from src.plotting import get_pareto_plot, get_performance_plot

    # In callback
    plot_func = get_pareto_plot(plot_type)
    if plot_func:
        return plot_func(frontdata, series_labels, **kwargs)
"""

import plotly.graph_objects as go

from .pareto import (
    plot_basic,
    plot_subplots,
    plot_subplots_multi,
    plot_subplots_highlighted,
    plot_animation,
    plot_noisy,
    plot_ind_vs_dist,
    plot_igd_vs_dist,
    plot_progress_per_movement,
    plot_movement_correlation,
    plot_move_delta_histograms,
    plot_objective_vs_decision,
)

from .performance import (
    plot_line,
    plot_box,
    plot_line_mo,
    plot_box_mo,
)


# ==============================================================================
# Pareto Plot Registry
# ==============================================================================

# Map plot type strings (matching dropdown values in Dashboard.py) to functions
PARETO_PLOTS = {
    'Basic': plot_basic,
    'Subplots': plot_subplots,
    'SubplotsMulti': plot_subplots_multi,
    'SubplotsHighlight': plot_subplots_highlighted,
    'paretoAnimation': plot_animation,
    'Noisy': plot_noisy,
    'IndVsDist': plot_ind_vs_dist,
    'IGDVsDist': plot_igd_vs_dist,
    'PPM': plot_progress_per_movement,
    'MoveCorr': plot_movement_correlation,
    'Hist': plot_move_delta_histograms,
    'Scatter': plot_objective_vs_decision,
}


def get_pareto_plot(plot_type: str):
    """
    Get a Pareto plot function by type name.

    Args:
        plot_type: The plot type string (e.g., 'Basic', 'Subplots')

    Returns:
        Callable or None: The plot function, or None if not found
    """
    return PARETO_PLOTS.get(plot_type)


def list_pareto_plot_types():
    """
    Get a list of available Pareto plot type names.

    Returns:
        list: Available plot type names
    """
    return list(PARETO_PLOTS.keys())


# ==============================================================================
# Performance Plot Registry
# ==============================================================================

PERFORMANCE_PLOTS = {
    'line': plot_line,
    'box': plot_box,
    'line_mo': plot_line_mo,
    'box_mo': plot_box_mo,
}


def get_performance_plot(plot_type: str):
    """
    Get a performance plot function by type name.

    Args:
        plot_type: The plot type string (e.g., 'line', 'box')

    Returns:
        Callable or None: The plot function, or None if not found
    """
    return PERFORMANCE_PLOTS.get(plot_type)


def list_performance_plot_types():
    """
    Get a list of available performance plot type names.

    Returns:
        list: Available plot type names
    """
    return list(PERFORMANCE_PLOTS.keys())


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_pareto_plot(plot_type: str, frontdata, series_labels, **kwargs):
    """
    Create a Pareto plot by type name.

    Convenience function that handles the lookup and fallback to empty figure.

    Args:
        plot_type: The plot type string
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        **kwargs: Additional arguments passed to the plot function

    Returns:
        go.Figure: The resulting plot, or an empty figure if type not found
    """
    plot_func = get_pareto_plot(plot_type)
    if plot_func:
        return plot_func(frontdata, series_labels, **kwargs)
    return go.Figure()  # Fallback empty figure


def create_performance_plot(plot_type: str, dataframe, **kwargs):
    """
    Create a performance plot by type name.

    Convenience function that handles the lookup and fallback to empty figure.

    Args:
        plot_type: The plot type string
        dataframe: DataFrame with performance data
        **kwargs: Additional arguments passed to the plot function

    Returns:
        go.Figure: The resulting plot, or an empty figure if type not found
    """
    plot_func = get_performance_plot(plot_type)
    if plot_func:
        return plot_func(dataframe, **kwargs)
    return go.Figure()  # Fallback empty figure
