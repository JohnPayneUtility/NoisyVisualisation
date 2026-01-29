"""
Base utilities and common functions for the plotting module.

This module provides:
- Color schemes and style constants for 2D plots
- Re-exports of shared distance metrics from common module
- Utility functions for creating and styling Plotly figures
"""

import numpy as np
from plotly.colors import sample_colorscale
import plotly.graph_objects as go

# Re-export distance metrics from common for convenience
from ..common import (
    front_distance,
    avg_min_hamming_A_to_B,
    sol_tuple_ints,
    hamming_distance,
    normed_hamming_distance,
)


# ==============================================================================
# Color Schemes and Style Constants
# ==============================================================================

# Default colorscales used across plots
DEFAULT_COLORSCALE = 'Viridis'
PARETO_COLORSCALE = 'sunsetdark'

# Common template
DEFAULT_TEMPLATE = 'plotly_white'

# Default marker sizes
DEFAULT_MARKER_SIZE = 8
DEFAULT_LINE_WIDTH = 1.5


# ==============================================================================
# Color Utilities
# ==============================================================================

def generation_color(generation, gen_min, gen_max, scale=PARETO_COLORSCALE):
    """
    Get a color from a colorscale based on generation index.

    Args:
        generation: Current generation index
        gen_min: Minimum generation index
        gen_max: Maximum generation index
        scale: Plotly colorscale name (default: 'sunsetdark')

    Returns:
        str: RGB color string (e.g., "rgb(r,g,b)")
    """
    t = 0.5 if gen_max == gen_min else (generation - gen_min) / (gen_max - gen_min)
    return sample_colorscale(scale, [t])[0]


def symmetric_range(arr, pad=0.05):
    """
    Compute a symmetric range around zero for an array.

    Args:
        arr: Input array
        pad: Padding factor (default: 0.05, i.e., 5% padding)

    Returns:
        list: [min, max] range symmetric around zero
    """
    m = np.nanmax(np.abs(arr))
    if not np.isfinite(m) or m == 0:
        m = 1.0
    m *= (1 + pad)
    return [-m, m]


# ==============================================================================
# Figure Creation Helpers
# ==============================================================================

def create_empty_figure(title="No data available"):
    """
    Create an empty figure with a title message.

    Args:
        title: Title to display on the empty figure

    Returns:
        go.Figure: Empty Plotly figure with the given title
    """
    fig = go.Figure()
    fig.update_layout(title=title, template=DEFAULT_TEMPLATE)
    return fig


def apply_standard_layout(fig, title=None, xaxis_title=None, yaxis_title=None,
                          legend_title=None, height=None):
    """
    Apply standard layout settings to a figure.

    Args:
        fig: Plotly figure to update
        title: Figure title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        legend_title: Legend title
        height: Figure height in pixels

    Returns:
        go.Figure: The updated figure (also modifies in place)
    """
    layout_kwargs = {'template': DEFAULT_TEMPLATE}

    if title is not None:
        layout_kwargs['title'] = title
    if xaxis_title is not None:
        layout_kwargs['xaxis_title'] = xaxis_title
    if yaxis_title is not None:
        layout_kwargs['yaxis_title'] = yaxis_title
    if legend_title is not None:
        layout_kwargs['legend_title'] = legend_title
    if height is not None:
        layout_kwargs['height'] = height

    fig.update_layout(**layout_kwargs)
    return fig


# ==============================================================================
# Data Extraction Helpers
# ==============================================================================

def get_series_info(frontdata, series_labels, group_idx=0):
    """
    Extract series information from frontdata.

    Args:
        frontdata: List of data groups
        series_labels: List of series labels
        group_idx: Index of the group to extract (default: 0)

    Returns:
        tuple: (runs_full, series_name) or (None, None) if invalid
    """
    if not frontdata:
        return None, None

    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    return runs_full, series_name


def get_run_entries(runs_full, run_idx=0):
    """
    Get generation entries for a specific run.

    Args:
        runs_full: List of runs
        run_idx: Index of the run to extract (default: 0)

    Returns:
        list: Generation entries for the run, or empty list if invalid
    """
    if not runs_full or len(runs_full) <= run_idx:
        return []
    return runs_full[run_idx]


def compute_generation_range(gen_entries):
    """
    Compute the min and max generation indices from entries.

    Args:
        gen_entries: List of generation entry dictionaries

    Returns:
        tuple: (gen_min, gen_max)
    """
    gens = [entry.get('gen_idx', 0) for entry in gen_entries]
    if not gens:
        return 0, 0
    return min(gens), max(gens)
