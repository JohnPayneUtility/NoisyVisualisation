"""
Performance comparison plotting module.

This module provides 2D line and box plots for comparing algorithm performance
across different noise levels. Functions are organized by plot type:

- line_plots: Line plots with error bars
- box_plots: Box plots for distribution comparison

All functions follow a consistent interface:
    plot_*(dataframe, **kwargs) -> go.Figure
"""

from .line_plots import (
    plot_line,
    plot_line_mo,
)
from .box_plots import (
    plot_box,
    plot_box_mo,
)

# Backward-compatible aliases (original function names from DashboardHelpers)
plot2d_line = plot_line
plot2d_box = plot_box
plot2d_line_mo = plot_line_mo
plot2d_box_mo = plot_box_mo

__all__ = [
    # New consistent names
    'plot_line',
    'plot_line_mo',
    'plot_box',
    'plot_box_mo',
    # Backward-compatible aliases
    'plot2d_line',
    'plot2d_box',
    'plot2d_line_mo',
    'plot2d_box_mo',
]
