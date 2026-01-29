"""
Plotting module for the NoisyVisualisation dashboard.

This module provides a clean, unified interface for all 2D plotting functions,
organized by purpose:

- pareto: Pareto front visualizations (multi-objective optimization)
- performance: 2D performance comparison plots (line/box)
- registry: Dynamic plot dispatch by type name

Quick Start:
    # Using the registry (recommended for dynamic dispatch)
    from src.plotting import get_pareto_plot, create_pareto_plot

    plot_func = get_pareto_plot('Basic')
    fig = plot_func(frontdata, series_labels)

    # Or use the convenience function
    fig = create_pareto_plot('Basic', frontdata, series_labels)

    # Direct imports for static usage
    from src.plotting.pareto import plot_basic, plot_subplots
    from src.plotting.performance import plot_line, plot_box

Backward Compatibility:
    All original function names from plotParetoFrontMain.py are preserved
    as aliases. Existing code will continue to work unchanged.
"""

# Registry functions for dynamic dispatch
from .registry import (
    get_pareto_plot,
    get_performance_plot,
    list_pareto_plot_types,
    list_performance_plot_types,
    create_pareto_plot,
    create_performance_plot,
    PARETO_PLOTS,
    PERFORMANCE_PLOTS,
)

# Pareto plot functions (new consistent names)
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

# Pareto plot functions (backward-compatible aliases)
from .pareto import (
    plotParetoFront,
    plotParetoFrontSubplots,
    plotParetoFrontSubplotsMulti,
    PlotparetoFrontSubplotsHighlighted,
    plotParetoFrontAnimation,
    plotParetoFrontNoisy,
    plotParetoFrontIndVsDist,
    plotParetoFrontIGDVsDist,
    plotProgressPerMovementRatio,
    plotMovementCorrelation,
    plotMoveDeltaHistograms,
    plotObjectiveVsDecisionScatter,
)

# Performance plot functions (new consistent names)
from .performance import (
    plot_line,
    plot_line_mo,
    plot_box,
    plot_box_mo,
)

# Performance plot functions (backward-compatible aliases)
from .performance import (
    plot2d_line,
    plot2d_box,
    plot2d_line_mo,
    plot2d_box_mo,
)

# Base utilities
from .base import (
    front_distance,
    generation_color,
    symmetric_range,
    create_empty_figure,
    apply_standard_layout,
    get_series_info,
    get_run_entries,
    compute_generation_range,
    DEFAULT_TEMPLATE,
    DEFAULT_COLORSCALE,
    PARETO_COLORSCALE,
)

__all__ = [
    # Registry
    'get_pareto_plot',
    'get_performance_plot',
    'list_pareto_plot_types',
    'list_performance_plot_types',
    'create_pareto_plot',
    'create_performance_plot',
    'PARETO_PLOTS',
    'PERFORMANCE_PLOTS',

    # Pareto plots (new names)
    'plot_basic',
    'plot_subplots',
    'plot_subplots_multi',
    'plot_subplots_highlighted',
    'plot_animation',
    'plot_noisy',
    'plot_ind_vs_dist',
    'plot_igd_vs_dist',
    'plot_progress_per_movement',
    'plot_movement_correlation',
    'plot_move_delta_histograms',
    'plot_objective_vs_decision',

    # Pareto plots (backward-compatible)
    'plotParetoFront',
    'plotParetoFrontSubplots',
    'plotParetoFrontSubplotsMulti',
    'PlotparetoFrontSubplotsHighlighted',
    'plotParetoFrontAnimation',
    'plotParetoFrontNoisy',
    'plotParetoFrontIndVsDist',
    'plotParetoFrontIGDVsDist',
    'plotProgressPerMovementRatio',
    'plotMovementCorrelation',
    'plotMoveDeltaHistograms',
    'plotObjectiveVsDecisionScatter',

    # Performance plots (new names)
    'plot_line',
    'plot_line_mo',
    'plot_box',
    'plot_box_mo',

    # Performance plots (backward-compatible)
    'plot2d_line',
    'plot2d_box',
    'plot2d_line_mo',
    'plot2d_box_mo',

    # Base utilities
    'front_distance',
    'generation_color',
    'symmetric_range',
    'create_empty_figure',
    'apply_standard_layout',
    'get_series_info',
    'get_run_entries',
    'compute_generation_range',
    'DEFAULT_TEMPLATE',
    'DEFAULT_COLORSCALE',
    'PARETO_COLORSCALE',
]
