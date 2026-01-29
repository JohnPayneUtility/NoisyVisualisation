"""
Pareto front plotting module.

This module provides functions for visualizing Pareto fronts from multi-objective
optimization runs. Functions are organized by plot type:

- basic: Single Pareto front plots
- subplots: Grid layouts showing multiple generations
- animation: Animated generation progression
- noisy: Comparing noisy vs clean fitness values
- analysis: Hypervolume and distance analysis plots
- correlation: Movement correlation and scatter plots

All functions follow a consistent interface:
    plot_*(frontdata, series_labels, **kwargs) -> go.Figure
"""

from .basic import plot_basic
from .subplots import (
    plot_subplots,
    plot_subplots_multi,
    plot_subplots_highlighted,
)
from .animation import plot_animation
from .noisy import plot_noisy
from .analysis import (
    plot_ind_vs_dist,
    plot_igd_vs_dist,
    plot_progress_per_movement,
)
from .correlation import (
    plot_movement_correlation,
    plot_move_delta_histograms,
    plot_objective_vs_decision,
)

# Backward-compatible aliases (original function names)
plotParetoFront = plot_basic
plotParetoFrontSubplots = plot_subplots
plotParetoFrontSubplotsMulti = plot_subplots_multi
PlotparetoFrontSubplotsHighlighted = plot_subplots_highlighted
plotParetoFrontAnimation = plot_animation
plotParetoFrontNoisy = plot_noisy
plotParetoFrontIndVsDist = plot_ind_vs_dist
plotParetoFrontIGDVsDist = plot_igd_vs_dist
plotProgressPerMovementRatio = plot_progress_per_movement
plotMovementCorrelation = plot_movement_correlation
plotMoveDeltaHistograms = plot_move_delta_histograms
plotObjectiveVsDecisionScatter = plot_objective_vs_decision

__all__ = [
    # New consistent names
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
    # Backward-compatible aliases
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
]
