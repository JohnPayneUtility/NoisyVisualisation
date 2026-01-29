"""
Dashboard layout module.

This module provides the layout components for the Dashboard application.
The main entry point is the create_layout function.
"""

from .main_layout import create_layout

# Export style constants that may be needed by callbacks
from .styles import (
    TAB_STYLE,
    TAB_SELECTED_STYLE,
    SECTION_STYLE,
    SELECTION_OUTPUT_STYLE,
)

# Export store ID constants for use in callbacks
from .stores import (
    # Selection stores
    TABLE1_SELECTED_STORE,
    TABLE1_TAB2_SELECTED_STORE,
    TABLE2_SELECTED_STORE,
    DATA_PROBLEM_SPECIFIC,
    OPTIMUM_STORE,
    PID_STORE,
    OPT_GOAL_STORE,
    # Plotting data stores
    PLOT_2D_DATA,
    STN_DATA,
    LON_DATA,
    STN_DATA_PROCESSED,
    STN_SERIES_LABELS,
    NOISY_FITNESSES_DATA,
    AXIS_VALUES,
    # Multiobjective stores
    STN_MO_DATA,
    STN_MO_SERIES_LABELS,
    MO_DATA_PPP,
)

__all__ = [
    # Main layout function
    'create_layout',
    # Style constants
    'TAB_STYLE',
    'TAB_SELECTED_STYLE',
    'SECTION_STYLE',
    'SELECTION_OUTPUT_STYLE',
    # Store IDs
    'TABLE1_SELECTED_STORE',
    'TABLE1_TAB2_SELECTED_STORE',
    'TABLE2_SELECTED_STORE',
    'DATA_PROBLEM_SPECIFIC',
    'OPTIMUM_STORE',
    'PID_STORE',
    'OPT_GOAL_STORE',
    'PLOT_2D_DATA',
    'STN_DATA',
    'LON_DATA',
    'STN_DATA_PROCESSED',
    'STN_SERIES_LABELS',
    'NOISY_FITNESSES_DATA',
    'AXIS_VALUES',
    'STN_MO_DATA',
    'STN_MO_SERIES_LABELS',
    'MO_DATA_PPP',
]
