"""
dcc.Store components for Dashboard state management.

Store IDs are kept as constants to ensure consistency between
layout and callback definitions.
"""
from dash import dcc


# ==========
# Store ID Constants
# ==========

# Selection stores (for table selections)
TABLE1_SELECTED_STORE = "table1-selected-store"
TABLE1_TAB2_SELECTED_STORE = "table1tab2-selected-store"
TABLE2_SELECTED_STORE = "table2-selected-store"
DATA_PROBLEM_SPECIFIC = "data-problem-specific"
OPTIMUM_STORE = "optimum"
PID_STORE = "PID"
OPT_GOAL_STORE = "opt_goal"

# Plotting data stores
PLOT_2D_DATA = "plot_2d_data"
STN_DATA = "STN_data"
LON_DATA = "LON_data"
STN_DATA_PROCESSED = "STN_data_processed"
STN_SERIES_LABELS = "STN_series_labels"
NOISY_FITNESSES_DATA = "noisy_fitnesses_data"
AXIS_VALUES = "axis-values"

# Multiobjective data stores
STN_MO_DATA = "STN_MO_data"
STN_MO_SERIES_LABELS = "STN_MO_series_labels"
MO_DATA_PPP = "MO_data_PPP"


def create_selection_stores():
    """
    Create dcc.Store components for table selection state.

    Returns:
        list: List of dcc.Store components for selection management.
    """
    return [
        dcc.Store(id=TABLE1_SELECTED_STORE, data=[]),
        dcc.Store(id=TABLE1_TAB2_SELECTED_STORE, data=[]),
        dcc.Store(id=DATA_PROBLEM_SPECIFIC, data=[]),
        dcc.Store(id=TABLE2_SELECTED_STORE, data=[]),
        dcc.Store(id=OPTIMUM_STORE, data=[]),
        dcc.Store(id=PID_STORE, data=[]),
        dcc.Store(id=OPT_GOAL_STORE, data=[]),
    ]


def create_plotting_data_stores():
    """
    Create dcc.Store components for plotting data.

    Returns:
        list: List of dcc.Store components for plot data storage.
    """
    return [
        dcc.Store(id=PLOT_2D_DATA, data=[]),
        dcc.Store(id=STN_DATA, data=[]),
        dcc.Store(id=LON_DATA, data=[]),
        dcc.Store(id=STN_DATA_PROCESSED, data=[]),
        dcc.Store(id=STN_SERIES_LABELS, data=[]),
        dcc.Store(id=NOISY_FITNESSES_DATA, data=[]),
        dcc.Store(id=AXIS_VALUES, data=[]),
    ]


def create_multiobjective_stores():
    """
    Create dcc.Store components for multiobjective optimization data.

    Returns:
        list: List of dcc.Store components for MO data storage.
    """
    return [
        dcc.Store(id=STN_MO_DATA, data=[]),
        dcc.Store(id=STN_MO_SERIES_LABELS, data=[]),
        dcc.Store(id=MO_DATA_PPP, data=[]),
    ]


def create_all_stores():
    """
    Create all dcc.Store components for the dashboard.

    Returns:
        list: Complete list of all dcc.Store components.
    """
    stores = []
    stores.extend(create_selection_stores())
    stores.extend(create_plotting_data_stores())
    stores.extend(create_multiobjective_stores())
    return stores
