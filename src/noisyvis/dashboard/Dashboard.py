import math
import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px  # for continuous color scales
import networkx as nx
import numpy as np
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import ClassicalMDS
from sklearn.manifold import TSNE
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace as dataclass_replace
import os

from .helpers import (
    convert_to_single_edges_format,
    convert_to_split_edges_format,
    filter_local_optima,
    filter_negative_LO,
    get_mean_run,
    get_median_run,
    select_top_runs_by_fitness,
)
from ..problems.instances import load_problem_KP, get_knapsack_problem_stats, interpret_correlation
from .layout import create_layout, TAB_STYLE, TAB_SELECTED_STYLE, _build_schematic_figure, _build_schematic_legend
from .layout.stores import LON_TABLE_SELECTED_PID_STORE

# Visualization module imports
from ..viz import (
    parse_callback_inputs,
    PlotConfig,
    generate_run_summary_string,
    add_stn_trajectories,
    add_mo_fronts,
    add_prior_noise_stn_v4,
    add_prior_noise_stn_v5,
    add_prior_noise_stn_algo_pov,
    add_lon_nodes,
    add_lon_edges,
    debug_mo_counts,
    style_nodes,
    calculate_positions,
    LON_SCATTER_AXIS_LABELS,
    LON_SCATTER_DEFAULT_X_AXIS,
    LON_SCATTER_DEFAULT_Y_AXIS,
    LON_SCATTER_DEFAULT_PLOT_STYLE,
    plot_lon_stats,
    plot_lon_stats_multi,
    build_all_traces,
    create_guide_traces,
    create_axis_settings,
    create_figure,
)
from ..analysis.graph_stats import (
    calculate_lon_statistics,
    compute_node_feasibility_error,
    compute_pairwise_correlations,
    compute_correlation_pair,
)
from .components import (
    build_correlation_table,
    build_selected_correlation_display,
)
from ..common import is_continuous_solution

# Plotting module imports - using registry for dynamic dispatch
from ..viz.plots import get_pareto_plot
from ..viz.plots.performance import plot2d_line, plot2d_box, plot2d_line_mo, plot2d_box_mo, plot2d_line_evals, plot2d_box_evals, plot2d_box_penalty, plot2d_box_misjudgements_so, plot2d_box_advanced_misjudgements_so

# ==========
# Data Loading
# ==========
from .data import DashboardData, DISPLAY2_HIDDEN_COLUMNS, LON_HIDDEN_COLUMNS
from .tables import create_display2_df
from ..analysis.misjudgements import (
    increasing_noise_step_indices,
    comparison_misjudgement_step_indices,
    constraint_misjudgement_step_indices,
)
from .columns import DISPLAY1_COLUMNS

from .data import (
    df,
    df_LONs,
    df_no_lists,
    display1_df,
    display2_df,
    LON_display_columns,
    display2_hidden_cols,
    experiment_names,
    experiment_descriptions,
)


# ==========
# Main Dashboard App
# ==========

from .instance import app  # noqa: E402

# ---------- Layout Definition ----------
# The layout is defined in the layout module for better organization.

app.layout = create_layout(display2_df, display2_hidden_cols, display1_df, df_LONs, LON_display_columns, experiment_names, experiment_descriptions)


# ------------------------------
# Callbacks: Schematic
# ------------------------------

from .callbacks import schematic  # noqa: E402,F401


# ------------------------------
# Callbacks: Update Selection Stores
# ------------------------------

from .callbacks import selection  # noqa: E402,F401

# ------------------------------
# 2D Plot
# ------------------------------
from .callbacks import performance  # noqa: E402,F401
# ------------------------------
# LON Plots
# ------------------------------

from .callbacks import graph_data  # noqa: E402,F401

from .callbacks import visualization  # noqa: E402,F401

from .callbacks import pareto  # noqa: E402,F401
    
# ==========
# RUN
# ==========

if __name__ == '__main__':
    # app.run_server(debug=True)
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8050, debug=True, use_reloader=False)