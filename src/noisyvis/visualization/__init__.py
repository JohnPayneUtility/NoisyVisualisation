"""
Visualization module for the NoisyVisualisation dashboard.

This module provides a clean interface for building and styling network graphs,
calculating node positions, and creating Plotly visualizations.

Main entry points:
- parse_callback_inputs: Convert raw callback parameters to PlotConfig
- build_graph: Add nodes and edges to a NetworkX graph
- calculate_positions: Calculate 2D positions for nodes
- style_nodes: Apply sizes and colors to nodes
- build_traces: Create Plotly trace objects
- create_figure: Assemble traces into a figure
"""

# Configuration
from ..viz.config import (
    PlotConfig,
    NodeSizeConfig,
    OpacityConfig,
    AxisConfig,
    CameraConfig,
    STNConfig,
    LONConfig,
    NoisyLONConfig,
    parse_callback_inputs,
)

# Distance metrics (imported from common module for sharing with plotting)
from ..common import (
    hamming_distance,
    normed_hamming_distance,
    sol_tuple_ints,
    avg_min_hamming_A_to_B,
    front_distance,
    sol_key_str,
    lookup_map,
)

# Graph building
from ..viz.graph.stn import (
    generate_run_summary_string,
    print_hamming_transitions,
    add_stn_trajectories,
    add_mo_fronts,
    add_prior_noise_stn_v4,
    add_prior_noise_stn_v5,
    add_prior_noise_stn_algo_pov,
    debug_mo_counts,
)
from ..viz.graph.lon import (
    add_lon_nodes,
    add_lon_edges,
)

# Node styling
from ..viz.styling import (
    apply_generation_coloring,
    apply_node_sizes,
    apply_node_colors,
    style_nodes,
)

# Node positioning
from ..viz.layout import (
    calculate_positions_mo,
    calculate_positions_so,
    calculate_positions,
    create_hover_text,
)

# LON stats plots
from ..viz.plots.lon_stats import (
    AXIS_OPTIONS as LON_SCATTER_AXIS_OPTIONS,
    AXIS_LABELS as LON_SCATTER_AXIS_LABELS,
    DEFAULT_X_AXIS as LON_SCATTER_DEFAULT_X_AXIS,
    DEFAULT_Y_AXIS as LON_SCATTER_DEFAULT_Y_AXIS,
    PLOT_STYLE_OPTIONS as LON_SCATTER_PLOT_STYLE_OPTIONS,
    DEFAULT_PLOT_STYLE as LON_SCATTER_DEFAULT_PLOT_STYLE,
    plot_lon_scatter,
    plot_lon_violin,
    plot_lon_stats,
    plot_lon_stats_multi,
)

# Trace building
from ..viz.traces import (
    create_edge_traces,
    create_edge_label_trace,
    create_node_traces,
    create_boxplot_traces,
    create_axis_settings,
    create_figure,
    build_all_traces,
    create_guide_traces,
)

__all__ = [
    # Config
    'PlotConfig',
    'NodeSizeConfig',
    'OpacityConfig',
    'AxisConfig',
    'CameraConfig',
    'STNConfig',
    'LONConfig',
    'NoisyLONConfig',
    'parse_callback_inputs',
    # Distance metrics
    'hamming_distance',
    'normed_hamming_distance',
    'sol_tuple_ints',
    'avg_min_hamming_A_to_B',
    'front_distance',
    'sol_key_str',
    'lookup_map',
    # Graph building
    'generate_run_summary_string',
    'print_hamming_transitions',
    'add_stn_trajectories',
    'add_mo_fronts',
    'add_prior_noise_stn_v4',
    'add_prior_noise_stn_v5',
    'add_prior_noise_stn_algo_pov',
    'add_lon_nodes',
    'add_lon_edges',
    'debug_mo_counts',
    # Node styling
    'apply_generation_coloring',
    'apply_node_sizes',
    'apply_node_colors',
    'style_nodes',
    # Node positioning
    'calculate_positions_mo',
    'calculate_positions_so',
    'calculate_positions',
    'create_hover_text',
    # LON stats plots
    'LON_SCATTER_AXIS_OPTIONS',
    'LON_SCATTER_AXIS_LABELS',
    'LON_SCATTER_DEFAULT_X_AXIS',
    'LON_SCATTER_DEFAULT_Y_AXIS',
    'LON_SCATTER_PLOT_STYLE_OPTIONS',
    'LON_SCATTER_DEFAULT_PLOT_STYLE',
    'plot_lon_scatter',
    'plot_lon_violin',
    'plot_lon_stats',
    'plot_lon_stats_multi',
    # Trace building
    'create_edge_traces',
    'create_edge_label_trace',
    'create_node_traces',
    'create_boxplot_traces',
    'create_axis_settings',
    'create_figure',
    'build_all_traces',
    'create_guide_traces',
]
