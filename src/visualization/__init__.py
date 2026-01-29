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
from .config import (
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
from .graph_builder import (
    generate_run_summary_string,
    print_hamming_transitions,
    add_stn_trajectories,
    add_mo_fronts,
    add_lon_nodes,
    add_lon_edges,
    debug_mo_counts,
)

# Node styling
from .node_styling import (
    apply_generation_coloring,
    apply_node_sizes,
    apply_node_colors,
    style_nodes,
)

# Node positioning
from .node_positioning import (
    calculate_positions_mo,
    calculate_positions_so,
    calculate_positions,
    create_hover_text,
)

# Statistics
from .statistics import (
    LONStatistics,
    calculate_lon_statistics,
    calculate_stn_statistics,
    calculate_graph_summary,
)

# Trace building
from .trace_builder import (
    create_edge_traces,
    create_edge_label_trace,
    create_node_traces,
    create_boxplot_traces,
    create_axis_settings,
    create_figure,
    build_all_traces,
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
    # Statistics
    'LONStatistics',
    'calculate_lon_statistics',
    'calculate_stn_statistics',
    'calculate_graph_summary',
    # Trace building
    'create_edge_traces',
    'create_edge_label_trace',
    'create_node_traces',
    'create_boxplot_traces',
    'create_axis_settings',
    'create_figure',
    'build_all_traces',
]
