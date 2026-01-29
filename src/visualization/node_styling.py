"""
Node styling functions for the visualization module.

This module contains functions for assigning sizes and colors to nodes
in the graph. Functions modify node attributes in place.
"""

from typing import Dict, Optional, Tuple
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.colors as pc

from .config import PlotConfig
from ..common import sol_tuple_ints, lookup_map


def apply_generation_coloring(G: nx.MultiDiGraph, colorscale: str = "Viridis") -> None:
    """
    Apply generation-based coloring to nodes in the graph.

    Colors nodes based on their generation index, using a continuous colorscale.
    Nodes without a gen_idx attribute are not modified.

    Args:
        G: NetworkX MultiDiGraph to modify
        colorscale: Name of the Plotly colorscale to use
    """
    # Collect all gen_idx to find maximum
    gens = [
        int(d.get("gen_idx"))
        for _, d in G.nodes(data=True)
        if d.get("gen_idx") is not None
    ]
    if not gens:
        return  # no-op if missing

    gmax = max(gens)
    span = max(gmax, 1)  # avoid divide-by-zero if only one generation

    scale = pc.get_colorscale(colorscale)

    def interp_color(t):
        return pc.sample_colorscale(scale, t)[0]

    for n, d in G.nodes(data=True):
        gen = d.get("gen_idx")
        if gen is None:
            continue

        t = gen / span  # Normalize generation into 0..1
        d["color_val"] = t
        d["color"] = interp_color(t)


def apply_node_sizes(
    G: nx.MultiDiGraph,
    config: PlotConfig,
    optimum: Optional[float] = None
) -> None:
    """
    Calculate and assign sizes to all nodes in the graph.

    Node sizes are determined based on node type:
    - LON nodes: sized by sum of incoming edge weights
    - STN_MO nodes: sized by front size
    - STN nodes: sized by iteration count
    - Other nodes: default size of 1

    Args:
        G: NetworkX MultiDiGraph to modify
        config: PlotConfig object with size settings
        optimum: Optional optimum fitness value for highlighting
    """
    STN_node_min = config.node_size.stn_min
    STN_node_max = config.node_size.stn_max
    LON_node_min = config.node_size.lon_min
    LON_node_max = config.node_size.lon_max

    # Normalize solution iterations for STN nodes
    stn_iterations = [
        G.nodes[node].get('iterations', 1)
        for node in G.nodes()
        if "STN" in node and "MO" not in node
    ]
    if stn_iterations:
        min_STN_iter = min(stn_iterations)
        max_STN_iter = max(stn_iterations)
    else:
        min_STN_iter = max_STN_iter = 1

    # Determine front sizes for multiobjective
    front_sizes = [d.get("front_size", 1) for _, d in G.nodes(data=True) if d.get("type") == "STN_MO"]
    if front_sizes:
        min_front_size, max_front_size = min(front_sizes), max(front_sizes)
    else:
        min_front_size = max_front_size = 1

    # Assign node sizes
    for node, data in G.nodes(data=True):
        if "Local Optimum" in node:
            # For LON nodes: weight is the sum of incoming edge weights
            incoming_edges = G.in_edges(node, data=True)
            node_weight = sum(edge_data.get('weight', 0) for _, _, edge_data in incoming_edges)
            node_size = LON_node_min + node_weight * (LON_node_max - LON_node_min)
            G.nodes[node]['weight'] = node_weight

        elif "STN_MO" in data.get("type", ""):
            # Multiobjective STN nodes: use front size to scale node size
            front_size = data.get("front_size", 1)
            # Normalise within observed range across all STN_MO nodes
            node_size = STN_node_min + (
                (front_size - min_front_size) / (max_front_size - min_front_size)
                if max_front_size > min_front_size else 0.5
            ) * (STN_node_max - STN_node_min)
            G.nodes[node]["size"] = node_size
            continue  # skip remaining checks, already set size

        elif "STN" in node:
            # For STN nodes: weight comes from the 'iterations' attribute
            iter_val = G.nodes[node].get('iterations', 1)
            # Normalize to the 0-1 range
            norm_iter = (iter_val - min_STN_iter) / (max_STN_iter - min_STN_iter) if max_STN_iter > min_STN_iter else 0.5
            node_size = STN_node_min + norm_iter * (STN_node_max - STN_node_min)
            if optimum is not None and data.get('fitness') == optimum and "Noisy" not in node:
                node_size = STN_node_max

        else:
            # For any other node, assign a default size of 1
            node_size = 1

        # Set the computed size as a node property
        G.nodes[node]['size'] = node_size


def apply_node_colors(
    G: nx.MultiDiGraph,
    config: PlotConfig,
    opt_feas_map: Optional[Dict] = None,
    neigh_feas_map: Optional[Dict] = None,
    optimum: Optional[float] = None
) -> None:
    """
    Apply colors to nodes based on their type and configuration settings.

    STN nodes are colored based on their role (start/end) or default color.
    LON nodes are colored based on the selected mode (fitness, feasibility, neighbor).

    Args:
        G: NetworkX MultiDiGraph to modify
        config: PlotConfig object with color settings
        opt_feas_map: Optional mapping of solutions to feasibility values
        neigh_feas_map: Optional mapping of solutions to neighbor feasibility proportions
        optimum: Optional optimum fitness value for highlighting
    """
    node_colour_mode = config.lon.node_colour_mode
    opt_feas_map = opt_feas_map or {}
    neigh_feas_map = neigh_feas_map or {}

    # Compute fitness range among LON nodes for 'fitness' mode colouring
    local_optimum_nodes = [node for node in G.nodes() if "Local Optimum" in node]
    if local_optimum_nodes:
        all_fitness = [G.nodes[node]['fitness'] for node in local_optimum_nodes]
        min_fit = min(all_fitness)
        max_fit = max(all_fitness)
    else:
        min_fit = max_fit = 0.0

    # Node colours
    for node, data in G.nodes(data=True):
        if "STN" in node and "Local Optimum" not in node:
            if data.get('start_node', False):
                data['color'] = 'yellow'
            elif data.get('end_node', False):
                data['color'] = 'brown'
            continue

        if "Local Optimum" in node:
            sol_tuple = sol_tuple_ints(data.get('solution', []))

            if node_colour_mode == 'fitness':
                # Continuous colourscale across LON fitness range
                if max_fit > min_fit:
                    norm = (float(data['fitness']) - min_fit) / (max_fit - min_fit)
                else:
                    norm = 0.5
                data['color'] = px.colors.sample_colorscale('plasma', float(np.clip(norm, 0.0, 0.9999)))[0]

            elif node_colour_mode == 'feasible':
                feas = lookup_map(opt_feas_map, sol_tuple)
                data['color'] = ('green' if int(feas) == 1 else 'red') if feas is not None else 'grey'

            elif node_colour_mode == 'neigh':
                p = lookup_map(neigh_feas_map, sol_tuple)
                data['color'] = px.colors.sample_colorscale('RdYlGn', float(np.clip(p, 0.0, 0.9999)))[0] if p is not None else 'grey'

            else:
                data['color'] = 'grey'

            # Keep the "optimum = red" override ONLY when in fitness mode
            if node_colour_mode == 'fitness' and optimum is not None and data.get('fitness') == optimum and "Noisy" not in node:
                data['color'] = 'red'


def style_nodes(
    G: nx.MultiDiGraph,
    config: PlotConfig,
    opt_feas_map: Optional[Dict] = None,
    neigh_feas_map: Optional[Dict] = None
) -> None:
    """
    Apply all node styling (sizes, colors, and generation coloring).

    This is the main entry point for node styling that combines all styling
    operations.

    Args:
        G: NetworkX MultiDiGraph to modify
        config: PlotConfig object with all settings
        opt_feas_map: Optional mapping of solutions to feasibility values
        neigh_feas_map: Optional mapping of solutions to neighbor feasibility proportions
    """
    optimum = config.optimum

    # Apply generation coloring first (for MO mode)
    if config.mo_mode:
        apply_generation_coloring(G)

    # Apply sizes
    apply_node_sizes(G, config, optimum)

    # Apply colors
    apply_node_colors(G, config, opt_feas_map, neigh_feas_map, optimum)

    print('\033[32mNode Sizes and Colours Assigned\033[0m')
