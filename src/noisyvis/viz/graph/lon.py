"""LON graph population for the shared display graph.

The Local-Optima-Network half of the former `visualization/graph_builder.py`. These functions
populate the same `nx.MultiDiGraph` the STN population functions in `noisyvis.viz.graph.stn` write
to, after them. Node ids, node attributes and edge attributes written here are contracts read by
`noisyvis.viz.layout`, `noisyvis.viz.styling`, `noisyvis.viz.traces`,
`noisyvis.analysis.graph_stats` and the dashboard's annotation logic.
"""

from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import numpy as np

from ..config import PlotConfig
from ...common import hamming_distance, sol_tuple_ints, lookup_map


def add_lon_nodes(
    G: nx.MultiDiGraph,
    local_optima: Dict,
    lon_node_mapping: Dict[Tuple, str],
    config: PlotConfig,
    problem_id: str,
    fitness_func_params: Optional[Dict] = None
) -> Tuple[Dict[Tuple, str], Dict[str, List[float]]]:
    """
    Add LON (Local Optima Network) nodes to the graph.

    Creates nodes for each local optimum and optionally computes noisy
    fitness samples for each node.

    Args:
        G: NetworkX MultiDiGraph to modify
        local_optima: Dictionary containing local_optima, fitness_values, and edges
        lon_node_mapping: Existing mapping of (solution, type) -> node_label
        config: PlotConfig object with settings
        problem_id: Problem identifier for loading problem data
        fitness_func_params: Optional dict with items_dict, capacity, etc.

    Returns:
        Tuple of (updated lon_node_mapping, node_noise dictionary)
    """
    from ...problems.knapsack import (
        eval_noisy_kp_v1_simple, eval_noisy_kp_v2_simple,
        eval_noisy_kp_v1, eval_noisy_kp_v2, eval_noisy_kp_v3,
        eval_noisy_kp_v1_penalty, eval_noisy_kp_v2_penalty,
        eval_noisy_kp_prior_bitflip, eval_noisy_kp_prior_mult_bitflip,
        eval_noisy_kp_pq_prior_bitwise, eval_noisy_kp_1q_prior_bitwise
    )
    from ...problems.instances import load_problem_KP

    node_noise = {}

    n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP(problem_id)

    for opt, fitness in zip(local_optima["local_optima"], local_optima["fitness_values"]):
        solution_tuple = tuple(opt)
        key = (solution_tuple, "LON")
        if key not in lon_node_mapping:
            node_label = f"Local Optimum {len(lon_node_mapping) + 1}"
            lon_node_mapping[key] = node_label
            G.add_node(node_label, solution=opt, fitness=fitness, type="LON")
        else:
            node_label = lon_node_mapping[key]

        # NOISE BOX PLOTS FOR LON
        node_noise[node_label] = []

        nlon_config = config.noisy_lon
        for i in range(nlon_config.samples):
            # Compute the noisy fitness
            if nlon_config.fit_func == 'kpv1s':
                noisy_fitness = eval_noisy_kp_v1_simple(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv2s':
                noisy_fitness = eval_noisy_kp_v2_simple(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv1mw':
                noisy_fitness = eval_noisy_kp_v1(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv2mw':
                noisy_fitness = eval_noisy_kp_v2(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv3':
                noisy_fitness = eval_noisy_kp_v3(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpv1p':
                noisy_fitness = eval_noisy_kp_v1_penalty(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity, penalty=nlon_config.penalty)[0]
            elif nlon_config.fit_func == 'kpv2p':
                noisy_fitness = eval_noisy_kp_v2_penalty(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity, penalty=nlon_config.penalty)[0]
            elif nlon_config.fit_func == 'kppbf':
                noisy_fitness = eval_noisy_kp_prior_bitflip(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kppmbf':
                noisy_fitness = eval_noisy_kp_prior_mult_bitflip(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpppqbw':
                noisy_fitness = eval_noisy_kp_pq_prior_bitwise(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity)[0]
            elif nlon_config.fit_func == 'kpp1qbw':
                noisy_fitness = eval_noisy_kp_1q_prior_bitwise(opt, items_dict=items_dict, capacity=capacity, noise_intensity=nlon_config.intensity, penalty=nlon_config.penalty)[0]
            else:
                print('NO NOISY FITNESS FUNCTION PROVIDED')
                noisy_fitness = 0
            node_noise[node_label].append(noisy_fitness)

    return lon_node_mapping, node_noise


def add_lon_edges(
    G: nx.MultiDiGraph,
    local_optima: Dict,
    lon_node_mapping: Dict[Tuple, str],
    config: PlotConfig,
    opt_feas_map: Optional[Dict] = None
) -> None:
    """
    Add LON edges to the graph.

    Creates edges between local optima nodes based on the transition data,
    with optional coloring based on feasibility.

    Args:
        G: NetworkX MultiDiGraph to modify
        local_optima: Dictionary containing edges data
        lon_node_mapping: Mapping of (solution, type) -> node_label
        config: PlotConfig object with settings
        opt_feas_map: Optional feasibility map for edge coloring
    """
    import plotly.express as px
    # lookup_map imported at module level from common

    colour_edges_by_feas = config.lon.edge_colour_feas

    for (source, target), weight in local_optima["edges"].items():
        source_tuple = tuple(source)
        target_tuple = tuple(target)
        src_key = (source_tuple, "LON")
        tgt_key = (target_tuple, "LON")
        if src_key in lon_node_mapping and tgt_key in lon_node_mapping:
            src_label = lon_node_mapping[src_key]
            tgt_label = lon_node_mapping[tgt_key]

            edge_color = 'black'  # default
            if colour_edges_by_feas and opt_feas_map:
                tgt_sol = G.nodes[tgt_label].get('solution', [])
                feas = lookup_map(opt_feas_map, tgt_sol)
                if feas is not None:
                    edge_color = 'green' if int(feas) == 1 else 'red'

            G.add_edge(src_label, tgt_label, weight=weight, color=edge_color, edge_type='LON')

    # ONLY recolor by weight if we're NOT colouring by feasibility
    if not colour_edges_by_feas:
        # Calculate min and max edge weight for LON for normalisation
        LON_edge_weight_all = [
            data.get('weight', 2)
            for u, v, key, data in G.edges(data=True, keys=True)
            if "Local Optimum" in u and "Local Optimum" in v
        ]
        if LON_edge_weight_all:
            LON_edge_weight_min = min(LON_edge_weight_all)
            LON_edge_weight_max = max(LON_edge_weight_all)
        else:
            LON_edge_weight_min = LON_edge_weight_max = 1

        # Normalise edge weights for edges between Local Optimum nodes and colour
        for u, v, key, data in G.edges(data=True, keys=True):
            if "Local Optimum" in u and "Local Optimum" in v:
                weight = data.get('weight', 2)
                # Normalize the weight (if all weights are equal, default to 0.5)
                norm_weight = (weight - LON_edge_weight_min) / (LON_edge_weight_max - LON_edge_weight_min) if LON_edge_weight_max > LON_edge_weight_min else 0.5
                norm_weight = np.clip(norm_weight, 0, 0.9999)
                color = px.colors.sample_colorscale('plasma', norm_weight)[0]
                data['norm_weight'] = norm_weight
                data['color'] = color
