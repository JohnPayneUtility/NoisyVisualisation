"""
Statistics calculation functions for the visualization module.

This module contains functions for calculating various statistics
about the graph, particularly for Local Optima Networks (LON).
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import networkx as nx


@dataclass
class LONStatistics:
    """Statistics for a Local Optima Network."""
    num_local_optima: int = 0
    mean_fitness: float = 0.0
    max_fitness: float = 0.0
    mean_weight: float = 0.0
    max_weight: float = 0.0
    mean_in_degree: float = 0.0
    mean_out_degree: float = 0.0
    num_edges: int = 0
    mean_edge_weight: float = 0.0
    max_edge_weight: float = 0.0


def calculate_lon_statistics(G: nx.MultiDiGraph, verbose: bool = True) -> LONStatistics:
    """
    Calculate statistics for the Local Optima Network portion of the graph.

    Args:
        G: NetworkX MultiDiGraph containing LON nodes
        verbose: If True, print statistics to console

    Returns:
        LONStatistics dataclass with calculated values
    """
    stats = LONStatistics()

    # Get all local optimum nodes
    local_optimum_nodes = [node for node in G.nodes() if "Local Optimum" in node]

    if not local_optimum_nodes:
        if verbose:
            print('No local optima found in graph')
        return stats

    stats.num_local_optima = len(local_optimum_nodes)

    # Calculate degree statistics
    stats.mean_in_degree = sum(G.in_degree(node) for node in local_optimum_nodes) / stats.num_local_optima
    stats.mean_out_degree = sum(G.out_degree(node) for node in local_optimum_nodes) / stats.num_local_optima

    # Calculate weight statistics
    stats.mean_weight = (
        sum(G.nodes[node].get('weight', 0) for node in local_optimum_nodes) / stats.num_local_optima
    )
    stats.max_weight = max(G.nodes[node].get('weight', 0) for node in local_optimum_nodes)

    # Calculate fitness statistics
    stats.mean_fitness = (
        sum(G.nodes[node].get('fitness', 0) for node in local_optimum_nodes) / stats.num_local_optima
    )
    stats.max_fitness = max(G.nodes[node].get('fitness', 0) for node in local_optimum_nodes)

    # Edge statistics
    local_optimum_edges = [
        (u, v) for u, v, data in G.edges(data=True)
        if "Local Optimum" in u and "Local Optimum" in v
    ]
    local_optimum_edge_weights = [
        data.get('weight', 0)
        for u, v, data in G.edges(data=True)
        if "Local Optimum" in u and "Local Optimum" in v
    ]

    stats.num_edges = len(local_optimum_edges)

    if local_optimum_edge_weights:
        stats.max_edge_weight = max(local_optimum_edge_weights)
        stats.mean_edge_weight = sum(local_optimum_edge_weights) / len(local_optimum_edge_weights)

    if verbose:
        print('LOCAL OPTIMA STATS CALCULATED')
        print('num_local_optima', stats.num_local_optima)
        print('max_fitness', stats.max_fitness)
        print(f'mean_fitness: {stats.mean_fitness:.2f}')
        print(f'mean_weight: {stats.mean_weight:.2f}')
        print(f'max_weight: {stats.max_weight:.2f}')
        print(f'mean_in_degree_local: {stats.mean_in_degree:.2f}')
        print(f'mean_out_degree_local: {stats.mean_out_degree:.2f}')
        print("num_edges:", stats.num_edges)
        print(f'max_edge_weight: {stats.max_edge_weight:.2f}')
        print(f'mean_edge_weight: {stats.mean_edge_weight:.2f}')

    return stats


def calculate_stn_statistics(G: nx.MultiDiGraph, verbose: bool = True) -> Dict[str, Any]:
    """
    Calculate statistics for the Search Trajectory Network portion of the graph.

    Args:
        G: NetworkX MultiDiGraph containing STN nodes
        verbose: If True, print statistics to console

    Returns:
        Dictionary with STN statistics
    """
    stn_nodes = [node for node in G.nodes() if "STN" in node and "Local Optimum" not in node and "MO" not in node]

    if not stn_nodes:
        if verbose:
            print('No STN nodes found in graph')
        return {}

    stats = {
        'num_stn_nodes': len(stn_nodes),
        'num_unique_solutions': len(set(
            tuple(G.nodes[node].get('solution', []))
            for node in stn_nodes
            if G.nodes[node].get('solution')
        )),
    }

    # Fitness statistics
    fitnesses = [G.nodes[node].get('fitness', 0) for node in stn_nodes if G.nodes[node].get('fitness') is not None]
    if fitnesses:
        stats['min_fitness'] = min(fitnesses)
        stats['max_fitness'] = max(fitnesses)
        stats['mean_fitness'] = sum(fitnesses) / len(fitnesses)

    # Iteration statistics
    iterations = [G.nodes[node].get('iterations', 1) for node in stn_nodes]
    if iterations:
        stats['min_iterations'] = min(iterations)
        stats['max_iterations'] = max(iterations)
        stats['mean_iterations'] = sum(iterations) / len(iterations)

    if verbose:
        print('STN STATS CALCULATED')
        for key, value in stats.items():
            if isinstance(value, float):
                print(f'{key}: {value:.2f}')
            else:
                print(f'{key}: {value}')

    return stats


def calculate_graph_summary(G: nx.MultiDiGraph, verbose: bool = True) -> Dict[str, Any]:
    """
    Calculate summary statistics for the entire graph.

    Args:
        G: NetworkX MultiDiGraph to analyze
        verbose: If True, print summary to console

    Returns:
        Dictionary with graph summary statistics
    """
    summary = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
    }

    # Count by type
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    summary['nodes_by_type'] = node_types

    # Count edges by type
    edge_types = {}
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    summary['edges_by_type'] = edge_types

    if verbose:
        print(f"Graph Summary: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
        print(f"  Nodes by type: {node_types}")
        print(f"  Edges by type: {edge_types}")

    return summary
