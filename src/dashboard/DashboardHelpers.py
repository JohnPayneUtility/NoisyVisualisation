"""
Dashboard helper functions.

This module contains utility functions for the dashboard that are NOT plotting functions.
Plotting functions have been moved to src/plotting/performance/.

For backward compatibility, the plot functions are re-exported from the new location.
"""

import numpy as np
import math

# Re-export plot functions from their new location for backward compatibility
from ..plotting.performance import (
    plot2d_line,
    plot2d_box,
    plot2d_line_mo,
    plot2d_box_mo,
)

# Re-export hamming_distance from common for backward compatibility
from ..common import hamming_distance


# ==============================================================================
# Color Utilities
# ==============================================================================

def convert_to_rgba(color, opacity=1.0):
    """Convert a color to RGBA string format."""
    from matplotlib.colors import to_rgba
    rgba = to_rgba(color, alpha=opacity)
    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"


def fitness_to_color(fitness, min_fitness, max_fitness, alpha):
    """
    Convert a fitness value to a color on a blue-to-green gradient.

    Args:
        fitness: The fitness value to convert
        min_fitness: Minimum fitness in the range
        max_fitness: Maximum fitness in the range
        alpha: Opacity value (0-1)

    Returns:
        str: RGBA color string
    """
    # Normalize fitness value to a 0-1 scale
    if max_fitness == min_fitness:
        ratio = 0.5
    else:
        ratio = (fitness - min_fitness) / (max_fitness - min_fitness)
    # Define colors for the extremes
    low_rgb = np.array([0, 0, 255])   # Blue
    high_rgb = np.array([0, 255, 0])  # Green
    rgb = low_rgb + ratio * (high_rgb - low_rgb)
    rgb = rgb.astype(int)
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'


# ==============================================================================
# Run Selection Utilities
# ==============================================================================

def select_top_runs_by_fitness(all_run_trajectories, n_runs_display, optimisation_goal):
    """
    Select the top N runs based on final fitness.

    Args:
        all_run_trajectories: List of trajectory runs
        n_runs_display: Number of runs to select
        optimisation_goal: 'max' or 'min'

    Returns:
        list: Top N runs sorted by fitness
    """
    if optimisation_goal == 'max':
        sorted_runs = sorted(all_run_trajectories,
                            key=lambda run: run[1][-1],
                            reverse=True)
    else:
        sorted_runs = sorted(all_run_trajectories,
                            key=lambda run: run[1][-1],
                            reverse=False)
    top_runs = sorted_runs[:n_runs_display]
    return top_runs


def get_mean_run(all_run_trajectories):
    """Get the run closest to the mean final fitness."""
    final_fitnesses = [run[1][-1] for run in all_run_trajectories]
    mean_final_fitness = np.mean(final_fitnesses)
    closest_run_idx = np.argmin([abs(fitness - mean_final_fitness) for fitness in final_fitnesses])
    return all_run_trajectories[closest_run_idx]


def get_median_run(all_run_trajectories):
    """Get the run closest to the median final fitness."""
    final_fitnesses = [run[1][-1] for run in all_run_trajectories]
    median_final_fitness = np.median(final_fitnesses)
    closest_run_idx = np.argmin([abs(fitness - median_final_fitness) for fitness in final_fitnesses])
    return all_run_trajectories[closest_run_idx]


def determine_optimisation_goal(all_trajectories_list):
    """Determine if optimization is minimization or maximization."""
    first_run = all_trajectories_list[0][0]
    starting_fitness = first_run[1][0]
    ending_fitness = first_run[1][-1]
    return "min" if ending_fitness < starting_fitness else "max"


# ==============================================================================
# Local Optima Network (LON) Utilities
# ==============================================================================

def filter_negative_LO(local_optima):
    """
    Filter out local optima with negative fitness values.

    Args:
        local_optima: Dictionary with local_optima, fitness_values, and edges

    Returns:
        dict: Filtered local optima data
    """
    filtered_nodes = []
    filtered_fitness_values = []

    for opt, fitness in zip(local_optima["local_optima"], local_optima["fitness_values"]):
        if fitness >= 0:
            filtered_nodes.append(opt)
            filtered_fitness_values.append(fitness)

    allowed_nodes = {tuple(opt) for opt in filtered_nodes}

    filtered_edges = {}
    for (source, target), weight in local_optima["edges"].items():
        if tuple(source) in allowed_nodes and tuple(target) in allowed_nodes:
            filtered_edges[(source, target)] = weight

    return {
        "local_optima": filtered_nodes,
        "fitness_values": filtered_fitness_values,
        "edges": filtered_edges
    }


# ==============================================================================
# Data Format Conversion
# ==============================================================================

def convert_to_split_edges_format(data):
    """
    Convert a compressed LON dictionary with `edges` to a format with separate
    `edge_transitions` and `edge_weights`.

    Args:
        data: Original compressed LON data with edges dict

    Returns:
        dict: Modified data with edge_transitions and edge_weights lists
    """
    converted_data = {
        "local_optima": data["local_optima"],
        "fitness_values": data["fitness_values"],
        "edge_transitions": [],
        "edge_weights": [],
    }

    for (source, target), weight in data["edges"].items():
        converted_data["edge_transitions"].append((source, target))
        converted_data["edge_weights"].append(weight)

    return converted_data


def convert_to_single_edges_format(data):
    """
    Convert a compressed LON dictionary with separate `edge_transitions` and
    `edge_weights` to a format with `edges`.

    Args:
        data: Modified compressed LON data with edge_transitions and edge_weights

    Returns:
        dict: Original format with edges dict
    """
    converted_data = {
        "local_optima": data["local_optima"],
        "fitness_values": data["fitness_values"],
        "edges": {},
    }

    for transition, weight in zip(data["edge_transitions"], data["edge_weights"]):
        if isinstance(transition, (list, tuple)) and len(transition) == 2:
            source, target = map(tuple, transition)
            converted_data["edges"][(source, target)] = weight
        else:
            raise ValueError(f"Invalid transition format: {transition}")

    return converted_data


def filter_local_optima(converted_data, fitness_percent):
    """
    Filter local optima to keep only those in the top fitness percentage.

    Args:
        converted_data: Dictionary with local_optima, fitness_values, and edges
        fitness_percent: Percentage (0 < x <= 100) of top fitness values to keep

    Returns:
        dict: Filtered data
    """
    if not (0 < fitness_percent <= 100):
        raise ValueError("fitness_percent must be between 0 and 100.")

    local_optima = converted_data["local_optima"]
    fitness_values = converted_data["fitness_values"]
    edges = converted_data.get("edges", {})

    n = len(local_optima)
    num_to_keep = max(1, math.ceil((fitness_percent / 100) * n))

    pairs = [(i, opt, fit) for i, (opt, fit) in enumerate(zip(local_optima, fitness_values))]
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    top_pairs = pairs_sorted[:num_to_keep]

    kept_indices = {i for i, _, _ in top_pairs}

    new_local_optima = [local_optima[i] for i in range(n) if i in kept_indices]
    new_fitness_values = [fitness_values[i] for i in range(n) if i in kept_indices]

    kept_set = set(tuple(opt) if isinstance(opt, list) else opt for opt in new_local_optima)

    new_edges = {}
    for (source, target), weight in edges.items():
        src = tuple(source) if isinstance(source, list) else source
        tgt = tuple(target) if isinstance(target, list) else target
        if src in kept_set and tgt in kept_set:
            new_edges[(source, target)] = weight

    return {
        "local_optima": new_local_optima,
        "fitness_values": new_fitness_values,
        "edges": new_edges,
    }


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def quadratic_bezier(start, end, curvature=0.2, n_points=20):
    """
    Compute points for a quadratic Bezier curve between start and end.

    Args:
        start: Starting point (x, y)
        end: Ending point (x, y)
        curvature: Fraction of distance to offset midpoint (default: 0.2)
        n_points: Number of points along the curve (default: 20)

    Returns:
        np.array: Array of points along the curve
    """
    start = np.array(start)
    end = np.array(end)
    mid = (start + end) / 2.0

    direction = end - start
    if np.all(direction == 0):
        return np.array([start])

    perp = np.array([-direction[1], direction[0]])
    perp = perp / np.linalg.norm(perp)

    distance = np.linalg.norm(direction)
    control = mid + curvature * distance * perp

    t_values = np.linspace(0, 1, n_points)
    curve_points = []
    for t in t_values:
        point = (1 - t)**2 * start + 2 * (1 - t) * t * control + t**2 * end
        curve_points.append(point)

    return np.array(curve_points)


def should_label_edge(u, v, STN_hamming, LON_hamming):
    """
    Determine if an edge should have a Hamming distance label.

    Args:
        u: Source node name
        v: Target node name
        STN_hamming: Whether to label STN edges
        LON_hamming: Whether to label LON edges

    Returns:
        bool: True if edge should be labeled
    """
    # Noisy edges should never be labeled
    if ("Noisy" in u) or ("Noisy" in v):
        return False

    is_STN = ("STN" in u)
    is_LON = ("Local Optimum" in u) or ("Local Optimum" in v)

    # If edge qualifies as both STN and LON, only label if both enabled
    if is_STN and is_LON:
        return STN_hamming and LON_hamming

    if is_STN:
        return STN_hamming

    if is_LON:
        return LON_hamming

    return True
