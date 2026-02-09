"""
Common utilities shared across visualization and plotting modules.

This package contains pure functions and utilities that are used by both
the 2D plotting module (src/plotting) and the 3D visualization module
(src/visualization).
"""

from .distance_metrics import (
    # Type aliases
    Solution,
    Front,

    # Distance functions
    hamming_distance,
    normed_hamming_distance,
    euclidean_distance,
    avg_min_hamming_A_to_B,
    front_distance,

    # Solution utilities
    sol_tuple_ints,
    sol_tuple_floats,
    is_continuous_solution,
    sol_key_str,
    lookup_map,
)

__all__ = [
    # Type aliases
    'Solution',
    'Front',

    # Distance functions
    'hamming_distance',
    'normed_hamming_distance',
    'euclidean_distance',
    'avg_min_hamming_A_to_B',
    'front_distance',

    # Solution utilities
    'sol_tuple_ints',
    'sol_tuple_floats',
    'is_continuous_solution',
    'sol_key_str',
    'lookup_map',
]
