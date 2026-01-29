"""
Distance metric functions - Re-exported from common module.

DEPRECATED: Import from src.common instead.
This module is kept for backward compatibility.
All distance metric functions have been moved to src/common/distance_metrics.py
to allow sharing between visualization (3D) and plotting (2D) modules.
"""

# Re-export everything from the common module for backward compatibility
from ..common.distance_metrics import (
    # Type aliases
    Solution,
    Front,
    # Functions
    hamming_distance,
    normed_hamming_distance,
    sol_tuple_ints,
    avg_min_hamming_A_to_B,
    front_distance,
    sol_key_str,
    lookup_map,
)

__all__ = [
    'Solution',
    'Front',
    'hamming_distance',
    'normed_hamming_distance',
    'sol_tuple_ints',
    'avg_min_hamming_A_to_B',
    'front_distance',
    'sol_key_str',
    'lookup_map',
]
