"""
Geometry helpers for graph rendering.

Moved unchanged from dashboard/DashboardHelpers.py (reorganisation Stage 5) so that the
visualization layer no longer imports the dashboard package.
"""

import numpy as np


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

    is_STN = ("STN" in u) or u.startswith("MO_")
    is_LON = ("Local Optimum" in u) or ("Local Optimum" in v)

    # If edge qualifies as both STN and LON, only label if both enabled
    if is_STN and is_LON:
        return STN_hamming and LON_hamming

    if is_STN:
        return STN_hamming

    if is_LON:
        return LON_hamming

    return True
