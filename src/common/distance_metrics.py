"""
Distance metric functions shared across visualization and plotting modules.

This module contains pure functions for calculating various distance metrics
between solutions, fronts, and other data structures. These functions have
no dependencies on graph or plotting state and can be used independently.
"""

from typing import List, Tuple, Union

# Type aliases for clarity
Solution = Union[Tuple[int, ...], List[int]]
Front = List[Solution]


def hamming_distance(sol1: Solution, sol2: Solution) -> int:
    """
    Calculate the Hamming distance between two solutions.

    The Hamming distance is the number of positions at which the
    corresponding elements are different.

    Args:
        sol1: First solution (list or tuple of values)
        sol2: Second solution (list or tuple of values)

    Returns:
        The number of differing positions between the two solutions.
    """
    return sum(el1 != el2 for el1, el2 in zip(sol1, sol2))


def normed_hamming_distance(sol1: Solution, sol2: Solution) -> float:
    """
    Calculate the normalized Hamming distance between two solutions.

    The normalized distance is the Hamming distance divided by the
    length of the solutions, resulting in a value between 0 and 1.

    Args:
        sol1: First solution (list or tuple of values)
        sol2: Second solution (list or tuple of values)

    Returns:
        The normalized Hamming distance (0.0 to 1.0).
    """
    L = len(sol1)
    if L == 0:
        return 0.0
    return sum(el1 != el2 for el1, el2 in zip(sol1, sol2)) / L


def sol_tuple_ints(sol: Solution) -> Tuple[int, ...]:
    """
    Convert any iterable solution into a tuple of integers.

    This ensures consistent representation of solutions regardless
    of whether they were provided as lists, tuples, or with different
    numeric types. Used for creating consistent keys in lookup dictionaries
    and for comparing solutions.

    Args:
        sol: Solution to convert

    Returns:
        A tuple of integers representing the solution.
    """
    return tuple(int(x) for x in sol)


def avg_min_hamming_A_to_B(front_a: Front, front_b: Front) -> float:
    """
    Calculate the asymmetric average minimum Hamming distance from A to B.

    For each solution in front A, finds the minimum Hamming distance to any
    solution in front B, then returns the average of these minimum distances.
    The result is normalized by solution length.

    Args:
        front_a: Source front (list of solutions)
        front_b: Target front (list of solutions)

    Returns:
        The average minimum normalized Hamming distance from A to B.
    """
    if not front_a or not front_b:
        return 0.0

    A = [sol_tuple_ints(a) for a in front_a]
    B = [sol_tuple_ints(b) for b in front_b]

    L = len(A[0]) if A else 1
    norm = float(L) if L > 0 else 1.0

    total = 0.0
    for a in A:
        # Minimum Hamming(a, b) over b in B
        min_dist = min(sum(aa != bb for aa, bb in zip(a, b)) for b in B) / norm
        total += min_dist

    return total / len(A)


def front_distance(front_a: Front, front_b: Front) -> float:
    """
    Calculate the symmetric average-min Hamming distance between two fronts.

    This is the mean of the asymmetric distances in both directions,
    providing a symmetric measure of dissimilarity between two fronts.

    Args:
        front_a: First front (list of solutions)
        front_b: Second front (list of solutions)

    Returns:
        The symmetric front distance.
    """
    d1 = avg_min_hamming_A_to_B(front_a, front_b)
    d2 = avg_min_hamming_A_to_B(front_b, front_a)
    return 0.5 * (d1 + d2)


def sol_key_str(sol: Solution) -> str:
    """
    Convert a solution to a string key format "1,0,1,...".

    This format is used in some data stores for solution lookup.

    Args:
        sol: Solution to convert

    Returns:
        A comma-separated string of the solution values.
    """
    return ",".join(str(int(x)) for x in sol)


def lookup_map(mapp: dict, sol: Solution):
    """
    Look up a solution in a map that may use tuple or string keys.

    Tries both the tuple form and the string form of the solution
    as keys, returning the first match found.

    Args:
        mapp: Dictionary to search in
        sol: Solution to look up

    Returns:
        The value from the map, or None if not found.
    """
    if not isinstance(mapp, dict):
        return None

    t = sol_tuple_ints(sol)
    s = sol_key_str(sol)

    if t in mapp:
        return mapp[t]
    if s in mapp:
        return mapp[s]

    return None
