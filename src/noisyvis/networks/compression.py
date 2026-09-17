"""LON compression: merge local optima whose fitness values lie within an accuracy threshold (plan Stage 8).

Moved verbatim from `algorithms/LONs.py`. Uses builtins only.
"""

# ----------------------------------------------------
# LON Compression
# ----------------------------------------------------
def compress_lon_aggregated(LON_data, accuracy=1e-5):
    """
    Compress a Landscape of Optima Network (LON) by combining local optima with
    fitness values within a given accuracy threshold, aggregating edges accordingly.

    Args:
        LON_data (dict): Dictionary containing LON data with keys:
            - "local_optima": List of unique local optima (each a tuple).
            - "fitness_values": List of fitness values (float or int) corresponding to the local optima.
            - "edges": Dictionary with keys as (source, target) tuples (each source/target a tuple),
                    and values as numeric edge weights.
        accuracy (float): Threshold for grouping local optima with close fitness values.

    Returns:
        dict: A new LON dictionary with the same format and types, but aggregated
            according to the provided accuracy.
    """

    # 1. Group local optima based on fitness similarity-
    grouped_optima = []          # Representative local optima for each group
    grouped_fitness_values = []  # Representative fitness for each group
    membership = []              # For each original local optimum, which group does it belong to?

    for opt, fit in zip(LON_data["local_optima"], LON_data["fitness_values"]):
        assigned_group = None
        # Check if this fitness is close enough to a group representative's fitness
        for g_idx, g_fit in enumerate(grouped_fitness_values):
            if abs(fit - g_fit) <= accuracy:
                assigned_group = g_idx
                break

        # If not found in any group, create a new group
        if assigned_group is None:
            grouped_optima.append(opt)
            grouped_fitness_values.append(fit)
            assigned_group = len(grouped_optima) - 1

        membership.append(assigned_group)

    # 2. Build a new edge dictionary based on the groups
    # map each original local optimum to an index to easily find its group
    opt_to_index = {opt: i for i, opt in enumerate(LON_data["local_optima"])}

    new_edges = {}
    for (source, target), weight in LON_data["edges"].items():
        # Identify the groups of the source and target
        source_group = membership[opt_to_index[source]]
        target_group = membership[opt_to_index[target]]

        # The new source/target in the aggregated LON
        new_source = grouped_optima[source_group]
        new_target = grouped_optima[target_group]

        # Aggregate edge weights if the same group-pair already exists
        if (new_source, new_target) not in new_edges:
            new_edges[(new_source, new_target)] = weight
        else:
            new_edges[(new_source, new_target)] += weight

    # 3. Construct the new, aggregated LON data structure
    compressed_lon_data = {
        "local_optima": grouped_optima,
        "fitness_values": grouped_fitness_values,
        "edges": new_edges,
    }

    # 4. Validate output to ensure it matches required format
    assert isinstance(compressed_lon_data, dict), "Output must be a dictionary."
    assert "local_optima" in compressed_lon_data, "Output dictionary must contain 'local_optima'."
    assert "fitness_values" in compressed_lon_data, "Output dictionary must contain 'fitness_values'."
    assert "edges" in compressed_lon_data, "Output dictionary must contain 'edges'."
    assert all(isinstance(opt, tuple) for opt in compressed_lon_data["local_optima"]), \
        "All local optima in 'local_optima' must be tuples."
    assert all(isinstance(f, (float, int)) for f in compressed_lon_data["fitness_values"]), \
        "All values in 'fitness_values' must be numeric."
    assert all(
        isinstance(k, tuple) and len(k) == 2 
        and isinstance(k[0], tuple) and isinstance(k[1], tuple)
        for k in compressed_lon_data["edges"].keys()
    ), "All edge keys must be 2-tuples of solutions (which are tuples)."
    assert all(isinstance(v, (float, int)) for v in compressed_lon_data["edges"].values()), \
        "All edge weights must be numeric."

    return compressed_lon_data
