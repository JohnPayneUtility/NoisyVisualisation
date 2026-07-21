def load_problem_KP(filename, verbose=False):
    " Description of function "
    import numpy as np
    import os

    if os.path.exists('instances_01_KP/low-dimensional/'+filename):
        problem_path = 'instances_01_KP/low-dimensional/'+filename
        solution_path = 'instances_01_KP/low-dimensional-optimum/'+filename

    elif os.path.exists('instances_01_KP/large_scale/'+filename):
        problem_path = 'instances_01_KP/large_scale/'+filename
        solution_path = 'instances_01_KP/large_scale-optimum/'+filename

    else:
        raise FileNotFoundError(f"No knapsack instance found for PID: {filename}")

    data = np.loadtxt(problem_path, dtype=int, usecols=(0, 1))
    col_1 = data[:, 0]
    col_2 = data[:, 1]

    n_items = col_1[0]
    capacity = col_2[0]
    values = data[1:, 0]
    weights = data[1:, 1]

    optimal = np.loadtxt(solution_path, dtype=float)

    items_dict = {}
    for i in range(n_items):
        items_dict[i] = (values[i], weights[i])
    
    problem_info = {
        'number of items': n_items,
        'capcity': capacity,
        'optimal': optimal,
        'values': values,
        'weights': weights
        }

    # Print problem information
    if verbose:
        print("number of items:", n_items)
        print("max weight:", capacity)
        print("values:", values)
        print("weights:", weights)
        print("optimal solution:", optimal)

    # Return problem data
    return n_items, capacity, optimal, values, weights, items_dict, problem_info


def get_knapsack_problem_stats(pid):
    """
    Returns a dict of summary statistics for a knapsack PID, or None if the
    PID does not correspond to a known knapsack instance.
    """
    import numpy as np
    import os

    if not (os.path.exists('instances_01_KP/low-dimensional/' + pid)
            or os.path.exists('instances_01_KP/large_scale/' + pid)):
        return None

    n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP(pid)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # The instance's optimum file stores the optimal fitness value directly
    # (not a solution vector), e.g. `295` for f1_l-d_kp_10_269.
    global_optimum = float(optimal)
    correlation = float(np.corrcoef(values, weights)[0, 1]) if n_items > 1 else float('nan')

    return {
        'n_items': int(n_items),
        'capacity': int(capacity),
        'max_value': float(values.max()),
        'min_value': float(values.min()),
        'avg_value': float(values.mean()),
        'std_value': float(values.std()),
        'max_weight': float(weights.max()),
        'min_weight': float(weights.min()),
        'avg_weight': float(weights.mean()),
        'std_weight': float(weights.std()),
        'value_weight_correlation': correlation,
        'global_optimum': global_optimum,
        'sum_values': float(values.sum()),
        'sum_weights': float(weights.sum()),
    }