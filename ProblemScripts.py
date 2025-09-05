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

    else: print('problem not found')

    data = np.loadtxt(problem_path, dtype=int, usecols=(0, 1))
    col_1 = data[:, 0]
    col_2 = data[:, 1]

    n_items = col_1[0]
    capacity = col_2[0]
    values = data[1:, 0]
    weights = data[1:, 1]

    optimal = np.loadtxt(solution_path, dtype=int)

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