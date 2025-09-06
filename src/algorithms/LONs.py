import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from ExperimentsHelpers import *

from tqdm import trange

import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# Custom Mutation Functions
# ----------------------------------------------------

# Random bit flip for Binary LON
def random_bit_flip(bit_list, n_flips=1, exclude_indices=None):
    # test_random_seed()
    # Ensure n_flips does not exceed the length of bit_list
    n_flips = min(n_flips, len(bit_list))
    
    flipped_indices = set()
    if exclude_indices:
        flipped_indices.update(exclude_indices)
    if len(flipped_indices) == len(bit_list):
            return bit_list, flipped_indices

    for _ in range(n_flips):
        # Select a unique random index to flip
        index_to_flip = random.randint(0, len(bit_list) - 1)
        
        while index_to_flip in flipped_indices:
            index_to_flip = random.randint(0, len(bit_list) - 1)
        
        bit_list[index_to_flip] = 1 - bit_list[index_to_flip] # bit flip
        
        # Record the flipped index
        flipped_indices.add(index_to_flip)
        if len(flipped_indices) == len(bit_list):
            return bit_list, flipped_indices
    
    return bit_list, flipped_indices

# ----------------------------------------------------
# Binary LON
# ----------------------------------------------------

def BinaryLON(pert_attempts, len_sol, weights,
              attr_function=None,
              n_flips_mut=1,
              n_flips_pert=2,
              mutate_function=None,
              perturb_function=None,
              improv_method='best',
              fitness_function=None,
              starting_solution=None,
              true_fitness_function=None,
              target_stop = None):
    """
    """

    # 1) Create Fitness and Individual classes if not existing
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)

    # 2) Helper function to generate n_flips bit-flips
    def generate_bit_flip_combinations(ind, n_flips):
        import itertools
        indices = range(len(ind))
        combinations = itertools.combinations(indices, n_flips)
        mutants = []
        for combo in combinations:
            mutant = toolbox.clone(ind)
            for index in combo:
                mutant[index] = 1 - mutant[index]
            mutants.append(mutant)
        return mutants

    # 3) Define Toolbox
    toolbox = base.Toolbox()
    toolbox.register("attribute", attr_function)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len_sol)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: fitness_function[0](ind, **fitness_function[1]))
    toolbox.register("mutate", lambda ind: mutate_function[0](ind, **mutate_function[1]))
    toolbox.register("perturb", lambda ind: perturb_function[0](ind, **perturb_function[1]))

    # 4) Create and evaluate a starting solution
    if starting_solution is not None:
        individual = creator.Individual(starting_solution)
    else:
        individual = creator.Individual([toolbox.attribute() for _ in range(len_sol)])
    individual.fitness.values = toolbox.evaluate(individual)

    # 5) Initialize data recording
    local_optima = []
    fitness_values = []
    edges = {}
    prev_local_opt = None
    prev_fitness = None

    # 6) Run basin-hopping
    mut_attempts = 15000
    pert_attempt = 0

    while pert_attempt < pert_attempts:
        if improv_method == 'best':
            # Best-improvement local search
            improvement = True
            while improvement:
                # Generate all bit-flip neighbors of size n_flips_mut
                mutants = generate_bit_flip_combinations(individual, n_flips_mut)
                # Evaluate them
                for mutant in mutants:
                    del mutant.fitness.values
                    mutant.fitness.values = toolbox.evaluate(mutant)
                
                # Get the best mutant from among all neighbors + current solution
                best_mutant = tools.selBest(mutants + [individual], 1)[0]
                
                # Check if best_mutant is strictly better than the current individual
                if best_mutant.fitness > individual.fitness:
                    # Move to new best solution
                    individual[:] = best_mutant
                    del individual.fitness.values
                    individual.fitness.values = toolbox.evaluate(individual)
                else:
                    # No strictly better neighbor found => we are at a local optimum
                    improvement = False
            
            current_local_opt = tuple(individual)
            current_fitness = individual.fitness.values[0]

            if prev_fitness is not None and current_fitness < prev_fitness:
                print("Alert: Non-monotonic transition detected. Previous fitness:", prev_fitness, "Current fitness:", current_fitness)
            
            # Now, after local search, record the local optimum
            current_local_opt = tuple(individual)
            if current_local_opt not in local_optima:
                local_optima.append(current_local_opt)
                fitness_values.append(individual.fitness.values[0])
            
            # Record an edge from the previous local optimum to the current one
            if prev_local_opt is not None and prev_local_opt != current_local_opt:
                edges[(prev_local_opt, current_local_opt)] = edges.get((prev_local_opt, current_local_opt), 0) + 1
            
            # Update the previous optimum for the next iteration
            prev_local_opt = current_local_opt
            prev_fitness = current_fitness

            # Early stopping condition
            if target_stop is not None and current_fitness >= target_stop:
                logging.info(f"Target fitness reached: {current_fitness} >= {target_stop}")
                break # exit loop
            
            # Perturbation step: try to escape the basin of attraction
            pert_attempt += 1
            perturbed = toolbox.clone(individual)
            perturbed[:], _ = random_bit_flip(perturbed, n_flips=n_flips_pert, exclude_indices=None)
            del perturbed.fitness.values
            perturbed.fitness.values = toolbox.evaluate(perturbed)
            if pert_attempt % 100 == 0:
                logging.info(f"PertAttmpt: {pert_attempt}")
            
            # If the perturbed solution is better, switch to it and reset the perturbation counter
            tolerance = 1e-6
            if perturbed.fitness.values[0] > individual.fitness.values[0] + tolerance:
                individual[:] = perturbed
                del individual.fitness.values
                individual.fitness.values = toolbox.evaluate(individual)
                pert_attempt = 0
                logging.info(f"Current fitness: {individual.fitness.values[0]:.6f}, Perturbed: {perturbed.fitness.values[0]:.6f}")
                logging.info(f"resetting PertAttmpts: {pert_attempt}")
                

        elif improv_method == 'first':
            # Implement first improvement if needed
            pass

    # 7) Convert edges dict to a list
    edges_list = [(source, target, weight) for (source, target), weight in edges.items()]

    return local_optima, fitness_values, edges_list

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

# ----------------------------------------------------
# RUN FUNCTION
# ----------------------------------------------------

def create_binary_LON(prob_info,
                      base_params,
                      n_flips_mut,
                      n_flips_pert,
                      pert_attempts,
                      fitness_function,
                      n_runs,
                      compression_accs = [None],
                      ):
    n_items = base_params['sol_length']
    fit_weights = base_params['opt_weights']
    binary_attribute = base_params['attr_function']
    target_stop = base_params['target_stop']
    base_seed = 0

    aggregated_lon_data = {
        "local_optima": [],
        "fitness_values": [],
        "edges": {},
        }
    
    for i in trange(n_runs, desc="Computing LON"):
        seed = base_seed + i
        random.seed(seed)
        np.random.seed(seed)
        local_optima, fitness_values, edges_list = BinaryLON(pert_attempts, 
                                                            n_items, 
                                                            fit_weights, 
                                                            attr_function=binary_attribute, 
                                                            n_flips_mut=n_flips_mut, 
                                                            n_flips_pert=n_flips_pert,
                                                            mutate_function=None, 
                                                            perturb_function=None, 
                                                            improv_method='best', 
                                                            fitness_function=fitness_function, 
                                                            starting_solution=None, 
                                                            true_fitness_function=None,
                                                            target_stop = target_stop)
        # Aggregate optima
        for opt, fitness in zip(local_optima, fitness_values):
            if opt not in aggregated_lon_data["local_optima"]:
                aggregated_lon_data["local_optima"].append(opt)
                aggregated_lon_data["fitness_values"].append(fitness)

        # Aggregate edges
        for (source, target, weight) in edges_list:
            edge = (source, target)
            if edge in aggregated_lon_data["edges"]:
                aggregated_lon_data["edges"][edge] += weight  # Increment edge weight
            else:
                aggregated_lon_data["edges"][edge] = weight  # Add new edge with its weight
    
    # aggregated_lon_data_SE = convert_to_split_edges_format(aggregated_lon_data)
    # compressed_lon_data_SE = convert_to_split_edges_format(compressed_lon_data)
    for comp_acc in compression_accs:
        if comp_acc == 'None':
            LON_data = aggregated_lon_data
        else:
            compressed_lon_data = compress_lon_aggregated(aggregated_lon_data, accuracy=comp_acc)
            LON_data = compressed_lon_data

        LON_results = {
            "problem_name": prob_info['name'],
            "problem_type": prob_info['type'],
            "problem_goal": prob_info['goal'],
            "dimensions": prob_info['dimensions'],
            "opt_global": prob_info['opt_global'],
            'PID': prob_info['PID'],
            'LON_Algo': 'IteratedLocalSearch',
            'n_flips_mut': n_flips_mut,
            'n_flips_pert': n_flips_pert,
            'compression_val': comp_acc,
            'n_local_optima': len(LON_data["local_optima"]),
            'local_optima': LON_data['local_optima'],
            'fitness_values': LON_data['fitness_values'],
            'edges': LON_data['edges']
        }
        LON_results_df = pd.DataFrame([LON_results])
        save_or_append_results(df = LON_results_df, filename = 'results_LON.pkl')


