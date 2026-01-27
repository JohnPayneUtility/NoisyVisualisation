# IMPORTS
import numpy as np
import random
from ..algorithms.LONs import random_bit_flip
from ..algorithms.Logger import get_active_logger

# ==============================

def mean_weight(items_dict):
    total_weight = sum(weight for _, weight in items_dict.values())
    mean_weight = total_weight / len(items_dict)
    return mean_weight

# ==============================
# Combinatorial Fitness Functions
# ==============================

def OneMax_fitness(individual, noise_function=None, noise_intensity=0):
    """ Function calculates fitness for OneMax problem individual """
    if noise_function is not None: # Provide noise function for noise applied to individual
        individual = noise_function(individual[:], noise_intensity)
        fitness = sum(individual)
    else: # standard noisy
        fitness = sum(individual) + random.gauss(0, noise_intensity)
    return (fitness,)

def jump_fitness(individual, gap_size, noise_intensity):
    """ Calculates fitness for jump problem """
    # print(individual)
    n = len(individual)
    ones = sum(individual)
    
    if ones == n or ones <= n-gap_size:
        return (ones,)
    else:
        return (n - ones + gap_size,)

def eval_ind_kp(individual, items_dict, capacity, penalty=1):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    
    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

def eval_noisy_kp_v1_simple(individual, items_dict, capacity, noise_intensity=0, penalty=1):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    
    noise = random.gauss(0, noise_intensity)
    value = value + noise

    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

def eval_noisy_kp_v2_simple(individual, items_dict, capacity, noise_intensity=0, penalty=1):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    
    noise = random.gauss(0, noise_intensity)
    value = value + noise

    # Check if over capacity and return reduced value
    if (weight + noise) > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

def eval_noisy_kp_v1(individual, items_dict, capacity, noise_intensity=0, penalty=1):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    
    noise = random.gauss(0, noise_intensity * mean_weight(items_dict))
    value = value + noise

    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

def eval_noisy_kp_v2_backup(individual, items_dict, capacity, noise_intensity=0, penalty=1):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    
    noise = random.gauss(0, noise_intensity * mean_weight(items_dict))
    value = value + noise

    # Check if over capacity and return reduced value
    if (weight + noise) > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

def eval_noisy_kp_v2(individual, items_dict, capacity, noise_intensity=0, penalty=1):
    """
    Calculates fitness for knapsack problem with posterior noise.

    Noise is added to the fitness value after evaluation (not to the solution).
    If a logger is active, the evaluation is recorded.
    Note: For posterior noise, original and noisy individual are the same,
    but true_fitness and noisy_fitness differ.
    """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items))
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items))

    # Calculate true fitness (no noise)
    if weight > capacity:
        if penalty == 1:
            true_fitness = capacity - weight
        else:
            true_fitness = 0
    else:
        true_fitness = value

    # Calculate noisy fitness
    noise = random.gauss(0, noise_intensity * mean_weight(items_dict))
    noisy_value = value + noise

    if (weight + noise) > capacity:
        if penalty == 1:
            noisy_fitness = capacity - weight
        else:
            noisy_fitness = 0
    else:
        noisy_fitness = noisy_value

    # Log the evaluation if logger is active
    # For posterior noise: same solution, different fitnesses
    logger = get_active_logger()
    if logger:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_prior(individual, items_dict, capacity, noise_intensity=0, penalty=1):
    """
    Calculates fitness for knapsack problem with prior noise.

    The individual is perturbed (bits flipped) before evaluation.
    If a logger is active, the original and noisy solutions are recorded.
    Note: For prior noise, original and noisy individual differ,
    true_fitness is for original, noisy_fitness is for perturbed.
    """
    n_items = len(individual)

    # Calculate true fitness of original (unperturbed) solution
    orig_weight = sum(items_dict[i][1] * individual[i] for i in range(n_items))
    orig_value = sum(items_dict[i][0] * individual[i] for i in range(n_items))

    if orig_weight > capacity:
        if penalty == 1:
            true_fitness = capacity - orig_weight
        else:
            true_fitness = 0
    else:
        true_fitness = orig_value

    # Create noisy (perturbed) solution and calculate its fitness
    noisy_individual, _ = random_bit_flip(list(individual), n_flips=noise_intensity)
    noisy_weight = sum(items_dict[i][1] * noisy_individual[i] for i in range(n_items))
    noisy_value = sum(items_dict[i][0] * noisy_individual[i] for i in range(n_items))

    if noisy_weight > capacity:
        if penalty == 1:
            noisy_fitness = capacity - noisy_weight
        else:
            noisy_fitness = 0
    else:
        noisy_fitness = noisy_value

    # Log the evaluation if logger is active
    # For prior noise: different solutions, with their respective fitnesses
    logger = get_active_logger()
    if logger:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

# ==============================
# Continuous Fitness Functions
# ==============================

def rastrigin_eval(individual, amplitude=10, noise_intensity=0):
    A = amplitude
    n = len(individual)
    fitness = A * n + sum((x ** 2 - A * np.cos(2 * np.pi * x)) for x in individual) + random.gauss(0, noise_intensity),
    return fitness

def birastrigin_eval(individual, d=1, s=None):
    """
    Fitness evaluation for the Birastrigin problem

    Args:
        individual (list or np.ndarray): The input vector representing an individual.
        d (float, optional): Parameter `d`, standardized to 1 unless specified otherwise.
        s (float, optional): Parameter `s`, if not provided, it is calculated as per the formula.

    Returns:
        tuple: A single-element tuple containing the fitness value.
    """
    # Define parameters
    mu1 = 2.5
    if s is None:
        s = 1 - (1 / (2 * np.sqrt(2) + 20 - 8.2))
    mu2 = -np.sqrt(mu1**2 - d / s)

    n = len(individual)

    # Compute the two components of the fitness function
    term1 = sum((x - mu1)**2 for x in individual)
    term2 = d * n + s * sum((x - mu2)**2 for x in individual)
    term3 = 10 * sum(1 - np.cos(2 * np.pi * (x - mu1)) for x in individual)

    # Final fitness calculation
    fitness = min(term1, term2) + term3

    return fitness

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    Compute the Ackley function value for a given input vector x.
    
    :param x: List or NumPy array of input values.
    :param a: Parameter controlling the function's steepness (default 20).
    :param b: Parameter controlling the exponential term (default 0.2).
    :param c: Parameter controlling the cosine term (default 2π).
    :return: Ackley function value.
    """
    x = np.array(x)
    d = len(x)
    
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    
    return term1 + term2 + a + np.exp(1)

