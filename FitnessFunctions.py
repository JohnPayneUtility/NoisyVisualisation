# IMPORTS
import numpy as np
import random
from LONs import random_bit_flip

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

def eval_noisy_kp_v2(individual, items_dict, capacity, noise_intensity=0, penalty=1):
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

def eval_noisy_kp_prior(individual, items_dict, capacity, noise_intensity=0, penalty=1, return_sol=False):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    noisy_individual = random_bit_flip(individual, n_flips=noise_intensity)
    weight = sum(items_dict[i][1] * noisy_individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * noisy_individual[i] for i in range(n_items)) # Calc solution value

    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            if return_sol: return (value_with_penalty,), noisy_individual
            else: return (value_with_penalty,)
        else:
            if return_sol: return (0,), noisy_individual
            else: return (0,)
    if return_sol: return (value,), noisy_individual
    else: return (value,)

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

