# IMPORTS
import numpy as np
import random

# ==============================

def mean_weight(items_dict):
    total_weight = sum(weight for _, weight in items_dict.values())
    mean_weight = total_weight / len(items_dict)
    return mean_weight

# ==============================
# Combinatorial Fitness Functions
# ==============================

def eval_noisy_kp_v1_mo(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0, penalty=0):
    """ Function calculates fitness for knapsack problem individual """
    # Calculate weights and values
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    # initialise and generate noise
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity * mean_weight(items_dict))
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity * mean_weight(items_dict))
    # Add noise to objectives
    value = value + noise1
    weight = weight + noise2
    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty, weight)
        else:
            # return (0, 100000)
            return (value, weight)
    return (value, weight) # Not over capacity return value

def eval_noisy_kp_v1_mo_violation(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0, penalty=0):
    """ Function calculates fitness for knapsack problem individual """
    def knap_violation(ind, items_dict, capacity):
        # items_dict[i] -> (value, weight); feasible iff total_w - capacity <= 0
        total_w = sum(int(ind[i]) * items_dict[i][1] for i in range(len(ind)))
        return max(0, float(total_w - capacity))

    # Calculate weights and values
    n_items = len(individual)
    v = knap_violation(individual, items_dict, capacity)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    # initialise and generate noise
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity * mean_weight(items_dict))
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity * mean_weight(items_dict))
    # Add noise to objectives
    value = value + noise1
    weight = weight + noise2
    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty, weight)
        else:
            # return (0, 100000)
            return (value, v)
    return (value, v) # Not over capacity return value

def countingOnesCountingZeros(individual, noise_intensity=0, noisy_objective=0):
    """
    Function to evaluate solutions to counting ones counting zeros
    """
    # initialise and generate noise
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity)
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity)
    # Count ones and zeros
    num_ones = sum(individual)
    num_zeros = len(individual) - num_ones
    # Add noise to objectives
    value_ones = num_ones + noise1
    value_zeros = num_zeros + noise2
    # print((value_ones, value_zeros))
    return (value_ones, value_zeros)


