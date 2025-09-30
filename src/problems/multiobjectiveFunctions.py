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

def eval_noisy_kp_v1_mo(individual, items_dict, capacity, noise_intensity=0, penalty=1):
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
            return (value_with_penalty, weight)
        else:
            return (0, 100000)
    return (value, weight) # Not over capacity return value

