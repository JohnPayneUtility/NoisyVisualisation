# IMPORTS
import numpy as np
import random

from ..tracking.logger import get_active_logger

# ==============================

def mean_weight(items_dict):
    total_weight = sum(weight for _, weight in items_dict.values())
    mean_weight = total_weight / len(items_dict)
    return mean_weight

# ==============================
# Combinatorial Fitness Functions
# ==============================
#
# Evaluation logging: when an MO evaluation logger is recording this call (noisyvis.tracking.mo_logger),
# each evaluator reports the genotype it evaluated (x~ = the individual: posterior noise), the true
# objectives f(x) and the observed vector it returns. f(x) is computed from the same sums with the zero
# noise terms a clean evaluation (noise_intensity=0) adds, so it equals that clean evaluation in value
# and type. It is computed only while logging, draws no RNG, and the returned vector is unchanged.

def eval_noisy_kp_v1_mo(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0, penalty=0):
    """ Function calculates fitness for knapsack problem individual """
    def objectives(value, weight):
        # Check if over capacity and return reduced value
        if weight > capacity:
            if penalty == 1:
                value_with_penalty = capacity - weight
                return (value_with_penalty, weight)
            else:
                return (value, weight)
        return (value, weight) # Not over capacity return value

    # Calculate weights and values
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    # initialise and generate noise; at zero noise nothing is drawn, so a clean evaluation
    # (noise_intensity=0) leaves the RNG untouched.
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity * mean_weight(items_dict)) if noise_intensity != 0 else 0
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity * mean_weight(items_dict)) if noise_intensity != 0 else 0
    # Add noise to objectives
    observed = objectives(value + noise1, weight + noise2)
    # log and return fitness values
    log_mo_eval = getattr(get_active_logger(), "log_mo_eval", None)
    if log_mo_eval is not None:
        log_mo_eval(individual, objectives(value, weight), observed)
    return observed

def noisy_mokp_v1(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0, penalty=0):
    noisy_objective = 1
    observed = eval_noisy_kp_v1_mo(individual, items_dict, capacity, noise_intensity, noisy_objective, penalty)
    return observed

def noisy_mokp_v2(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0, penalty=0):
    noisy_objective = 2
    observed = eval_noisy_kp_v1_mo(individual, items_dict, capacity, noise_intensity, noisy_objective, penalty)
    return observed

def noisy_mokp_both(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0, penalty=0):
    noisy_objective = 0
    observed = eval_noisy_kp_v1_mo(individual, items_dict, capacity, noise_intensity, noisy_objective, penalty)
    return observed

def eval_noisy_kp_v1_mo_violation(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0):
    """ Function calculates fitness for knapsack problem individual """
    def knap_violation(ind, items_dict, capacity):
        # items_dict[i] -> (value, weight); feasible iff total_w - capacity <= 0
        total_w = sum(int(ind[i]) * items_dict[i][1] for i in range(len(ind)))
        return max(0, float(total_w - capacity))

    # Calculate value and violation
    n_items = len(individual)
    violation = knap_violation(individual, items_dict, capacity)
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    # initialise and generate noise; at zero noise nothing is drawn, so a clean evaluation
    # (noise_intensity=0) leaves the RNG untouched.
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity * mean_weight(items_dict)) if noise_intensity != 0 else 0
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity * mean_weight(items_dict)) if noise_intensity != 0 else 0
    # Add noise to objectives
    observed = (value + noise1, violation + noise2)
    log_mo_eval = getattr(get_active_logger(), "log_mo_eval", None)
    if log_mo_eval is not None:
        log_mo_eval(individual, (value, violation), observed)
    return observed

def noisy_mokp_violation_v1(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0):
    noisy_objective = 1
    observed = eval_noisy_kp_v1_mo_violation(individual, items_dict, capacity, noise_intensity, noisy_objective)
    return observed

def noisy_mokp_violation_v2(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0):
    noisy_objective = 2
    observed = eval_noisy_kp_v1_mo_violation(individual, items_dict, capacity, noise_intensity, noisy_objective)
    return observed

def noisy_mokp_violation_both(individual, items_dict, capacity, noise_intensity=0, noisy_objective=0):
    noisy_objective = 0
    observed = eval_noisy_kp_v1_mo_violation(individual, items_dict, capacity, noise_intensity, noisy_objective)
    return observed

def oneMinMax(individual, noise_intensity=0, noisy_objective=0):
    """
    Function to evaluate solutions to one min max
    """
    # initialise and generate noise; at zero noise nothing is drawn, so a clean evaluation
    # (noise_intensity=0) leaves the RNG untouched.
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity) if noise_intensity != 0 else 0
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity) if noise_intensity != 0 else 0
    # Count ones and zeros
    num_ones = sum(individual)
    num_zeros = len(individual) - num_ones
    # Add noise to objectives
    value_ones = num_ones + noise1
    value_zeros = num_zeros + noise2
    # print((value_ones, value_zeros))
    observed = (value_ones, value_zeros)
    log_mo_eval = getattr(get_active_logger(), "log_mo_eval", None)
    if log_mo_eval is not None:
        log_mo_eval(individual, (num_ones, num_zeros), observed)
    return observed

def noisyMax_OneMinMax(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 1
    observed = oneMinMax(individual, noise_intensity, noisy_objective)
    return observed

def noisymin_OneMinMax(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 2
    observed = oneMinMax(individual, noise_intensity, noisy_objective)
    return observed

def noisyboth_OneMinMax(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 0
    observed = oneMinMax(individual, noise_intensity, noisy_objective)
    return observed

def countingOnesCountingZeros(individual, noise_intensity=0, noisy_objective=0):
    """
    Function to evaluate solutions to counting ones counting zeros
    """
    # initialise and generate noise; at zero noise nothing is drawn, so a clean evaluation
    # (noise_intensity=0) leaves the RNG untouched.
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity) if noise_intensity != 0 else 0
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity) if noise_intensity != 0 else 0
    # Calculate COCZ objectives
    n = len(individual)
    half = n // 2
    num_ones = sum(individual)
    ones_first_half = sum(individual[:half])
    zeros_second_half = half - sum(individual[half:])
    cocz = ones_first_half + zeros_second_half
    # Add noise to objectives
    value_ones = num_ones + noise1
    value_cocz = cocz + noise2
    # record and return observation
    observed = (value_ones, value_cocz)
    log_mo_eval = getattr(get_active_logger(), "log_mo_eval", None)
    if log_mo_eval is not None:
        log_mo_eval(individual, (num_ones, cocz), observed)
    return observed

def noisyOnes_COCZ(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 1
    observed = countingOnesCountingZeros(individual, noise_intensity, noisy_objective)
    return observed

def noisyCOCZ_COCZ(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 2
    observed = countingOnesCountingZeros(individual, noise_intensity, noisy_objective)
    return observed

def noisyboth_COCZ(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 0
    observed = countingOnesCountingZeros(individual, noise_intensity, noisy_objective)
    return observed

def leadingOnesTrailingZeros(individual, noise_intensity=0, noisy_objective=0):
    """
    Function to evaluate solutions to leading ones trailing zeros
    """
    # initialise and generate noise; at zero noise nothing is drawn, so a clean evaluation
    # (noise_intensity=0) leaves the RNG untouched.
    noise1, noise2 = 0, 0
    if noisy_objective == 0 or noisy_objective == 1:
        noise1 = random.gauss(0, noise_intensity) if noise_intensity != 0 else 0
    if noisy_objective == 0 or noisy_objective == 2:
        noise2 = random.gauss(0, noise_intensity) if noise_intensity != 0 else 0
    # Calculate LOTZ objectives
    # Count leading ones
    leading_ones = 0
    for bit in individual:
        if bit != 1:
            break
        leading_ones += 1
    # Count trailing zeros
    trailing_zeros = 0
    for bit in reversed(individual):
        if bit != 0:
            break
        trailing_zeros += 1
    # Add noise to objectives
    value_leading_ones = leading_ones + noise1
    value_trailing_zeros = trailing_zeros + noise2
    # record and return observation
    observed = (value_leading_ones, value_trailing_zeros)
    log_mo_eval = getattr(get_active_logger(), "log_mo_eval", None)
    if log_mo_eval is not None:
        log_mo_eval(individual, (leading_ones, trailing_zeros), observed)
    return observed

def noisyLO_LOTZ(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 1
    observed = leadingOnesTrailingZeros(individual, noise_intensity, noisy_objective)
    return observed

def noisyTZ_LOTZ(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 2
    observed = leadingOnesTrailingZeros(individual, noise_intensity, noisy_objective)
    return observed

def noisyboth_LOTZ(individual, noise_intensity=0, noisy_objective=0):
    noisy_objective = 0
    observed = leadingOnesTrailingZeros(individual, noise_intensity, noisy_objective)
    return observed