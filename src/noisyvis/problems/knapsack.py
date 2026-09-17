"""Single-objective knapsack fitness functions (moved verbatim from FitnessFunctions.py, Stage 9)."""

# IMPORTS
import random
from ..algorithms.operators import random_bit_flip
from ..tracking.logger import get_active_logger
from .onemax import bitflip_prior_noise

# ==============================

def mean_weight(items_dict):
    total_weight = sum(weight for _, weight in items_dict.values())
    mean_weight = total_weight / len(items_dict)
    return mean_weight

# ==============================
# Combinatorial Fitness Functions
# ==============================

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

    if weight > capacity:
        if penalty == 1:
            noisy_fitness = capacity - weight
        else:
            noisy_fitness = 0
    else:
        noisy_fitness = noisy_value

    # Log the evaluation if logger is active
    # For posterior noise: same solution, different fitnesses
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_v1_penalty(individual, items_dict, capacity, noise_intensity=0, penalty=10):
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
        true_fitness = value - penalty * (weight - capacity)
    else:
        true_fitness = value

    # Calculate noisy fitness
    noise = random.gauss(0, noise_intensity * mean_weight(items_dict))
    noisy_value = value + noise

    if weight > capacity:
        noisy_fitness = noisy_value - penalty * (weight - capacity)
    else:
        noisy_fitness = noisy_value

    # Log the evaluation if logger is active
    # For posterior noise: same solution, different fitnesses
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

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
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_v2_penalty(individual, items_dict, capacity, noise_intensity=0, penalty=10):
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

    # Calculate true fitness
    if weight > capacity:
        true_fitness = value - penalty * (weight - capacity)
    else:
        true_fitness = value

    # Calculate noisy fitness
    noise = random.gauss(0, noise_intensity * mean_weight(items_dict))
    noisy_value = value + noise
    noise = random.gauss(0, noise_intensity * mean_weight(items_dict))
    noisy_weight = weight + noise

    if (noisy_weight) > capacity:
        noisy_fitness = noisy_value - penalty * (noisy_weight - capacity)
    else:
        noisy_fitness = noisy_value

    # Log the evaluation if logger is active
    # For posterior noise same solution, different fitnesses
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_v3(individual, items_dict, capacity, noise_intensity=0, penalty=1):
    """
    Noise only added to the weight
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
    noisy_value = value
    
    if (weight + noise) > capacity:
        if penalty == 1:
            noisy_fitness = capacity - (weight + noise)
        else:
            noisy_fitness = 0
    else:
        noisy_fitness = noisy_value

    # Log the evaluation if logger is active
    # For posterior noise: same solution, different fitnesses
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_prior_bitflip(individual, items_dict, capacity, noise_intensity=0, penalty=1):
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
    noisy_individual = bitflip_prior_noise(list(individual), noise_intensity / n_items)
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
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_prior_mult_bitflip(individual, items_dict, capacity, noise_intensity=0, penalty=1):
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
    # noisy_individual = bitflip_prior_noise(list(individual), noise_intensity / n_items)
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
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_pq_prior_bitwise(individual, items_dict, capacity, noise_intensity=0, noise_probability=1, penalty=1):
    """
    Calculates fitness for knapsack problem with prior noise.

    The individual is perturbed (bits flipped) before evaluation.
    If a logger is active, the original and noisy solutions are recorded.
    Note: For prior noise, original and noisy individual differ,
    true_fitness is for original, noisy_fitness is for perturbed.

    p = noise_probability / n_items: probability that noise is applied at all.
    q = noise_intensity / n_items: probability of flipping each bit, given noise is applied.
    """
    n_items = len(individual)
    p = noise_probability / n_items  # probability noise is applied
    q = noise_intensity / n_items    # per-bit flip probability

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
    if random.random() < p:
        noisy_individual = [1 - b if random.random() < q else b for b in individual]
    else:
        noisy_individual = list(individual)
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
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def eval_noisy_kp_1q_prior_bitwise(individual, items_dict, capacity, noise_intensity=0, penalty=10):
    """
    Calculates fitness for knapsack problem with prior noise.

    The individual is perturbed (bits flipped) before evaluation.
    If a logger is active, the original and noisy solutions are recorded.
    Note: For prior noise, original and noisy individual differ,
    true_fitness is for original, noisy_fitness is for perturbed.

    p = noise_probability / n_items: probability that noise is applied at all.
    q = noise_intensity / n_items: probability of flipping each bit, given noise is applied.
    """
    n_items = len(individual)
    q = noise_intensity / n_items    # per-bit flip probability

    # Calculate true fitness of original (unperturbed) solution
    orig_weight = sum(items_dict[i][1] * individual[i] for i in range(n_items))
    orig_value = sum(items_dict[i][0] * individual[i] for i in range(n_items))

    # Calculate true fitness (no noise)
    if orig_weight > capacity:
        true_fitness = orig_value - penalty * (orig_weight - capacity)
    else:
        true_fitness = orig_value

    # Create noisy (perturbed) solution and calculate its fitness
    noisy_individual = [1 - b if random.random() < q else b for b in individual]
    noisy_weight = sum(items_dict[i][1] * noisy_individual[i] for i in range(n_items))
    noisy_value = sum(items_dict[i][0] * noisy_individual[i] for i in range(n_items))

    # Calculate noisy fitness
    if noisy_weight > capacity:
        noisy_fitness = noisy_value - penalty * (noisy_weight - capacity)
    else:
        noisy_fitness = noisy_value

    # Log the evaluation if logger is active
    # For prior noise: different solutions, with their respective fitnesses
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)
