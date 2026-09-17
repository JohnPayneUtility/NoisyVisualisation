"""OneMax fitness functions and their prior-noise helper (moved verbatim from FitnessFunctions.py, Stage 9)."""

# IMPORTS
import random
from ..algorithms.operators import random_bit_flip
from ..tracking.logger import get_active_logger

# ==============================

def bitflip_prior_noise(individual, probability):
    """Flip a single randomly chosen bit with the given probability."""
    if random.random() < probability:
        idx = random.randrange(len(individual))
        individual[idx] = 1 - individual[idx]
    return individual

# ==============================
# Combinatorial Fitness Functions
# ==============================

def OneMax_fitness(individual, noise_intensity=0):
    """
    Calculates fitness for OneMax problem with posterior noise.

    Gaussian noise is added to the fitness value after evaluation.
    If a logger is active, the evaluation is recorded.
    """
    # Calculate true fitness (no noise)
    true_fitness = sum(individual)

    # Posterior noise: add noise to fitness value
    noisy_fitness = true_fitness + random.gauss(0, noise_intensity)

    # Log the evaluation if logger is active
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def OneMax_prior_bitflip_fitness(individual, noise_intensity=0):
    """
    Calculates fitness for OneMax problem with prior noise.

    The individual is perturbed via random bit flips before evaluation.
    If a logger is active, the original and noisy solutions are recorded.
    """
    # Calculate true fitness (no noise)
    true_fitness = sum(individual)
    n_items = len(individual)

    # Prior noise: perturb individual via random bit flips, then evaluate
    # noisy_individual, _ = random_bit_flip(list(individual), n_flips=noise_intensity)
    noisy_individual = bitflip_prior_noise(list(individual), noise_intensity / n_items)
    noisy_fitness = sum(noisy_individual)

    # Log the evaluation if logger is active
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def OneMax_prior_mult_bitflip_fitness(individual, noise_intensity=0):
    """
    Calculates fitness for OneMax problem with prior noise.

    The individual is perturbed via random bit flips before evaluation.
    If a logger is active, the original and noisy solutions are recorded.
    """
    # Calculate true fitness (no noise)
    true_fitness = sum(individual)
    # n_items = len(individual)

    # Prior noise: perturb individual via random bit flips, then evaluate
    noisy_individual, _ = random_bit_flip(list(individual), n_flips=noise_intensity)
    # noisy_individual = bitflip_prior_noise(list(individual), noise_intensity / n_items)
    noisy_fitness = sum(noisy_individual)

    # Log the evaluation if logger is active
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def OneMax_prior_pq_bitwise_fitness(individual, noise_probability=1, noise_intensity=1):
    """
    Calculates fitness for OneMax problem with prior bitwise noise.

    Each bit in the individual is independently flipped with probability
    noise_intensity * 1/n before evaluation.
    If a logger is active, the original and noisy solutions are recorded.
    """
    n_items = len(individual)
    p = noise_probability / n_items  # probability noise is applied
    q = noise_intensity / n_items    # per-bit flip probability

    # Calculate true fitness (no noise)
    true_fitness = sum(individual)

    # Prior noise: flip each bit independently with probability p
    if random.random() < p:
        noisy_individual = [1 - b if random.random() < q else b for b in individual]
    else:
        noisy_individual = list(individual)
    noisy_fitness = sum(noisy_individual)

    # Log the evaluation if logger is active
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def OneMax_prior_1q_bitwise_fitness(individual, noise_intensity=1):
    """
    Calculates fitness for OneMax problem with prior bitwise noise.

    Each bit in the individual is independently flipped with probability
    noise_intensity * 1/n before evaluation.
    If a logger is active, the original and noisy solutions are recorded.
    """
    n_items = len(individual)
    q = noise_intensity / n_items    # per-bit flip probability

    # Calculate true fitness (no noise)
    true_fitness = sum(individual)

    # Prior noise: flip each bit independently with probability p
    noisy_individual = [1 - b if random.random() < q else b for b in individual]
    noisy_fitness = sum(noisy_individual)

    # Log the evaluation if logger is active
    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, noisy_individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)
