"""Continuous fitness functions (moved verbatim from FitnessFunctions.py, Stage 9)."""

# IMPORTS
import numpy as np
import random
from ..tracking.logger import get_active_logger

# ==============================
# Continuous Fitness Functions
# ==============================

def rastrigin_eval(individual, amplitude=10, noise_intensity=0):
    A = amplitude
    n = len(individual)
    true_fitness = A * n + sum((x ** 2 - A * np.cos(2 * np.pi * x)) for x in individual)
    noisy_fitness = true_fitness + random.gauss(0, noise_intensity)

    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def birastrigin_eval(individual, d=1, s=None, noise_intensity=0):
    """
    Fitness evaluation for the Birastrigin problem

    Args:
        individual (list or np.ndarray): The input vector representing an individual.
        d (float, optional): Parameter `d`, standardized to 1 unless specified otherwise.
        s (float, optional): Parameter `s`, if not provided, it is calculated as per the formula.
        noise_intensity (float, optional): Standard deviation of posterior Gaussian noise.

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
    true_fitness = min(term1, term2) + term3
    noisy_fitness = true_fitness + random.gauss(0, noise_intensity)

    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return (noisy_fitness,)

def ackley(x, a=20, b=0.2, c=2*np.pi, noise_intensity=0):
    """
    Compute the Ackley function value for a given input vector x.

    :param x: List or NumPy array of input values.
    :param a: Parameter controlling the function's steepness (default 20).
    :param b: Parameter controlling the exponential term (default 0.2).
    :param c: Parameter controlling the cosine term (default 2π).
    :param noise_intensity: Standard deviation of posterior Gaussian noise (default 0).
    :return: Ackley function value.
    """
    individual = x
    x = np.array(x)
    d = len(x)

    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)

    true_fitness = term1 + term2 + a + np.exp(1)
    noisy_fitness = true_fitness + random.gauss(0, noise_intensity)

    logger = get_active_logger()
    if logger is not None:
        logger.log_noisy_eval(individual, individual, true_fitness, noisy_fitness)

    return noisy_fitness
