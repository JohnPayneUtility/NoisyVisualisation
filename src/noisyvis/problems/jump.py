"""Jump fitness function (moved verbatim from FitnessFunctions.py, Stage 9)."""

# ==============================
# Combinatorial Fitness Functions
# ==============================

def jump_fitness(individual, gap_size, noise_intensity):
    """ Calculates fitness for jump problem """
    # print(individual)
    n = len(individual)
    ones = sum(individual)
    
    if ones == n or ones <= n-gap_size:
        return (ones,)
    else:
        return (n - ones + gap_size,)
