"""Evolutionary operators shared by the single-objective, multi-objective and LON code (plan Stage 8, D4).

Attribute generators, mutation and crossover operators that were byte-identical copies in the algorithm
and LON modules. Definitions moved verbatim; `noisyvis.algorithms` re-exports every name, because configs
look `attr_function` names up there.
"""

import random

def binary_attribute():
        return random.randint(0, 1)

def Rastrigin_attribute():
    return random.uniform(-5.12, 5.12)

def mutSwapBit(individual, indpb):
    if random.random() < indpb and len(individual) >= 2:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return (individual,)

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

def complementary_crossover(parent1, parent2):
    assert len(parent1) == len(parent2), "Parents must have the same length."
    
    # Create empty offspring as lists
    offspring1 = type(parent1)([])
    offspring2 = type(parent2)([])
    
    # Generate the offspring
    for x1, x2 in zip(parent1, parent2):
        a = random.randint(0, 1)  # Randomly choose 0 or 1 with equal probability
        offspring1.append(a * x1 + (1 - a) * x2)
        offspring2.append((1 - a) * x1 + a * x2)

    return offspring1, offspring2
