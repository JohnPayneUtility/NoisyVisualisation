"""SEMO: simple evolutionary multi-objective optimiser with a non-dominated archive."""

# IMPORTS
import random

from .base import OptimisationAlgorithm

# ==============================
# Mutation Functions
# ==============================

def mut_flip_one_bit(individual):
    i = random.randrange(len(individual))
    individual[i] = 1 - individual[i]  # assumes binary
    return (individual,)

# ==============================
# Evolutionary Algorithm Subclasses
# ==============================

class SEMO(OptimisationAlgorithm):
    def __init__(self, **kwargs):
        """
        Expect multi-objective fitness: opt_weights = (w1, w2, ..., wm)
        Use negative weights for minimization (DEAP maximizes by default).
        """
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.name = "SEMO"
        self.type = "SEMO"

        # Register one-bit mutation and (optional) helper
        self.toolbox.register("mutate_one_bit", mut_flip_one_bit)

        # Initialise archive P with a single random solution
        self.initialise_population(pop_size=1)   # P = {x}

    # ---- Pareto helpers ----
    @staticmethod
    def same_genotype(a, b):
        return tuple(a) == tuple(b)

    def dominated_by_archive(self, cand):
        # y' is dominated by any p in P?
        for p in self.population:
            if p.fitness.dominates(cand.fitness):
                return True
        return False

    def prune_dominated_by(self, cand):
        # Remove all p ∈ P that y' dominates
        newP = []
        for p in self.population:
            if cand.fitness.dominates(p.fitness):
                continue
            newP.append(p)
        self.population = newP

    def already_in_archive(self, cand):
        return any(self.same_genotype(cand, p) for p in self.population)

    # ---- One SEMO step ----
    def perform_generation(self):
        # 1) pick parent uniformly from archive P
        parent = random.choice(self.population)

        # 2) one-bit mutation
        offspring = self.toolbox.clone(parent)
        offspring, = self.toolbox.mutate_one_bit(offspring)

        # 3) evaluate
        del offspring.fitness.values
        offspring.fitness.values = self.toolbox.evaluate(offspring)
        self.evals += 1

        # 4) dominance-based archive update
        if self.dominated_by_archive(offspring):
            return  # discard y'
        if self.already_in_archive(offspring):
            return  # y' ∈ P -> do nothing

        # keep only non-dominated
        self.prune_dominated_by(offspring)
        self.population.append(offspring)
