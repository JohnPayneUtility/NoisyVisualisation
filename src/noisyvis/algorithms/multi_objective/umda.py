"""The multi-objective UMDA family: MoUMDA, MoUMDA_noDuplicates, MoUMDA_ParetoArchive and their helpers."""

# IMPORTS
import numpy as np

from typing import Optional

from deap import creator
from deap import tools

from .base import OptimisationAlgorithm, record_pareto_data

# ==============================
# Helpers
# ==============================

def _ind_to_key(ind):
    # hashable representation of a solution
    return tuple(ind)

def mo_umda_update_full(len_sol, population, pop_size, select_size, toolbox,
                        prob_margin=True, margin_scale=1.0, prevent_duplicates=False):
    """
    NSGA-II (non-dominated sorting + crowding) selection of μ parents,
    then UMDA-style model update and sampling of λ=pop_size offspring.

    prob_margin: for binary genes, clamp p to [1/n, 1-1/n] (classic UMDA margin).
    margin_scale: scale factor on 1/n margin (set >1 to be more conservative).
    """
    # --- 1) Select μ parents by Pareto rank + crowding distance
    # (Population must be evaluated already — your base class ensures this.)
    parents = tools.selNSGA2(population, select_size)

    # --- 2) Detect gene type
    gene_type = type(population[0][0])

    # --- 3) Fit the univariate model on parents & sample new offspring
    if gene_type == int:
        probs = np.mean(parents, axis=0)

        if prob_margin:
            n = float(len_sol)
            eps = (margin_scale / n)
            probs = np.clip(probs, eps, 1.0 - eps)

        new_solutions = []
        seen = set() if prevent_duplicates else None

        while len(new_solutions) < pop_size:
            bits = (np.random.rand(len_sol) < probs).astype(int).tolist()
            ind = creator.Individual(bits)

            if prevent_duplicates:
                key = _ind_to_key(ind)
                if key in seen:
                    continue
                seen.add(key)

            new_solutions.append(ind)

    elif gene_type == float:
        arr = np.array(parents, dtype=float)
        means = np.mean(arr, axis=0)
        stds  = np.maximum(np.std(arr, axis=0), 1e-12)

        new_solutions = []
        seen = set() if prevent_duplicates else None

        while len(new_solutions) < pop_size:
            vals = np.random.normal(means, stds, size=len_sol).tolist()
            ind = creator.Individual(vals)

            if prevent_duplicates:
                key = tuple(np.round(vals, 12))  # avoid FP noise
                if key in seen:
                    continue
                seen.add(key)

            new_solutions.append(ind)

    else:
        raise ValueError("Unsupported gene type for moUMDA. Use int (binary) or float.")

    return new_solutions

def _update_archive_nondominated(archive, candidates):
    combined = list(archive) + list(candidates)
    if not combined:
        return []

    fronts = tools.sortNondominated(combined, k=len(combined), first_front_only=False)
    nd = list(fronts[0])  # keep only non-dominated
    return nd


def mo_umda_update_with_archive(
    len_sol,
    population,
    pop_size,
    select_size,
    toolbox,
    archive,
    prob_margin=True,
    margin_scale=1.0,
):
    """
    Reuses your existing flow:
      1) selNSGA2 on current pop -> parents (μ)
      2) archive <- nondominated(archive ∪ parents)
      3) fit UMDA model on archive
      4) sample λ offspring
    Returns: (new_population, new_archive)
    """

    # 1) select μ
    parents = tools.selNSGA2(population, select_size)

    # 2) update archive with selected items; remove dominated
    new_archive = _update_archive_nondominated(
        archive, parents
    )

    # 3) detect gene type (same as you do)
    gene_type = type(population[0][0])

    # 4) fit on ARCHIVE (key change) and sample λ
    if gene_type == int:
        # if archive empty (can happen at very start), fallback to parents
        model_source = new_archive if new_archive else parents
        probs = np.mean(model_source, axis=0)

        if prob_margin:
            n = float(len_sol)
            eps = (margin_scale / n)
            probs = np.clip(probs, eps, 1.0 - eps)

        new_solutions = []
        for _ in range(pop_size):
            bits = (np.random.rand(len_sol) < probs).astype(int).tolist()
            new_solutions.append(creator.Individual(bits))

    elif gene_type == float:
        model_source = np.array(new_archive if new_archive else parents, dtype=float)
        means = np.mean(model_source, axis=0)
        stds  = np.std(model_source, axis=0)
        stds  = np.maximum(stds, 1e-12)

        new_solutions = []
        for _ in range(pop_size):
            vals = np.random.normal(means, stds, size=len_sol).tolist()
            new_solutions.append(creator.Individual(vals))

    else:
        raise ValueError("Unsupported gene type for moUMDA. Use int (binary) or float.")

    return new_solutions, new_archive

# ==============================
# Estimation of Distribution Algorithm Subclasses
# ==============================

class MoUMDA(OptimisationAlgorithm):
    def __init__(self,
                 pop_size: int,
                 select_size: Optional[int] = None,
                 prob_margin: bool = False,
                 margin_scale: float = 1.0,
                 prevent_duplicates: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.pop_size = pop_size
        self.select_size = int(pop_size/2) if select_size is None else select_size
        self.prob_margin = prob_margin
        self.margin_scale = margin_scale
        self.prevent_duplicates = prevent_duplicates

        if prevent_duplicates:
            self.name = f'MoUMDA_noDuplicates(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA_noDuplicates'
        else:
            self.name = f'MoUMDA(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA'

        # Initialise & evaluate μ population
        self.initialise_population(self.pop_size)

    def perform_generation(self):
        """One generation of MoUMDA."""
        # NSGA-II parent selection + UMDA model update
        self.population = mo_umda_update_full(
            self.sol_length,
            self.population,
            self.pop_size,
            self.select_size,
            self.toolbox,
            prob_margin=self.prob_margin,
            margin_scale=self.margin_scale,
        )

        # Evaluate new population
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        self.evals += self.pop_size

class MoUMDA_noDuplicates(OptimisationAlgorithm):
    def __init__(self,
                 pop_size: int,
                 select_size: Optional[int] = None,
                 prob_margin: bool = False,
                 margin_scale: float = 1.0,
                 prevent_duplicates: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.pop_size = pop_size
        self.select_size = int(pop_size/2) if select_size is None else select_size
        self.prob_margin = prob_margin
        self.margin_scale = margin_scale
        self.prevent_duplicates = prevent_duplicates

        if prevent_duplicates:
            self.name = f'MoUMDA_noDuplicates(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA_noDuplicates'
        else:
            self.name = f'MoUMDA(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA'

        # Initialise & evaluate μ population
        self.initialise_population(self.pop_size)

    def perform_generation(self):
        """One generation of MoUMDA."""
        # NSGA-II parent selection + UMDA model update
        self.population = mo_umda_update_full(
            self.sol_length,
            self.population,
            self.pop_size,
            self.select_size,
            self.toolbox,
            prob_margin=self.prob_margin,
            margin_scale=self.margin_scale,
        )

        # Evaluate new population
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        self.evals += self.pop_size

class MoUMDA_ParetoArchive(OptimisationAlgorithm):
    def __init__(
        self,
        pop_size: int,
        select_size: Optional[int] = None,
        prob_margin: bool = False,
        margin_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.pop_size = pop_size
        self.select_size = int(pop_size / 2) if select_size is None else select_size
        self.prob_margin = prob_margin
        self.margin_scale = margin_scale

        self.name = f"MoUMDA_ParetoArchive(λ={pop_size}, μ={self.select_size})"
        self.type = "MoUMDA_ParetoArchive"

        # NEW: archive of non-dominated solutions
        self.archive = []

        # keep your existing init behaviour
        self.initialise_population(self.pop_size)

    def record_state_pareto(self, population):
        # Record PF/HV based on the archive
        record_pareto_data(
            self.archive,
            self.pareto_solutions,
            self.pareto_fitnesses,
            self.pareto_true_fitnesses,
            self.true_pareto_solutions,
            self.true_pareto_fitnesses,
            self.noisy_pf_noisy_hypervolumes,
            self.noisy_pf_true_hypervolumes,
            self.true_pf_hypervolumes,
            self.n_gens_pareto_best,
            self.toolbox,
            self.opt_weights,
            self.true_fitness_function,
            self.ref_point,
            self.record_every_gen,
            self.gens,
            self.evals,
            self.seed_signature,
            self.verbose_rate
            )

    def perform_generation(self):
        # generate offspring AND update archive
        self.population, self.archive = mo_umda_update_with_archive(
            self.sol_length,
            self.population,
            self.pop_size,
            self.select_size,
            self.toolbox,
            archive=self.archive,
            prob_margin=self.prob_margin,
            margin_scale=self.margin_scale,
        )

        # evaluate offspring (same as before)
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        self.evals += self.pop_size
