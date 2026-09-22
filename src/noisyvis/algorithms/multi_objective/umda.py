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

# Stop triggers of the MoUMDA family
PROBABILITY_VECTOR_CONVERGED = "probability_vector_converged"
INSUFFICIENT_UNIQUE_SUPPORT = "insufficient_unique_support"

def _gene_kind(individuals):
    # "binary" (int genes) or "real" (float genes), from the first gene of the first individual
    gene_type = type(individuals[0][0])
    if gene_type == int:
        return "binary"
    if gene_type == float:
        return "real"
    raise ValueError("Unsupported gene type for moUMDA. Use int (binary) or float.")

# ==============================
# Binary UMDA: Bernoulli probability vector
# ==============================

def calculate_probability_vector(parents, len_sol, prob_margin, margin_scale):
    """
    Per-bit marginal probabilities of the parents.

    prob_margin: clamp p to [s/n, 1-s/n] with s = margin_scale (classic UMDA margin).
    """
    probs = np.mean(parents, axis=0)

    if prob_margin:
        n = float(len_sol)
        eps = (margin_scale / n)
        probs = np.clip(probs, eps, 1.0 - eps)

    return probs

def is_probability_vector_converged(probability_vector):
    # every p_i exactly 0 or 1: the vector can only generate one genotype
    if probability_vector is None:
        return False
    probs = np.asarray(probability_vector, dtype=float)
    return probs.size > 0 and bool(np.all((probs == 0.0) | (probs == 1.0)))

def count_free_bits(probability_vector):
    # positions with 0 < p_i < 1; the vector can generate 2**free_bits distinct genotypes
    probs = np.asarray(probability_vector, dtype=float)
    return int(np.count_nonzero((probs > 0.0) & (probs < 1.0)))

def binary_support_at_least(n_free_bits, required):
    # exactly 2**n_free_bits >= required, without building 2**n_free_bits
    return required <= 1 or n_free_bits >= (required - 1).bit_length()

def has_sufficient_unique_support(probability_vector, required):
    return binary_support_at_least(count_free_bits(probability_vector), required)

def no_duplicates_stop_reason(probability_vector, pop_size):
    """
    Why a duplicate-free population of pop_size cannot be sampled from this vector, or None.
    Convergence (support 1) is reported in preference to insufficient support.
    """
    if is_probability_vector_converged(probability_vector):
        return PROBABILITY_VECTOR_CONVERGED
    if not has_sufficient_unique_support(probability_vector, pop_size):
        return INSUFFICIENT_UNIQUE_SUPPORT
    return None

def sample_from_probability_vector(probability_vector, pop_size, prevent_duplicates=False):
    """
    Sample pop_size binary individuals, one np.random.rand draw per candidate.

    prevent_duplicates: rejection-sample distinct genotypes. Raises ValueError before drawing
    if the vector cannot generate pop_size distinct genotypes.
    """
    if prevent_duplicates and not has_sufficient_unique_support(probability_vector, pop_size):
        raise ValueError(
            f"Probability vector has {count_free_bits(probability_vector)} free bits, so it supports "
            f"fewer than pop_size={pop_size} distinct genotypes."
        )

    len_sol = len(probability_vector)
    new_solutions = []
    seen = set() if prevent_duplicates else None

    while len(new_solutions) < pop_size:
        bits = (np.random.rand(len_sol) < probability_vector).astype(int).tolist()
        ind = creator.Individual(bits)

        if prevent_duplicates:
            key = _ind_to_key(ind)
            if key in seen:
                continue
            seen.add(key)

        new_solutions.append(ind)

    return new_solutions

# ==============================
# Real-valued UMDA: Gaussian marginals
# ==============================

def calculate_gaussian_marginals(parents):
    # per-position means and standard deviations (floored to avoid degenerate sigma)
    arr = np.array(parents, dtype=float)
    means = np.mean(arr, axis=0)
    stds  = np.maximum(np.std(arr, axis=0), 1e-12)
    return means, stds

def sample_from_gaussian_marginals(means, stds, pop_size, prevent_duplicates=False):
    # sample pop_size real-valued individuals, one np.random.normal draw per candidate
    len_sol = len(means)
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

    return new_solutions

# ==============================
# Legacy one-shot update helpers
# ==============================

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

    # --- 2) Fit the univariate model on parents & sample new offspring
    if _gene_kind(population) == "binary":
        probability_vector = calculate_probability_vector(parents, len_sol, prob_margin, margin_scale)
        return sample_from_probability_vector(probability_vector, pop_size, prevent_duplicates=prevent_duplicates)

    means, stds = calculate_gaussian_marginals(parents)
    return sample_from_gaussian_marginals(means, stds, pop_size, prevent_duplicates=prevent_duplicates)

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

    # 3) fit on ARCHIVE (key change) and sample λ
    # if archive empty (can happen at very start), fallback to parents
    model_source = new_archive if new_archive else parents

    if _gene_kind(population) == "binary":
        probability_vector = calculate_probability_vector(model_source, len_sol, prob_margin, margin_scale)
        new_solutions = sample_from_probability_vector(probability_vector, pop_size)
    else:
        means, stds = calculate_gaussian_marginals(model_source)
        new_solutions = sample_from_gaussian_marginals(means, stds, pop_size)

    return new_solutions, new_archive

# ==============================
# Estimation of Distribution Algorithm Subclasses
# ==============================

class MoUMDABase(OptimisationAlgorithm):
    """
    Shared state and generation of the MoUMDA family: NSGA-II selection of μ parents, a univariate
    model fitted on the distribution source, and λ = pop_size sampled offspring replacing the population.

    probability_vector: the binary probability vector used by the last successfully completed
    generation (None before generation 1, and for real-valued genes).
    """
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
        self.probability_vector = None

        # Initialise & evaluate λ population
        self.initialise_population(self.pop_size)

    def _select_parents(self):
        # NSGA-II parent selection (Pareto rank + crowding distance)
        return tools.selNSGA2(self.population, self.select_size)

    def _distribution_source(self):
        """Individuals the model is fitted on. Called only by perform_generation (it may commit state)."""
        return self._select_parents()

    def perform_generation(self):
        """One generation: construct offspring, evaluate them, and only then commit."""
        # Construct
        source = self._distribution_source()
        if _gene_kind(self.population) == "binary":
            probability_vector = calculate_probability_vector(
                source, self.sol_length, self.prob_margin, self.margin_scale
            )
            # prevent_duplicates is not propagated yet (known MoUMDA_noDuplicates defect)
            offspring = sample_from_probability_vector(probability_vector, self.pop_size)
        else:
            means, stds = calculate_gaussian_marginals(source)
            offspring = sample_from_gaussian_marginals(means, stds, self.pop_size)
            probability_vector = None

        # Evaluate
        fitnesses = list(map(self.toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Commit, only after successful evaluation
        self.population = offspring
        self.evals += self.pop_size
        self.probability_vector = probability_vector

class MoUMDA(MoUMDABase):
    def __init__(self,
                 pop_size: int,
                 select_size: Optional[int] = None,
                 prob_margin: bool = False,
                 margin_scale: float = 1.0,
                 prevent_duplicates: bool = False,
                 **kwargs):
        super().__init__(pop_size, select_size, prob_margin, margin_scale, prevent_duplicates, **kwargs)

        if prevent_duplicates:
            self.name = f'MoUMDA_noDuplicates(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA_noDuplicates'
        else:
            self.name = f'MoUMDA(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA'

class MoUMDA_noDuplicates(MoUMDABase):
    def __init__(self,
                 pop_size: int,
                 select_size: Optional[int] = None,
                 prob_margin: bool = False,
                 margin_scale: float = 1.0,
                 prevent_duplicates: bool = True,
                 **kwargs):
        super().__init__(pop_size, select_size, prob_margin, margin_scale, prevent_duplicates, **kwargs)

        if prevent_duplicates:
            self.name = f'MoUMDA_noDuplicates(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA_noDuplicates'
        else:
            self.name = f'MoUMDA(p={pop_size}, μ={self.select_size})'
            self.type = 'MoUMDA'

class MoUMDA_ParetoArchive(MoUMDABase):
    def __init__(
        self,
        pop_size: int,
        select_size: Optional[int] = None,
        prob_margin: bool = False,
        margin_scale: float = 1.0,
        **kwargs
    ):
        # no prevent_duplicates parameter: passing one still raises TypeError
        super().__init__(pop_size, select_size, prob_margin, margin_scale, prevent_duplicates=False, **kwargs)

        self.name = f"MoUMDA_ParetoArchive(λ={pop_size}, μ={self.select_size})"
        self.type = "MoUMDA_ParetoArchive"

        # NEW: archive of non-dominated solutions
        self.archive = []

    def _distribution_source(self):
        # NSGA-II parents update the non-dominated archive; the model is fitted on the archive
        parents = self._select_parents()
        self.archive = _update_archive_nondominated(self.archive, parents)
        # if archive empty (can happen at very start), fallback to parents
        return self.archive if self.archive else parents

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
