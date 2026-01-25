# IMPORTS
import random
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Any

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# from pymoo.indicators.hv import HV
import optuna

import pydoc

try: # try import fast hypervolume else fallback to python implementation
    from deap.tools._hypervolume import hv as _hv # fast compiled version
    hypervolume = _hv.hypervolume
except Exception:
    from deap.tools._hypervolume.pyhv import hypervolume # python version

# ==============================
# Helpers
# ==============================

def _ind_to_key(ind):
    # hashable representation of a solution
    return tuple(ind)

# ==============================
# Attribute Functions
# ==============================

def binary_attribute():
        return random.randint(0, 1)

def Rastrigin_attribute():
    return random.uniform(-5.12, 5.12)

# ==============================
# Mutation Functions
# ==============================

def mutSwapBit(individual, indpb):
    if random.random() < indpb and len(individual) >= 2:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return (individual,)

def mut_flip_one_bit(individual):
    i = random.randrange(len(individual))
    individual[i] = 1 - individual[i]  # assumes binary
    return (individual,)

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

# def mo_umda_update_full(len_sol, population, pop_size, select_size, toolbox,
#                         prob_margin=True, margin_scale=1.0):
#     """
#     NSGA-II (non-dominated sorting + crowding) selection of μ parents,
#     then UMDA-style model update and sampling of λ=pop_size offspring.

#     prob_margin: for binary genes, clamp p to [1/n, 1-1/n] (classic UMDA margin).
#     margin_scale: scale factor on 1/n margin (set >1 to be more conservative).
#     """
#     # --- 1) Select μ parents by Pareto rank + crowding distance
#     # (Population must be evaluated already — your base class ensures this.)
#     parents = tools.selNSGA2(population, select_size)

#     # --- 2) Detect gene type
#     gene_type = type(population[0][0])

#     # --- 3) Fit the univariate model on parents & sample new offspring
#     if gene_type == int:
#         # Binary UMDA: per-bit marginals
#         probs = np.mean(parents, axis=0)

#         if prob_margin:
#             n = float(len_sol)
#             eps = (margin_scale / n)
#             probs = np.clip(probs, eps, 1.0 - eps)

#         new_solutions = []
#         for _ in range(pop_size):
#             bits = (np.random.rand(len_sol) < probs).astype(int).tolist()
#             new_solutions.append(creator.Individual(bits))

#     elif gene_type == float:
#         # Real-valued UMDA: mean & std per position
#         arr = np.array(parents, dtype=float)
#         means = np.mean(arr, axis=0)
#         stds  = np.std(arr, axis=0)
#         stds  = np.maximum(stds, 1e-12)  # avoid degenerate σ

#         new_solutions = []
#         for _ in range(pop_size):
#             vals = np.random.normal(means, stds, size=len_sol).tolist()
#             new_solutions.append(creator.Individual(vals))

#     else:
#         raise ValueError("Unsupported gene type for moUMDA. Use int (binary) or float.")

#     return new_solutions

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
# Helper Functions
# ==============================

def front_sig(front_inds):
    # represent each solution as a tuple of ints, then frozenset for order-insensitivity
    sols = [tuple(int(x) for x in ind) for ind in (front_inds or [])]
    return frozenset(sols)

# def record_population_state(data, population, toolbox, true_fitness_function):
#     """
#     Record the current state of the population.
#     """
#     all_generations, best_solutions, best_fitnesses, true_fitnesses = data

#     # Record the current population
#     all_generations.append([ind[:] for ind in population])
    
#     # Identify the best individual in the current population
#     best_individual = tools.selBest(population, 1)[0]
#     best_solutions.append(toolbox.clone(best_individual))
#     best_fitnesses.append(best_individual.fitness.values[0])
    
#     # If provided record true fitness
#     if true_fitness_function is not None:
#         true_fit = true_fitness_function[0](best_individual, **true_fitness_function[1])
#         true_fitnesses.append(true_fit[0])
#     else:
#         true_fitnesses.append(best_individual.fitness.values[0])

def record_pareto_data(
    population,
    pareto_solutions,  # noisy PF solutions
    pareto_fitnesses,  # noisy PF noisy fitnesses
    pareto_true_fitnesses,  # noisy PF true fitnesses
    true_pareto_solutions,  # true PF approx. solutions
    true_pareto_fitnesses,  # true PF approx. fitnesses
    noisy_pf_noisy_hypervolumes,  # noisy HV of noisy PF
    noisy_pf_true_hypervolumes,  # true HV of noisy PF
    true_pf_hypervolumes,  # HV of true PF approximation
    n_gens_pareto_best,
    toolbox,
    opt_weights,  # optimisation weights for multiobjective
    true_fitness_function=None,
    ref_point=None,  # reference point for HV calculation
    record_every_gen=False,
    gen=None,
    eval=None,
    seed_signature=None,
    verbose_rate=0
):
    """
    """
    # Asserts
    assert true_fitness_function is not None, "true_fitness_function must be provided."
    assert ref_point is not None, "ref_point must be provided for hypervolume calculation."

    # Use config ref point and objectives to determine ref for HV calculation
    w = np.asarray(opt_weights, dtype=float)
    sign = np.where(w > 0, -1.0, 1.0)  # flip max->min for HV
    hv_ref = np.asarray(ref_point, dtype=float) * sign

    def _should_print(g: int) -> bool:
        # Check if should print update statement to terminal
        if verbose_rate == 0:
            return False
        return (g % verbose_rate) == 0

    # =========================
    # 1) Noisy Pareto front
    # =========================
    pareto_front = tools.ParetoFront()
    pareto_front.update(population)
    pf_clone = [toolbox.clone(ind) for ind in pareto_front]

    # Check if PF changed and report
    curr_sig = front_sig(pf_clone)
    n_improvements = len(n_gens_pareto_best)

    if pareto_solutions:
        last_sig = front_sig(pareto_solutions[-1])
        if curr_sig == last_sig:
            n_gens_pareto_best[-1] += 1
            if _should_print(gen):
                print(
                    f"[SeedSig {seed_signature}] | "
                    f"[Gen {gen}] No PF change | "
                    f"[Eval {eval}] No PF Change | "
                    f"total improvements: {n_improvements} | "
                    f"since last improvement: {n_gens_pareto_best[-1]}"
                )
            if not record_every_gen:
                return
            pareto_solutions.append(pf_clone)
        else:
            n_gens_pareto_best.append(1)
            pareto_solutions.append(pf_clone)
            if _should_print(gen):
                print(
                    f"[SeedSig {seed_signature}] | "
                    f"[Gen {gen}] PF Changed | "
                    f"[Eval {eval}] PF Changed | "
                    f"PF size: {len(pf_clone)} | "
                    f"total improvements: {n_improvements + 1}"
                )
    else: # Initial Record
        n_gens_pareto_best.append(1)
        pareto_solutions.append(pf_clone)
        if verbose_rate != 0:
            print(f"[Gen {gen}] Initial PF recorded | PF size: {len(pf_clone)}")

    # noisy evals of the noisy PF
    noisy_fit_list = [ind.fitness.values for ind in pareto_front]
    pareto_fitnesses.append(noisy_fit_list)

    # true evals of the noisy PF
    tf, tf_kwargs = true_fitness_function
    true_fit_list = [tf(ind, **tf_kwargs) for ind in pareto_front]
    pareto_true_fitnesses.append(true_fit_list)

    # =========================
    # 2) HV for noisy pareto front
    # =========================
    noisy_pts = np.asarray(noisy_fit_list, dtype=float) * sign
    hv_noisy = hypervolume(noisy_pts, hv_ref)
    noisy_pf_noisy_hypervolumes.append(float(hv_noisy))

    true_pts_for_noisy_pf = np.asarray(true_fit_list, dtype=float) * sign
    hv_noisy_true = hypervolume(true_pts_for_noisy_pf, hv_ref)
    noisy_pf_true_hypervolumes.append(float(hv_noisy_true))

    # =========================
    # 3) TRUE Pareto front (FULL POP, TRUE EVALS) — without touching originals
    # =========================
    pop_true = [toolbox.clone(ind) for ind in population]
    for ind_clone, ind_orig in zip(pop_true, population):
        ind_clone.fitness.values = tf(ind_orig, **tf_kwargs)

    true_pf = tools.ParetoFront()
    true_pf.update(pop_true)

    if true_pareto_solutions is not None:
        true_pareto_solutions.append([toolbox.clone(ind) for ind in true_pf])

    true_pf_fit_true = [ind.fitness.values for ind in true_pf]
    if true_pareto_fitnesses is not None:
        true_pareto_fitnesses.append(true_pf_fit_true)

    # =========================
    # 4) TRUE HV (of the TRUE PF)
    # =========================
    true_pts = np.asarray(true_pf_fit_true, dtype=float) * sign
    hv_true = hypervolume(true_pts, hv_ref)
    true_pf_hypervolumes.append(float(hv_true))

# def extract_trajectory_data(best_solutions, best_fitnesses, true_fitnesses):
#     # Extract unique solutions and their corresponding fitness values
#     unique_solutions = []
#     unique_fitnesses = []
#     noisy_fitnesses = []
#     solution_iterations = []
#     seen_solutions = {}

#     for solution, fitness, true_fitness in zip(best_solutions, best_fitnesses, true_fitnesses):
#         solution_tuple = tuple(solution)
#         if solution_tuple not in seen_solutions:
#             seen_solutions[solution_tuple] = 1
#             unique_solutions.append(solution)
#             unique_fitnesses.append(true_fitness)
#             noisy_fitnesses.append(fitness)
#         else:
#             seen_solutions[solution_tuple] += 1

#     # Create a list of iteration counts for each unique solution
#     for solution in unique_solutions:
#         solution_tuple = tuple(solution)
#         solution_iterations.append(seen_solutions[solution_tuple])

#     return unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations

# def extract_transitions(unique_solutions):
#     # Extract transitions between solutions over generations
#     transitions = []

#     for i in range(1, len(unique_solutions)):
#         prev_solution = tuple(unique_solutions[i - 1])
#         current_solution = tuple(unique_solutions[i])
#         transitions.append((prev_solution, current_solution))

#     return transitions

# ==============================
# Base Algorithm Class
# ==============================

@dataclass
class OptimisationAlgorithm:
    sol_length: int
    opt_weights: Tuple[float, ...]
    gen_limit: Optional[int] = int(10e6)
    eval_limit: Optional[int] = None
    target_stop: Optional[float] = None
    stop_without_improvement_in_gens: Optional[int] = None
    attr_function: Optional[Callable] = None
    fitness_function: Optional[Tuple[Callable, dict]] = None
    starting_solution: Optional[List[Any]] = None
    true_fitness_function: Optional[Tuple[Callable, dict]] = None
    ref_point: Optional[list[Any]] = None
    record_every_gen: bool = False
    verbose_rate: int = 0
    
    # Create lists to store data, seperate for each instance
    # single objective data
    # all_generations: List[List[Any]] = field(default_factory=list)
    # best_solutions: List[Any] = field(default_factory=list)
    # best_fitnesses: List[float] = field(default_factory=list)
    # true_fitnesses: List[float] = field(default_factory=list)

    # multi objective data
    # noisy pareto front data
    pareto_solutions: List[List[Any]] = field(default_factory=list)
    pareto_fitnesses: List[List[Any]] = field(default_factory=list)
    pareto_true_fitnesses:List[List[Any]] = field(default_factory=list)
    # true approximated pareto front data
    true_pareto_solutions: List[List[Any]] = field(default_factory=list)
    true_pareto_fitnesses: List[List[Tuple[float, ...]]] = field(default_factory=list)
    # hypervolume data
    noisy_pf_noisy_hypervolumes: List[float] = field(default_factory=list)
    noisy_pf_true_hypervolumes: List[float] = field(default_factory=list)
    true_pf_hypervolumes: List[float] = field(default_factory=list)
    # iterations
    n_gens_pareto_best: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.stop_trigger = ''
        self.seed_signature = random.randint(0, 10**6)
        # self.noisy_pf = tools.ParetoFront()
        # self.true_pf = tools.ParetoFront()
        # self.data = [self.all_generations, self.best_solutions, self.best_fitnesses, self.true_fitnesses]

        # Fitness and individual creators
        if not hasattr(creator, "CustomFitness"):
            creator.create("CustomFitness", base.Fitness, weights=self.opt_weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.CustomFitness)

        # Create the toolbox and register common functions
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self.attr_function)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.sol_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", lambda ind: self.fitness_function[0](ind, **self.fitness_function[1]))

    def initialise_population(self, pop_size):
        self.population = self.toolbox.population(n=pop_size)
        # If a starting solution is provided, initialize all individuals with it
        if self.starting_solution is not None:
            for ind in self.population:
                ind[:] = self.starting_solution[:]
        # Evaluate initial population
        for ind in self.population:
            ind.fitness.values = self.toolbox.evaluate(ind)
        self.evals += pop_size

    @abstractmethod
    def perform_generation(self):
        """Perform one generation of specified algorithm"""
        pass
    
    def stop_condition(self) -> bool:
        """Check if stop condition has been met."""
        if self.eval_limit is not None and self.evals >= self.eval_limit:
            self.stop_trigger = 'eval_limit'
            return True
        if self.target_stop is not None and self.true_fitnesses and self.true_fitnesses[-1] >= self.target_stop:
            self.stop_trigger = 'target_reached'
            return True
        if self.gen_limit is not None and self.gens >= self.gen_limit:
            self.stop_trigger = 'gen_limit'
            return True
        if self.stop_without_improvement_in_gens is not None:
            if self.n_gens_pareto_best:
                no_improvement = int(self.n_gens_pareto_best[-1])
                if no_improvement >= int(self.stop_without_improvement_in_gens):
                    self.stop_trigger = 'no_improvement'
                    return True
        return False
    
    def run(self):
        """Run the algorithm using the common loop logic."""
        while not self.stop_condition():
            self.gens += 1
            self.perform_generation()
            # self.record_state(self.population)
            self.record_state_pareto(self.population)

    # def record_state(self, population):
    #     #Record the current population state.
    #     record_population_state(self.data, population, self.toolbox, self.true_fitness_function)
    
    def record_state_pareto(self, population):
        record_pareto_data(
            population,
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

    # def get_classic_data(self):
    #     return self.all_generations, self.best_solutions, self.best_fitnesses, self.true_fitnesses
    
    # def get_solution_data(self):
    #     return self.best_solutions, self.best_fitnesses, self.true_fitnesses
    
    # def get_trajectory_data(self):
    #     unique_sols, unique_fits, noisy_fitnesses, sol_iterations = extract_trajectory_data(self.best_solutions, self.best_fitnesses, self.true_fitnesses)
    #     sol_transitions = extract_transitions(unique_sols)
    #     return unique_sols, unique_fits, noisy_fitnesses, sol_iterations, sol_transitions

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
        # self.record_state(self.population)

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

# ==============================
# Estimation of Distribution Algorithm Subclasses
# ==============================

class MoUMDA(OptimisationAlgorithm):
    def __init__(self, 
                 pop_size: int,
                 select_size: Optional[int] = None,
                 prob_margin: bool = True,
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
        # self.record_state(self.population)  # optional first snapshot

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
                 prob_margin: bool = True,
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
        # self.record_state(self.population)  # optional first snapshot

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
        prob_margin: bool = True,
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
        # self.record_state(self.population)

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

class NSGA2(OptimisationAlgorithm):
    def __init__(
        self,
        pop_size: int = 100,
        cxpb: float = 0.9,
        mutpb: float = 0.1,
        mate_op=None,            # can be callable OR "deap.tools.cxTwoPoint"
        mutate_op=None,          # can be callable OR "deap.tools.mutFlipBit"
        mutate_params=None,      # keep this name to match your resolver
        mutate_kwargs=None,      # optional alias
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = "NSGA-II"
        self.type = "NSGA-II"

        self.pop_size = int(pop_size)
        self.cxpb = float(cxpb)
        self.mutpb = float(mutpb)

        self.gens = 0
        self.evals = 0

        # --- resolve operator references if they come in as strings/DictConfig ---
        def _resolve_callable(x):
            if x is None:
                return None
            # Hydra may pass OmegaConf nodes; str() gives dotted path nicely
            if not callable(x):
                x = str(x)
                obj = pydoc.locate(x)
                if obj is None or not callable(obj):
                    raise TypeError(f"Operator '{x}' could not be resolved to a callable.")
                return obj
            return x

        mate_fn = _resolve_callable(mate_op)
        mut_fn  = _resolve_callable(mutate_op)

        self.toolbox.register("select", tools.selNSGA2)

        if mate_fn is not None:
            self.toolbox.register("mate", mate_fn)

        # accept either mutate_params (your resolver) or mutate_kwargs
        if mutate_kwargs is None and mutate_params is not None:
            mutate_kwargs = dict(mutate_params)
        mutate_kwargs = mutate_kwargs or {}

        if mut_fn is not None:
            self.toolbox.register("mutate", mut_fn, **mutate_kwargs)

        # init + crowding distance
        self.initialise_population(pop_size=self.pop_size)
        self.population = self.toolbox.select(self.population, len(self.population))

        # self.record_state(self.population)
        self.record_state_pareto(self.population)

    def perform_generation(self):
        offspring = tools.selTournamentDCD(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cxpb:
                self.toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for ind in offspring:
            if random.random() < self.mutpb:
                out = self.toolbox.mutate(ind)
                if isinstance(out, tuple):
                    ind = out[0]
                del ind.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = self.toolbox.evaluate(ind)
        self.evals += len(invalid)

        self.population = self.toolbox.select(self.population + offspring, self.pop_size)


