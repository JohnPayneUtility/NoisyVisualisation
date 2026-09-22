"""Shared multi-objective infrastructure: the MO base class and Pareto-front / hypervolume recording."""

# IMPORTS
import random
import numpy as np

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Any

# deap.base is aliased: this module is itself the package's `base`
from deap import base as deap_base
from deap import creator
from deap import tools

try: # try import fast hypervolume else fallback to python implementation
    from deap.tools._hypervolume import hv as _hv # fast compiled version
    hypervolume = _hv.hypervolume
except Exception:
    from deap.tools._hypervolume.pyhv import hypervolume # python version

# ==============================
# Helper Functions
# ==============================

def front_sig(front_inds):
    # represent each solution as a tuple of ints, then frozenset for order-insensitivity
    sols = [tuple(int(x) for x in ind) for ind in (front_inds or [])]
    return frozenset(sols)

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

        # Fitness and individual creators
        # Check if CustomFitness exists with matching weights; recreate if weights differ
        if hasattr(creator, "CustomFitness"):
            if creator.CustomFitness.weights != self.opt_weights:
                del creator.CustomFitness
                del creator.Individual
        if not hasattr(creator, "CustomFitness"):
            creator.create("CustomFitness", deap_base.Fitness, weights=self.opt_weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.CustomFitness)

        # Create the toolbox and register common functions
        self.toolbox = deap_base.Toolbox()
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
            self.record_state_pareto(self.population)

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
