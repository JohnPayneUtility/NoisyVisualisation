"""Shared multi-objective infrastructure: the MO base class, its generation lifecycle and front stagnation.

Recording is the evaluation log's (noisyvis.tracking.mo_logger, on with log_evaluations); the MO runner
freezes it into the persistent mo_record read by noisyvis.results.mo_view.MORunView.
"""

# IMPORTS
import random

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Any

# deap.base is aliased: this module is itself the package's `base`
from deap import base as deap_base
from deap import creator
from deap import tools

from noisyvis.common.pareto import nondominated_mask

# ==============================
# Helper Functions
# ==============================

def front_sig(front_inds):
    # represent each solution as a tuple of ints, then frozenset for order-insensitivity
    sols = [tuple(int(x) for x in ind) for ind in (front_inds or [])]
    return frozenset(sols)

def front_signature(individuals):
    """
    Genotype set of the noisy non-dominated members of `individuals`: front_sig of the individuals
    tools.ParetoFront would keep, computed without building or deep-copying a ParetoFront.
    """
    if not individuals:
        return frozenset()
    mask = nondominated_mask([ind.fitness.wvalues for ind in individuals])
    return front_sig([ind for ind, keep in zip(individuals, mask) if keep])

def plain_fitness_function(fitness_function):
    """
    The (fitness function, keyword parameters) pair with plain-Python parameters. Hydra's instantiate hands
    the pair over as OmegaConf containers (the knapsack items_dict as a DictConfig), which are slow to read
    on every evaluation; they are converted once, with OmegaConf.to_container, keeping exactly the values
    the container holds (the evaluator results are unchanged). Anything else is returned as given.
    """
    if fitness_function is None:
        return None
    fn, params = fitness_function
    if type(params).__module__.startswith("omegaconf."):
        from omegaconf import OmegaConf  # only reached on the Hydra path, where omegaconf is loaded
        params = OmegaConf.to_container(params, resolve=True)
    return (fn, params)

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
    ref_point: Optional[list[Any]] = None
    # Print a progress line every verbose_rate generations (0: never); see _report_progress.
    verbose_rate: int = 0
    # Log every genuine evaluation (x, x~, f(x), y) and tag individuals with their provenance
    # (noisyvis.tracking.mo_logger). Observational only; requires an evaluator that reports log_mo_eval.
    # Off: optimisation only, nothing recorded. The MO runner turns it on.
    log_evaluations: bool = False

    def __post_init__(self):
        self.stop_trigger = ''
        self.fitness_function = plain_fitness_function(self.fitness_function)
        # Front stagnation (the no_improvement stop), owned by the algorithm, not the recorder:
        # the genotype set of the noisy non-dominated front at the last observation, and for how many
        # consecutive observations it has been unchanged (None before the first observation).
        self._front_signature = None
        self._front_unchanged_gens = None
        self._front_size = None  # members of that front (a progress-report detail)
        self.seed_signature = random.randint(0, 10**6)
        # Run-scoped evaluation log (None when logging is off). Imported here, not at module level, so
        # the module loads only for runs that log.
        self.eval_log = None
        if self.log_evaluations:
            from noisyvis.tracking.mo_logger import MOEvaluationLogger
            self.eval_log = MOEvaluationLogger(self.opt_weights, self.ref_point)

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
        self.toolbox.register("evaluate", self._evaluate_and_track)

    def _evaluate_and_track(self, ind):
        """
        toolbox.evaluate: every genuine evaluation goes through here, and the algorithm receives only
        the observed objective vector y. With logging on, the evaluation is logged exactly once and the
        individual is tagged with its provenance before the caller assigns fitness.values = y.
        """
        fn, kwargs = self.fitness_function
        if self.eval_log is None:
            return fn(ind, **kwargs)
        observed, eval_id = self.eval_log.evaluate(fn, ind, kwargs)
        self.eval_log.tag(ind, eval_id)
        return observed

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
            if self._front_unchanged_gens is not None:
                if self._front_unchanged_gens >= int(self.stop_without_improvement_in_gens):
                    self.stop_trigger = 'no_improvement'
                    return True
        return False

    def run(self):
        """
        Run the algorithm using the common loop logic. Generation 0 is the evaluated initial
        population the constructor built: it is observed once, before the first stop check, with no
        extra evaluation or RNG draw. Generation g >= 1 is the state after the g-th update.
        """
        self._observe_generation()
        while not self.stop_condition():
            self.gens += 1
            self.perform_generation()
            self._observe_generation()

    def _observe_generation(self):
        """
        Observe the current population, for every algorithm alike: update the algorithm's own
        front-stagnation state, then (with evaluation logging on) let the evaluation log record this
        generation's boundary and current fronts, then report progress. The current fronts and the
        no_improvement stop describe the solutions the algorithm currently holds; algorithm-specific
        state (such as MoUMDA_ParetoArchive's internal archive) is never the source. Stopping never
        depends on the log; it reads only the state updated here.
        """
        self._update_front_stagnation(self.population)
        if self.eval_log is not None:
            self.eval_log.observe_generation(self.gens, self.population)
        self._report_progress()

    def _update_front_stagnation(self, population):
        """
        no_improvement bookkeeping: the first observation sets the counter to 1; an unchanged
        genotype set of the population's noisy non-dominated front adds 1; a changed one resets it to 1.
        """
        mask = nondominated_mask([ind.fitness.wvalues for ind in population]) if population else []
        front = [ind for ind, keep in zip(population, mask) if keep]
        signature = front_sig(front)
        if self._front_unchanged_gens is not None and signature == self._front_signature:
            self._front_unchanged_gens += 1
        else:
            self._front_unchanged_gens = 1
        self._front_signature = signature
        self._front_size = len(front)

    def _report_progress(self):
        """
        Every verbose_rate generations, one line from the algorithm's own state: the generation, the
        cumulative evaluations, the current noisy front's size (non-dominated members and distinct
        genotypes) and whether its genotype set changed at this generation. Draws no RNG.
        """
        if not self.verbose_rate or self.gens % self.verbose_rate:
            return
        changed = "changed" if self._front_unchanged_gens == 1 else f"unchanged for {self._front_unchanged_gens}"
        print(f"[SeedSig {self.seed_signature}] gen {self.gens} | evals {self.evals} | noisy front "
              f"{self._front_size} members, {len(self._front_signature)} genotypes | {changed}", flush=True)
