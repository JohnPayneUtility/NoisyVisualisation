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
import optuna

from .Logger import ExperimentLogger, set_active_logger, clear_active_logger

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

def umda_update_full(len_sol, population, pop_size, select_size, toolbox):
    # Select from population
    selected_population = tools.selBest(population, select_size)
    # Determine the data type of the genes from the first individual in the population
    gene_type = type(population[0][0])
    # Calculate marginal probabilities for binary solutions (assumes binary values are either 0 or 1)
    if gene_type == int:
        probabilities = np.mean(selected_population, axis=0)

        new_solutions = []
        for _ in range(pop_size):
            new_solution = np.random.rand(len_sol) < probabilities
            new_solution = creator.Individual(new_solution.astype(int).tolist())  # Create as DEAP Individual
            new_solutions.append(new_solution)

    # For float-based solutions, calculate mean and standard deviation
    elif gene_type == float:
        selected_array = np.array(selected_population)
        means = np.mean(selected_array, axis=0)
        stds = np.std(selected_array, axis=0)
        
        new_solutions = []
        for _ in range(pop_size):
            new_solution = np.random.normal(means, stds, len_sol)
            new_solution = creator.Individual(new_solution.tolist())  # Create as DEAP Individual
            new_solutions.append(new_solution)
    else:
        raise ValueError("Unsupported gene type. Expected int or float.")
    return new_solutions

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
    attr_function: Optional[Callable] = None
    fitness_function: Optional[Tuple[Callable, dict]] = None
    starting_solution: Optional[List[Any]] = None
    progress_print_interval: Optional[int] = None  # print update every N evals; None = silent
    record_population: bool = False  # store full population snapshots (high RAM; off by default)
    nvme_path: Optional[str] = None  # path to NVMe-backed LMDB dir; None = in-memory

    # Logger — created in __post_init__ so nvme_path can be passed through
    logger: Optional[ExperimentLogger] = None

    def __post_init__(self):
        self.stop_trigger = ''
        self.seed_signature = random.randint(0, 10**6)

        # Create logger with the chosen fit-history backend
        if self.logger is None:
            self.logger = ExperimentLogger(nvme_path=self.nvme_path)

        # Propagate population-recording preference to the logger
        self.logger.record_population = self.record_population

        # Set this logger as the active logger for fitness functions to access
        set_active_logger(self.logger)

        # Fitness and individual creators
        # Check if CustomFitness exists with matching weights; recreate if weights differ
        if hasattr(creator, "CustomFitness"):
            if creator.CustomFitness.weights != self.opt_weights:
                del creator.CustomFitness
                del creator.Individual
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
        if self.target_stop is not None and self.logger._last_true_fitness is not None:
            current_fitness = self.logger._last_true_fitness
            if self.opt_weights[0] > 0:
                reached = current_fitness >= self.target_stop
            else:
                reached = current_fitness <= self.target_stop
            if reached:
                self.stop_trigger = 'target_reached'
                return True
        if self.gen_limit is not None and self.gens >= self.gen_limit:
            self.stop_trigger = 'gen_limit'
            return True
        return False

    def _print_progress(self):
        """Print a one-line progress update."""
        best_true = self.logger._last_true_fitness
        pct = f"  ({100 * self.evals / self.eval_limit:.1f}% of limit)" if self.eval_limit else ""
        fit_str = f"{best_true:.4g}" if best_true is not None else "n/a"
        print(f"[Evals: {self.evals:,}]{pct}  Best true fitness: {fit_str}")

    def run(self):
        """Run the algorithm using the common loop logic."""
        next_print = self.progress_print_interval  # None means never print
        while not self.stop_condition():
            self.gens += 1
            self.logger.current_generation = self.gens
            self.perform_generation()
            self.record_state(self.population)
            if next_print is not None and self.evals >= next_print:
                self._print_progress()
                next_print += self.progress_print_interval

    def record_state(self, population):
        """Record the current population state using the logger."""
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]  # noisy fitness seen by algorithm

        # true_fitness and noisy_solution are derived internally by log_generation
        # from the current-generation eval buffer (_gen_evals)
        self.logger.log_generation(
            generation=self.gens,
            population=population,
            best_solution=best_individual,
            best_fitness=best_fitness,
            evals=self.evals
        )

# ==============================
# Evolutionary Algorithm Subclasses
# ==============================

class MuPlusLamdaEA(OptimisationAlgorithm):
    def __init__(self, 
                 mu: int,
                 lam: int, 
                 mutate_function: str, 
                 mutate_params: dict,
                 **kwargs): # other parameters passed to the base class
        
        # Initialize common components via the base class
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.mu = mu
        self.lam = lam
        self.name = f'({mu}+{lam})EA'
        if mu > 1 and lam > 1:
            self.type = '(mu+lamda)EA'
        elif mu > 1:
            self.type = '(mu+1)EA'
        elif lam > 1:
            self.type = '(1+lam)EA'
        else: self.type = '(1+1)EA'

        # Register the mutation operator in the toolbox
        if mutate_function == None or mutate_function == "probFlipBit":
            self.toolbox.register("mutate", lambda ind: tools.mutFlipBit(ind, **mutate_params))
        elif mutate_function == "probSwapBit":
            self.toolbox.register("mutate", lambda ind: mutSwapBit(ind, **mutate_params))
        elif mutate_function == "mutGaussian":
            self.toolbox.register("mutate", lambda ind: tools.mutGaussian(ind, **mutate_params))
        # self.toolbox.register("mutate", lambda ind: mutate_function(ind, **mutate_params))

        # Create the initial population of size mu
        self.initialise_population(self.mu)
        self.record_state(self.population)

    def perform_generation(self):
        """Perform generation of (mu + lambda) Evolutionary Algorithm"""
        # Generate offspring 
        for _ in range(self.lam):
            parent = random.choice(self.population)              
            offspring = self.toolbox.clone(parent)
            offspring, = self.toolbox.mutate(offspring)
            
            # Evaluate the offspring
            del offspring.fitness.values
            offspring.fitness.values = self.toolbox.evaluate(offspring)
            self.evals += 1

            self.population.append(offspring)
        # Update population
        if self.mu == 1 and self.lam == 1:
            # Classic (1+1)EA: accept offspring if not worse, allowing lateral moves on plateaus
            parent, offspring = self.population[0], self.population[1]
            self.population = [offspring if offspring.fitness >= parent.fitness else parent]
        else:
            self.population = tools.selBest(self.population, self.mu)

class PCEA(OptimisationAlgorithm):
    def __init__(self, 
                 pop_size: int,
                 **kwargs): # other parameters passed to the base class
        
        # Initialize common components via the base class
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.pop_size = pop_size
        self.name = f'PCEA(p={pop_size})'
        self.type = 'PCEA'

        # Register the mutation operator in the toolbox
        self.toolbox.register("mate", complementary_crossover)

        # Create the initial population of size mu
        self.initialise_population(self.pop_size)
        self.record_state(self.population)

    def perform_generation(self):
        """Perform generation of PCEA Evolutionary Algorithm"""
        # Generate offspring 
        offspring = []
        for _ in range(len(self.population)):
            parent1, parent2 = random.sample(self.population, 2)
            offspring1, offspring2 = self.toolbox.mate(parent1, parent2)

            # Invalidate fitness to ensure re-evaluation if needed
            # del offspring1.fitness.values
            # del offspring2.fitness.values

            offspring1.fitness.values = self.toolbox.evaluate(offspring1)
            offspring2.fitness.values = self.toolbox.evaluate(offspring2)
            self.evals += 2

            # Select the fitter offspring and add to new population
            offspring.append(tools.selBest([offspring1, offspring2], 1)[0])

        self.population[:] = offspring # replace population

# ==============================
# Estimation of Distribution Algorithm Subclasses
# ==============================

class UMDA(OptimisationAlgorithm):
    def __init__(self, 
                 pop_size: int,
                 select_size: Optional[int] = None,
                 **kwargs): # other parameters passed to the base class
        
        # Initialize common components via the base class
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.pop_size = pop_size
        if select_size == None:
            self.select_size = int(self.pop_size/2)
        else: self.select_size = select_size
        self.name = f'UMDA(p={pop_size})'
        self.type = 'UMDA'

        # Create the initial population of size mu
        self.initialise_population(self.pop_size)
        self.record_state(self.population)

    def perform_generation(self):
        """Perform generation of UMDA Evolutionary Algorithm"""
        self.population = umda_update_full(self.sol_length, self.population, self.pop_size, self.select_size, self.toolbox)

        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        self.evals += self.pop_size

class CompactGA(OptimisationAlgorithm):
    def __init__(self, 
                 pop_size: int,
                 **kwargs):
        """
        Compact Genetic Algorithm.
        
        Parameters:
            cga_pop_size (int): The effective population size parameter used to
                                determine the update step (1/cga_pop_size).
            **kwargs: Other parameters passed to the base OptimisationAlgorithm.
        """
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.cga_pop_size = pop_size
        self.name = f"cGA(p={pop_size})"
        self.type = 'cGA'
        # Initialize the probability vector (one value per gene)
        self.p_vector = [0.5] * self.sol_length
        # Record the initial state by sampling a candidate solution.
        self.record_state([self.sample_solution()])

    def sample_solution(self):
        """
        Generate a candidate solution by rounding the probability vector.
        For example, if p >= 0.5, choose 1; otherwise, choose 0.
        """
        candidate_list = [1 if random.random() < p else 0 for p in self.p_vector]
        candidate = creator.Individual(candidate_list)
        candidate.fitness.values = self.toolbox.evaluate(candidate)
        return candidate

    def perform_generation(self):
        """
        Perform one generation of the compact GA.
        
        This involves:
          1. Sampling two individuals from the current probability vector.
          2. Evaluating their fitness.
          3. Determining the winner and loser.
          4. Updating the probability vector in each gene where they differ.
        """
        # Sample two individuals using the current probability vector.
        x = [1 if random.random() < p else 0 for p in self.p_vector]
        y = [1 if random.random() < p else 0 for p in self.p_vector]
        # Evaluate both individuals.
        fx = self.toolbox.evaluate(x)
        fy = self.toolbox.evaluate(y)
        self.evals += 2
        # Determine winner and loser.
        if fx[0] > fy[0]:
            winner, loser = x, y
        elif fy[0] > fx[0]:
            winner, loser = y, x
        else:
            # In case of a tie, choose randomly.
            if random.random() < 0.5:
                winner, loser = x, y
            else:
                winner, loser = y, x
        # Update each gene’s probability.
        update_step = 1.0 / self.cga_pop_size
        for i in range(self.sol_length):
            if winner[i] != loser[i]:
                if winner[i] == 1:
                    self.p_vector[i] = min(1.0, self.p_vector[i] + update_step)
                else:
                    self.p_vector[i] = max(0.0, self.p_vector[i] - update_step)
        # Optionally record a candidate solution for this generation.
        candidate = self.sample_solution()
        # We pass a list with the candidate to record_state (to mimic a population).
        self.population = [candidate]
    
    # def stop_condition(self) -> bool:
    #     """
    #     Stop if the probability vector has converged (all entries are 0 or 1)
    #     or if any base class stopping conditions are met.
    #     """
    #     if all(p in (0.0, 1.0) for p in self.p_vector):
    #         return True
    #     return super().stop_condition()

# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    """
    Example demonstrating the logging system.

    The logger builds trajectory data incrementally, storing one VisitRecord per
    solution change rather than a record for every generation.

    Key data access points:
    - logger.visits                              : list of VisitRecord (one per solution change)
    - logger.representative_solutions           : best solution at each visit
    - logger.representative_true_fitnesses      : true fitness at adoption per visit
    - logger.representative_noisy_fitnesses     : noisy fitness at adoption per visit
    - logger.solution_iterations                : generations per visit
    - logger.solution_evals                     : evaluations consumed per visit
    - logger.solution_transitions               : (from, to) solution pairs
    - logger.representative_estimated_true_fits_whenadopted   : median noisy fit at adoption
    - logger.representative_estimated_true_fits_whendiscarded : median noisy fit at discard
    """

    from .problems.FitnessFunctions import OneMax_fitness
    fitness_function = (OneMax_fitness, {'noise_intensity': 0})

    base_params = {
        'sol_length': 20,
        'opt_weights': (1.0,),
        'eval_limit': 1000,
        'attr_function': binary_attribute,
        'fitness_function': fitness_function,
    }

    algo = MuPlusLamdaEA(
        mu=1, lam=1,
        mutate_function="probFlipBit",
        mutate_params={'indpb': 0.05},
        **base_params
    )
    algo.run()

    print(f"\n=== Run Summary ===")
    print(f"Algorithm:       {algo.name}")
    print(f"Generations run: {len(algo.logger)}")
    print(f"Stop trigger:    {algo.stop_trigger}")
    print(f"Visits logged:   {len(algo.logger.visits)}")

    print(f"\n=== Trajectory Data ===")
    rep_sols, true_fits, noisy_fits, iterations, transitions = algo.logger.get_trajectory_data()
    print(f"Unique best solutions visited: {len(rep_sols)}")
    print(f"Transitions between solutions: {len(transitions)}")

    print(f"\nFirst 5 visits:")
    for i, (sol, true_f, noisy_f, iters) in enumerate(
        zip(rep_sols[:5], true_fits[:5], noisy_fits[:5], iterations[:5])
    ):
        print(f"  Visit {i}: true_fit={true_f:.2f}, noisy_fit={noisy_f:.2f}, iterations={iters}")

