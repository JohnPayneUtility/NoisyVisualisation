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

# ==============================
# Description
# ==============================



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
# Helper Functions
# ==============================

def record_population_state(data, population, toolbox, true_fitness_function):
    """
    Record the current state of the population.
    """
    all_generations, best_solutions, best_fitnesses, true_fitnesses = data

    # Record a snapshot of the current population
    all_generations.append([ind[:] for ind in population])
    
    # Identify the best individual in the current population
    best_individual = tools.selBest(population, 1)[0]
    best_solutions.append(toolbox.clone(best_individual))
    best_fitnesses.append(best_individual.fitness.values[0])
    
    # If provided, record the true (noise-free) fitness
    if true_fitness_function is not None:
        true_fit = true_fitness_function[0](best_individual, **true_fitness_function[1])
        true_fitnesses.append(true_fit[0])
    else:
        true_fitnesses.append(best_individual.fitness.values[0])

def extract_trajectory_data(best_solutions, best_fitnesses, true_fitnesses):
    # Extract unique solutions and their corresponding fitness values
    unique_solutions = []
    unique_fitnesses = []
    noisy_fitnesses = []
    solution_iterations = []
    seen_solutions = {}

    for solution, fitness, true_fitness in zip(best_solutions, best_fitnesses, true_fitnesses):
        # Convert solution to a tuple to make it hashable
        solution_tuple = tuple(solution)
        if solution_tuple not in seen_solutions:
            seen_solutions[solution_tuple] = 1
            unique_solutions.append(solution)
            unique_fitnesses.append(true_fitness)
            noisy_fitnesses.append(fitness)
        else:
            seen_solutions[solution_tuple] += 1

    # Create a list of iteration counts for each unique solution
    for solution in unique_solutions:
        solution_tuple = tuple(solution)
        solution_iterations.append(seen_solutions[solution_tuple])

    return unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations

def extract_transitions(unique_solutions):
    # Extract transitions between solutions over generations
    transitions = []

    for i in range(1, len(unique_solutions)):
        prev_solution = tuple(unique_solutions[i - 1])
        current_solution = tuple(unique_solutions[i])
        transitions.append((prev_solution, current_solution))

    return transitions

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
    true_fitness_function: Optional[Tuple[Callable, dict]] = None
    
    # Create lists to store data, seperate for each instance
    all_generations: List[List[Any]] = field(default_factory=list)
    best_solutions: List[Any] = field(default_factory=list)
    best_fitnesses: List[float] = field(default_factory=list)
    true_fitnesses: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.stop_trigger = ''
        self.seed_signature = random.randint(0, 10**6)
        self.data = [self.all_generations, self.best_solutions, self.best_fitnesses, self.true_fitnesses]

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
        return False
    
    def run(self):
        """Run the algorithm using the common loop logic."""
        while not self.stop_condition():
            self.gens += 1
            self.perform_generation()
            self.record_state(self.population)

    def record_state(self, population):
        #Record the current population state.
        record_population_state(self.data, population, self.toolbox, self.true_fitness_function)

    def get_classic_data(self):
        return self.all_generations, self.best_solutions, self.best_fitnesses, self.true_fitnesses
    
    def get_solution_data(self):
        return self.best_solutions, self.best_fitnesses, self.true_fitnesses
    
    def get_trajectory_data(self):
        unique_sols, unique_fits, noisy_fitnesses, sol_iterations = extract_trajectory_data(self.best_solutions, self.best_fitnesses, self.true_fitnesses)
        sol_transitions = extract_transitions(unique_sols)
        return unique_sols, unique_fits, noisy_fitnesses, sol_iterations, sol_transitions

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
            if offspring1.fitness.values[0] > offspring2.fitness.values[0]:  # Adjust for minimization
                offspring.append(offspring1)
            else:
                offspring.append(offspring2)

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
    
    mutation_rate = 1 / 100  # for a 100-bit solution
    mutate_function = (tools.mutFlipBit, {'indpb': mutation_rate})
    # mutate_function = (mutSwapBit, {'indpb': mutation_rate})

    from FitnessFunctions import OneMax_fitness
    fitness_function = (OneMax_fitness, {'noise_intensity': 0})
    true_fitness_function = (OneMax_fitness, {'noise_intensity': 0})

    ss = np.zeros(100, dtype=int)


    base_params = {
        'sol_length': 100,              # Length of the solution
        'opt_weights': (1.0,),           # Maximization problem
        'eval_limit': 10e3,               # Maximum fitness evaluations
        'attr_function': binary_attribute,
        'fitness_function': fitness_function,
        'true_fitness_function': true_fitness_function,
        'starting_solution': None
        # 'target_stop': 80
    }

    # algo = MuPlusLamdaEA(mu=5, lam=1, mutate_function=mutate_function, **base_params)
    algo = UMDA(pop_size=100, **base_params)

    # algo.run()
    # all_gens, best_sols, best_fits, true_fits = algo.get_classic_data()
    
    # # (Optional) Print the final best fitness
    # print("Algo name:", algo.name)
    # print("Final best fitness:", true_fits[-1])
    # # # print("First best sol:", best_sols[1])
    # # # print("Final best sol:", best_sols[-1])

