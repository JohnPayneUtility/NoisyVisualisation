# IMPORTS
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# ==============================
# Module-level Logger Access
# ==============================

_active_logger = None

def set_active_logger(logger):
    """Set the active logger for fitness functions to access."""
    global _active_logger
    _active_logger = logger

def get_active_logger():
    """Get the active logger. Returns None if no logger is set."""
    return _active_logger

def clear_active_logger():
    """Clear the active logger."""
    global _active_logger
    _active_logger = None

# ==============================
# Data Records
# ==============================

@dataclass
class GenerationRecord:
    """Record of a single generation's state."""
    generation: int
    population: List[List]
    best_solution: List
    best_fitness: float          # Observed (possibly noisy)
    true_fitness: float          # Noise-free
    evals_so_far: int = None

@dataclass
class EvaluationRecord:
    """Record of a single noisy evaluation."""
    true_sol: List           # Original solution submitted for evaluation
    noisy_sol: List          # Noisy/perturbed solution (same as true_sol for posterior noise)
    true_fitness: float      # Fitness without noise
    noisy_fitness: float     # Fitness with noise applied
    generation: int = None

# ==============================
# Experiment Logger
# ==============================

@dataclass
class ExperimentLogger:
    """
    Centralised logging for algorithm runs.

    Stores generation-level data and evaluation-level data (for prior noise tracking).
    Provides property accessors for backwards compatibility with existing code.
    """

    # Generation-level data
    generations: List[GenerationRecord] = field(default_factory=list)

    # Evaluation-level data
    # Maps original solution (as tuple) -> list of evaluation records
    evaluations: Dict[Tuple, List[EvaluationRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Track current generation for evaluation logging
    current_generation: int = 0

    def log_generation(self, generation: int, population, best_solution,
                       best_fitness: float, true_fitness: float, evals: int = None):
        """Log the state at the end of a generation."""
        record = GenerationRecord(
            generation=generation,
            population=[ind[:] for ind in population],
            best_solution=best_solution[:],
            best_fitness=best_fitness,
            true_fitness=true_fitness,
            evals_so_far=evals
        )
        self.generations.append(record)
        self.current_generation = generation

    def log_noisy_eval(self, original, noisy, true_fitness, noisy_fitness, generation: int = None):
        """
        Log a noisy evaluation.

        Args:
            original: The original solution submitted for evaluation
            noisy: The noisy/perturbed version that was actually evaluated
                   (same as original for posterior noise, different for prior noise)
            true_fitness: The fitness without noise applied
            noisy_fitness: The fitness with noise applied
            generation: Optional generation number (uses current_generation if not provided)
        """
        if generation is None:
            generation = self.current_generation

        key = tuple(original)
        record = EvaluationRecord(
            true_sol=list(original),
            noisy_sol=list(noisy),
            true_fitness=true_fitness,
            noisy_fitness=noisy_fitness,
            generation=generation
        )
        self.evaluations[key].append(record)

    # ==============================
    # Evaluation Accessors
    # ==============================

    def get_noisy_variants(self, solution) -> List[EvaluationRecord]:
        """Get all noisy evaluation records for a particular solution."""
        return self.evaluations[tuple(solution)]

    def get_all_evaluated_solutions(self) -> List[Tuple]:
        """Get all unique original solutions that were evaluated with noise."""
        return list(self.evaluations.keys())

    def get_all_noisy_evals(self) -> List[EvaluationRecord]:
        """Get a flat list of all noisy evaluation records."""
        all_evals = []
        for records in self.evaluations.values():
            all_evals.extend(records)
        return all_evals

    # ==============================
    # Generation Data Accessors (backwards compatibility)
    # ==============================

    @property
    def all_generations(self) -> List[List]:
        """Get all population snapshots."""
        return [g.population for g in self.generations]

    @property
    def best_solutions(self) -> List[List]:
        """Get best solution from each generation."""
        return [g.best_solution for g in self.generations]

    @property
    def best_fitnesses(self) -> List[float]:
        """Get best fitness (possibly noisy) from each generation."""
        return [g.best_fitness for g in self.generations]

    @property
    def true_fitnesses(self) -> List[float]:
        """Get true (noise-free) fitness from each generation."""
        return [g.true_fitness for g in self.generations]

    def get_best_per_generation(self) -> List[Tuple[List, float, float]]:
        """
        Get best individual with both fitnesses for each generation.

        Returns:
            List of tuples: (best_solution, noisy_fitness, true_fitness)
            where best is judged by noisy_fitness (what the algorithm saw).
        """
        return [(g.best_solution, g.best_fitness, g.true_fitness) for g in self.generations]

    # ==============================
    # Trajectory Data Extraction
    # ==============================

    def get_trajectory_data(self):
        """
        Extract trajectory data for plotting.

        Returns:
            unique_solutions: List of unique best solutions in order of first appearance
            unique_fitnesses: True fitness for each unique solution
            noisy_fitnesses: Noisy fitness for each unique solution
            solution_iterations: How many generations each solution was best
            solution_transitions: List of (from_solution, to_solution) tuples
        """
        unique_solutions = []
        unique_fitnesses = []
        noisy_fitnesses = []
        solution_iterations = []
        seen_solutions = {}

        for gen in self.generations:
            solution_tuple = tuple(gen.best_solution)
            if solution_tuple not in seen_solutions:
                seen_solutions[solution_tuple] = 1
                unique_solutions.append(gen.best_solution)
                unique_fitnesses.append(gen.true_fitness)
                noisy_fitnesses.append(gen.best_fitness)
            else:
                seen_solutions[solution_tuple] += 1

        # Create iteration counts for each unique solution
        for solution in unique_solutions:
            solution_tuple = tuple(solution)
            solution_iterations.append(seen_solutions[solution_tuple])

        # Extract transitions
        solution_transitions = []
        for i in range(1, len(unique_solutions)):
            prev_solution = tuple(unique_solutions[i - 1])
            current_solution = tuple(unique_solutions[i])
            solution_transitions.append((prev_solution, current_solution))

        return unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, solution_transitions

    # ==============================
    # Utility Methods
    # ==============================

    def clear(self):
        """Clear all logged data."""
        self.generations.clear()
        self.evaluations.clear()
        self.current_generation = 0

    def __len__(self):
        """Return number of generations logged."""
        return len(self.generations)
