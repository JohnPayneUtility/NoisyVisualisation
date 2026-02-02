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
    noisy_solution: List = None  # Perturbed solution (prior noise) or same as best_solution (posterior)
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

    # Cache for trajectory data (invalidated on clear())
    _trajectory_cache: Optional[Dict] = field(default=None, init=False, repr=False)

    def log_generation(self, generation: int, population, best_solution,
                       best_fitness: float, true_fitness: float,
                       noisy_solution=None, evals: int = None):
        """Log the state at the end of a generation."""
        record = GenerationRecord(
            generation=generation,
            population=[ind[:] for ind in population],
            best_solution=best_solution[:],
            best_fitness=best_fitness,
            true_fitness=true_fitness,
            noisy_solution=list(noisy_solution) if noisy_solution is not None else None,
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
    # Trajectory Data Extraction (with caching)
    # ==============================

    def _build_trajectory_cache(self):
        """
        Build and cache trajectory data. Called once, then cached.
        Cache is invalidated when clear() is called.
        """
        if hasattr(self, '_trajectory_cache') and self._trajectory_cache is not None:
            return self._trajectory_cache

        unique_solutions = []
        unique_true_fitnesses = []
        unique_noisy_fitnesses = []
        seen_solutions = {}

        unique_noisy_solutions = []

        for gen in self.generations:
            solution_tuple = tuple(gen.best_solution)
            if solution_tuple not in seen_solutions:
                seen_solutions[solution_tuple] = 1
                unique_solutions.append(gen.best_solution)
                unique_true_fitnesses.append(gen.true_fitness)
                unique_noisy_fitnesses.append(gen.best_fitness)
                unique_noisy_solutions.append(gen.noisy_solution)
            else:
                seen_solutions[solution_tuple] += 1

        # Create iteration counts in order of appearance
        iteration_counts = [seen_solutions[tuple(sol)] for sol in unique_solutions]

        # Build transitions
        transitions = []
        for i in range(1, len(unique_solutions)):
            prev_solution = tuple(unique_solutions[i - 1])
            current_solution = tuple(unique_solutions[i])
            transitions.append((prev_solution, current_solution))

        # Build noisy sol variants and their fitnesses from evaluation records
        noisy_sols_per_solution = []
        noisy_variant_fitnesses_per_solution = []
        for sol in unique_solutions:
            key = tuple(sol)
            if key in self.evaluations:
                noisy_sols = [record.noisy_sol for record in self.evaluations[key]]
                noisy_fits = [record.noisy_fitness for record in self.evaluations[key]]
            else:
                noisy_sols = []
                noisy_fits = []
            noisy_sols_per_solution.append(noisy_sols)
            noisy_variant_fitnesses_per_solution.append(noisy_fits)

        # Cache all computed data
        self._trajectory_cache = {
            'unique_solutions': unique_solutions,
            'unique_true_fitnesses': unique_true_fitnesses,
            'unique_noisy_fitnesses': unique_noisy_fitnesses,
            'unique_noisy_solutions': unique_noisy_solutions,
            'solution_iterations': iteration_counts,
            'solution_transitions': transitions,
            'unique_noisy_sols': noisy_sols_per_solution,
            'noisy_variant_fitnesses': noisy_variant_fitnesses_per_solution,
        }
        return self._trajectory_cache

    @property
    def unique_solutions(self) -> List[List]:
        """List of unique best solutions in order of first appearance."""
        return self._build_trajectory_cache()['unique_solutions']

    @property
    def unique_true_fitnesses(self) -> List[float]:
        """True (noise-free) fitness for each unique solution."""
        return self._build_trajectory_cache()['unique_true_fitnesses']

    @property
    def unique_noisy_fitnesses(self) -> List[float]:
        """Noisy (observed) fitness for each unique solution."""
        return self._build_trajectory_cache()['unique_noisy_fitnesses']

    @property
    def unique_noisy_solutions(self) -> List[List]:
        """Noisy (perturbed) solution for each unique solution, from when it first became best."""
        return self._build_trajectory_cache()['unique_noisy_solutions']

    @property
    def solution_iterations(self) -> List[int]:
        """Count of how many generations each unique solution was best."""
        return self._build_trajectory_cache()['solution_iterations']

    @property
    def solution_transitions(self) -> List[Tuple[Tuple, Tuple]]:
        """List of (from_solution, to_solution) tuples for each transition."""
        return self._build_trajectory_cache()['solution_transitions']

    @property
    def unique_noisy_sols(self) -> List[List[List]]:
        """
        Noisy solution variants for each unique solution from evaluation records.

        For posterior noise: noisy_sol == true_sol (same solution)
        For prior noise: noisy_sol != true_sol (perturbed solution)
        """
        return self._build_trajectory_cache()['unique_noisy_sols']

    @property
    def noisy_variant_fitnesses(self) -> List[List[float]]:
        """
        Noisy fitness values for each variant in unique_noisy_sols.
        Parallel structure: noisy_variant_fitnesses[i][j] is the noisy fitness
        for unique_noisy_sols[i][j].
        """
        return self._build_trajectory_cache()['noisy_variant_fitnesses']

    def get_trajectory_data(self):
        """
        Extract trajectory data for plotting (kept for backwards compatibility).

        Returns:
            unique_solutions, unique_true_fitnesses, unique_noisy_fitnesses,
            solution_iterations, solution_transitions
        """
        return (self.unique_solutions, self.unique_true_fitnesses,
                self.unique_noisy_fitnesses, self.solution_iterations,
                self.solution_transitions)

    # ==============================
    # Utility Methods
    # ==============================

    def clear(self):
        """Clear all logged data and invalidate cache."""
        self.generations.clear()
        self.evaluations.clear()
        self.current_generation = 0
        self._trajectory_cache = None  # Invalidate cache

    def __len__(self):
        """Return number of generations logged."""
        return len(self.generations)
