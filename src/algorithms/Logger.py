# IMPORTS
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
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

        Records every adoption of a solution as a separate representative entry,
        including revisits. sol1 -> sol2 -> sol1 produces three entries.
        solution_iterations[i] is the number of consecutive generations for visit i.
        """
        if hasattr(self, '_trajectory_cache') and self._trajectory_cache is not None:
            return self._trajectory_cache

        representative_solutions = []
        representative_true_fitnesses = []
        representative_noisy_fitnesses = []
        representative_noisy_solutions = []
        sol_iterations = []
        sol_evals = []
        transitions = []
        visit_start_gens = []
        visit_end_gens = []

        current_sol = None
        current_count = 0
        current_visit_start_idx = 0
        evals_before_visit = 0  # cumulative evals at the end of the previous visit

        for idx, gen in enumerate(self.generations):
            solution_tuple = tuple(gen.best_solution)
            if current_sol is None:
                # First generation — start first visit
                current_sol = solution_tuple
                current_count = 1
                current_visit_start_idx = idx
                evals_before_visit = 0
                representative_solutions.append(gen.best_solution)
                representative_true_fitnesses.append(gen.true_fitness)
                representative_noisy_fitnesses.append(gen.best_fitness)
                representative_noisy_solutions.append(gen.noisy_solution)
            elif solution_tuple == current_sol:
                # Same solution — continue current visit
                current_count += 1
            else:
                # Solution changed — close current visit, open new one
                sol_iterations.append(current_count)
                end_evals = self.generations[idx - 1].evals_so_far or 0
                sol_evals.append(end_evals - evals_before_visit)
                evals_before_visit = end_evals
                visit_start_gens.append(self.generations[current_visit_start_idx].generation)
                visit_end_gens.append(self.generations[idx - 1].generation)
                transitions.append((current_sol, solution_tuple))
                current_sol = solution_tuple
                current_count = 1
                current_visit_start_idx = idx
                representative_solutions.append(gen.best_solution)
                representative_true_fitnesses.append(gen.true_fitness)
                representative_noisy_fitnesses.append(gen.best_fitness)
                representative_noisy_solutions.append(gen.noisy_solution)

        # Close the last visit
        if current_sol is not None:
            sol_iterations.append(current_count)
            end_evals = self.generations[-1].evals_so_far or 0
            sol_evals.append(end_evals - evals_before_visit)
            visit_start_gens.append(self.generations[current_visit_start_idx].generation)
            visit_end_gens.append(self.generations[-1].generation)

        # Build noisy sol variants and estimated fits per visit (filtered to visit window)
        representative_noisy_sols = []
        representative_noisy_variant_fitnesses = []
        representative_estimated_true_fits_whenadopted = []
        representative_estimated_true_fits_whendiscarded = []
        representative_count_estimated_fits_whenadopted = []
        representative_count_estimated_fits_whendiscarded = []

        for i, sol in enumerate(representative_solutions):
            key = tuple(sol)
            visit_start = visit_start_gens[i]
            visit_end = visit_end_gens[i]

            if key in self.evaluations:
                visit_evals = [r for r in self.evaluations[key]
                               if visit_start <= r.generation <= visit_end]
                noisy_sols = [r.noisy_sol for r in visit_evals]
                noisy_fits = [r.noisy_fitness for r in visit_evals]
                # whenadopted: evals recorded at the moment of adoption
                adopted_fits = [r.noisy_fitness for r in visit_evals
                                if r.generation <= visit_start]
                # whendiscarded: all evals during the visit
                discarded_fits = noisy_fits
            else:
                noisy_sols = []
                noisy_fits = []
                adopted_fits = []
                discarded_fits = []

            representative_noisy_sols.append(noisy_sols)
            representative_noisy_variant_fitnesses.append(noisy_fits)
            representative_estimated_true_fits_whenadopted.append(
                median(adopted_fits) if adopted_fits else None)
            representative_estimated_true_fits_whendiscarded.append(
                median(discarded_fits) if discarded_fits else None)
            representative_count_estimated_fits_whenadopted.append(len(adopted_fits))
            representative_count_estimated_fits_whendiscarded.append(len(discarded_fits))

        # Cache all computed data
        self._trajectory_cache = {
            'representative_solutions': representative_solutions,
            'representative_true_fitnesses': representative_true_fitnesses,
            'representative_noisy_fitnesses': representative_noisy_fitnesses,
            'representative_noisy_solutions': representative_noisy_solutions,
            'solution_iterations': sol_iterations,
            'solution_evals': sol_evals,
            'solution_transitions': transitions,
            'representative_noisy_sols': representative_noisy_sols,
            'noisy_variant_fitnesses': representative_noisy_variant_fitnesses,
            'representative_estimated_true_fits_whenadopted': representative_estimated_true_fits_whenadopted,
            'representative_estimated_true_fits_whendiscarded': representative_estimated_true_fits_whendiscarded,
            'count_estimated_fits_whenadopted': representative_count_estimated_fits_whenadopted,
            'count_estimated_fits_whendiscarded': representative_count_estimated_fits_whendiscarded,
        }
        return self._trajectory_cache

    @property
    def representative_solutions(self) -> List[List]:
        """Representative solutions in order of adoption, including revisits."""
        return self._build_trajectory_cache()['representative_solutions']

    @property
    def representative_true_fitnesses(self) -> List[float]:
        """True (noise-free) fitness for each representative solution."""
        return self._build_trajectory_cache()['representative_true_fitnesses']

    @property
    def representative_noisy_fitnesses(self) -> List[float]:
        """Noisy (observed) fitness for each representative solution."""
        return self._build_trajectory_cache()['representative_noisy_fitnesses']

    @property
    def representative_noisy_solutions(self) -> List[List]:
        """Noisy (perturbed) solution at the time each representative solution was adopted."""
        return self._build_trajectory_cache()['representative_noisy_solutions']

    @property
    def solution_iterations(self) -> List[int]:
        """Consecutive generation count for each visit (entry in representative_solutions)."""
        return self._build_trajectory_cache()['solution_iterations']

    @property
    def solution_evals(self) -> List[int]:
        """Number of algorithm fitness evaluations consumed during each visit (entry in representative_solutions)."""
        return self._build_trajectory_cache()['solution_evals']

    @property
    def solution_transitions(self) -> List[Tuple[Tuple, Tuple]]:
        """List of (from_solution, to_solution) tuples for each transition."""
        return self._build_trajectory_cache()['solution_transitions']

    @property
    def representative_noisy_sols(self) -> List[List[List]]:
        """
        Noisy solution variants for each representative solution from evaluation
        records during that visit.

        For posterior noise: noisy_sol == true_sol (same solution)
        For prior noise: noisy_sol != true_sol (perturbed solution)
        """
        return self._build_trajectory_cache()['representative_noisy_sols']

    @property
    def noisy_variant_fitnesses(self) -> List[List[float]]:
        """
        Noisy fitness values for each variant in representative_noisy_sols.
        Parallel structure: noisy_variant_fitnesses[i][j] is the noisy fitness
        for representative_noisy_sols[i][j].
        """
        return self._build_trajectory_cache()['noisy_variant_fitnesses']

    @property
    def representative_estimated_true_fits_whenadopted(self) -> List[Optional[float]]:
        """
        Estimated true fitness for each representative solution, computed as the median
        of noisy evaluations up to the generation it was adopted as best.
        """
        return self._build_trajectory_cache()['representative_estimated_true_fits_whenadopted']

    @property
    def representative_estimated_true_fits_whendiscarded(self) -> List[Optional[float]]:
        """
        Estimated true fitness for each representative solution, computed as the median
        of all noisy evaluations during the visit.
        """
        return self._build_trajectory_cache()['representative_estimated_true_fits_whendiscarded']

    @property
    def count_estimated_fits_whenadopted(self) -> List[int]:
        """Number of noisy evaluations used to compute each whenadopted estimate."""
        return self._build_trajectory_cache()['count_estimated_fits_whenadopted']

    @property
    def count_estimated_fits_whendiscarded(self) -> List[int]:
        """Number of noisy evaluations used to compute each whendiscarded estimate."""
        return self._build_trajectory_cache()['count_estimated_fits_whendiscarded']

    def get_trajectory_data(self):
        """
        Extract trajectory data for plotting (kept for backwards compatibility).

        Returns:
            representative_solutions, representative_true_fitnesses,
            representative_noisy_fitnesses, solution_iterations, solution_transitions
        """
        return (self.representative_solutions, self.representative_true_fitnesses,
                self.representative_noisy_fitnesses, self.solution_iterations,
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
