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
class VisitRecord:
    """
    Compact record of a single visit to a solution.

    One record is written each time the best solution changes. For a run where the
    best solution changes N times, exactly N+1 VisitRecords are stored regardless
    of how many generations were run.
    """
    solution: List
    true_fitness: float          # Noise-free fitness at adoption
    noisy_fitness: float         # Observed (noisy) fitness at adoption
    noisy_solution: Optional[List]  # Perturbed solution (prior noise) or original (posterior)
    iterations: int              # Consecutive generations this solution was best
    evals: int                   # Evaluations consumed during this visit
    start_gen: int
    end_gen: int
    whenadopted_median: Optional[float]   # Median of all noisy fits up to adoption gen
    whenadopted_count: int
    whendiscarded_median: Optional[float] # Median of all noisy fits up to visit-end gen
    whendiscarded_count: int


@dataclass
class EvaluationRecord:
    """
    Record of a single noisy evaluation.
    Kept for backwards compatibility with fitness functions that call log_noisy_eval.
    The original (true) solution is not stored here.
    noisy_sol is None for posterior noise (noisy_sol == true_sol).
    """
    noisy_sol: Optional[List]
    true_fitness: float
    noisy_fitness: float
    generation: int = None

# ==============================
# Experiment Logger
# ==============================

@dataclass
class ExperimentLogger:
    """
    Centralised logging for algorithm runs.

    Builds trajectory incrementally: only persists a VisitRecord when the best
    solution changes, rather than storing a GenerationRecord for every generation.
    This reduces memory from O(N_generations) to O(N_visits).

    Evaluation data is also stored compactly: only noisy fitness values per unique
    solution are retained (for median estimates), not full EvaluationRecord objects.
    A per-generation buffer of full eval info is kept transiently for one generation
    to allow true_fitness / noisy_sol lookup at visit transitions, then cleared.
    """

    # Closed visit records — one per solution change
    visits: List[VisitRecord] = field(default_factory=list)

    # Track current generation for evaluation logging
    current_generation: int = 0

    # Whether to store population snapshots when representative solution changes
    # (not yet implemented; reserved for future use)
    record_population: bool = False

    # Cache for trajectory data (built once at end of run, invalidated on clear())
    _trajectory_cache: Optional[Dict] = field(default=None, init=False, repr=False)

    # --- Evaluation data ---
    # Current-generation buffer: sol_tuple -> [(noisy_fit, noisy_sol, true_fit)]
    # Cleared after each log_generation call.
    _gen_evals: Dict = field(default_factory=dict, init=False, repr=False)

    # Cumulative noisy-fitness history per unique solution: sol_tuple -> [noisy_fit, ...]
    # Never cleared — used to compute whenadopted / whendiscarded medians.
    _sol_noisy_fits: Dict = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )

    # --- State for stop_condition / progress printing ---
    _last_true_fitness: Optional[float] = field(default=None, init=False, repr=False)
    _last_generation: int = field(default=0, init=False, repr=False)

    # --- Rolling visit state ---
    _cur_sol: Optional[Tuple] = field(default=None, init=False, repr=False)
    _cur_solution: Optional[List] = field(default=None, init=False, repr=False)
    _cur_count: int = field(default=0, init=False, repr=False)
    _cur_start_gen: int = field(default=0, init=False, repr=False)
    _cur_true_fitness: float = field(default=0.0, init=False, repr=False)
    _cur_noisy_fitness: float = field(default=0.0, init=False, repr=False)
    _cur_noisy_sol: Optional[List] = field(default=None, init=False, repr=False)
    _cur_evals_before: int = field(default=0, init=False, repr=False)
    # Length of _sol_noisy_fits[cur_sol] at adoption time — defines whenadopted window
    _whenadopted_n: int = field(default=0, init=False, repr=False)
    # Length of _sol_noisy_fits[cur_sol] at the END of the most recently completed
    # generation — used as the whendiscarded boundary when the visit closes, so
    # that the closing generation's evals of the old solution are excluded.
    _cur_sol_n_at_gen_end: int = field(default=0, init=False, repr=False)
    # Evals count from the previous log_generation call — used as end_evals when
    # a visit is closed (the visit ended at the previous generation, not the current one)
    _prev_evals: int = field(default=0, init=False, repr=False)
    # Cumulative evals at the end of the most recently closed visit
    _last_closed_evals: int = field(default=0, init=False, repr=False)
    # Total generations logged (returned by __len__)
    _total_gens: int = field(default=0, init=False, repr=False)

    # ==============================
    # Core Logging Methods
    # ==============================

    def log_generation(self, generation: int, population, best_solution,
                       best_fitness: float, evals: int = None):
        """
        Log the state at the end of a generation.

        Detects changes in the best solution inline. A VisitRecord is only persisted
        when the best solution changes — not on every generation.
        """
        self._total_gens += 1
        self.current_generation = generation
        evals = evals or 0

        sol_tuple = tuple(best_solution)

        if self._cur_sol is None:
            # First call — start the first visit
            true_fitness, noisy_sol = self._get_eval_info(sol_tuple, best_fitness)
            self._last_true_fitness = true_fitness
            self._open_visit(
                sol_tuple, best_solution, best_fitness, true_fitness, noisy_sol,
                generation, evals
            )
        elif sol_tuple != self._cur_sol:
            # Solution changed — close the current visit using the PREVIOUS generation's
            # evals and noisy-fits count (the current generation belongs to the new visit)
            self._close_visit(
                end_gen=generation - 1,
                n_at_visit_end=self._cur_sol_n_at_gen_end,
                end_evals=self._prev_evals,
            )
            true_fitness, noisy_sol = self._get_eval_info(sol_tuple, best_fitness)
            self._last_true_fitness = true_fitness
            self._open_visit(
                sol_tuple, best_solution, best_fitness, true_fitness, noisy_sol,
                generation, evals
            )
        else:
            # Same solution — continue the current visit
            self._cur_count += 1
            self._last_true_fitness = self._cur_true_fitness

        # Snapshot end-of-generation state for use when this visit eventually closes
        self._cur_sol_n_at_gen_end = len(self._sol_noisy_fits[self._cur_sol])
        self._prev_evals = evals
        self._last_generation = generation

        # Clear the generation buffer — all needed data has been extracted
        self._gen_evals.clear()

    def log_noisy_eval(self, original, noisy, true_fitness, noisy_fitness,
                       generation: int = None):
        """
        Log a noisy evaluation.

        Updates:
        - _sol_noisy_fits: cumulative noisy-fitness history per solution
          (retained for the full run; used to compute whenadopted / whendiscarded medians)
        - _gen_evals: transient per-generation buffer
          (used to look up true_fitness and noisy_sol at visit transitions; cleared each gen)

        Args:
            original: The original solution submitted for evaluation
            noisy:    The noisy/perturbed version actually evaluated
                      (same as original for posterior noise)
            true_fitness:  Fitness without noise
            noisy_fitness: Fitness with noise applied
            generation: Unused; kept for API compatibility
        """
        key = tuple(original)
        noisy_sol = None if noisy == original else list(noisy)

        # Cumulative noisy-fitness history (retained)
        self._sol_noisy_fits[key].append(noisy_fitness)

        # Transient generation buffer (cleared each generation)
        if key not in self._gen_evals:
            self._gen_evals[key] = []
        self._gen_evals[key].append((noisy_fitness, noisy_sol, true_fitness))

    # ==============================
    # Visit State Helpers
    # ==============================

    def _get_eval_info(self, sol_tuple: Tuple, best_fitness: float
                       ) -> Tuple[float, Optional[List]]:
        """
        Look up true_fitness and noisy_sol for a solution from the current-gen buffer.

        Uses the last record for that solution (corresponding to the fitness value
        currently held by the individual). If the same solution was evaluated multiple
        times in one generation, the last evaluation's record is used — this matches
        the fitness value that DEAP assigns to the individual.

        Falls back to (best_fitness, None) if no eval record exists for this gen.
        For posterior noise (noisy_sol is None in the record), the original solution
        is returned as noisy_sol to match prior behaviour.
        """
        records = self._gen_evals.get(sol_tuple, [])
        if not records:
            return best_fitness, None
        _, noisy_sol, true_fitness = records[-1]
        # Posterior noise: noisy_sol is None, return original solution
        resolved_noisy_sol = list(sol_tuple) if noisy_sol is None else noisy_sol
        return true_fitness, resolved_noisy_sol

    def _open_visit(self, sol_tuple: Tuple, solution, noisy_fitness: float,
                    true_fitness: float, noisy_sol: Optional[List],
                    start_gen: int, evals: int):
        """Open a new visit for a newly-adopted best solution."""
        self._cur_sol = sol_tuple
        self._cur_solution = list(solution)
        self._cur_count = 1
        self._cur_start_gen = start_gen
        self._cur_true_fitness = true_fitness
        self._cur_noisy_fitness = noisy_fitness
        self._cur_noisy_sol = noisy_sol
        self._cur_evals_before = self._last_closed_evals
        # All noisy fits up to and including the adoption generation are already in
        # _sol_noisy_fits (log_noisy_eval runs before log_generation each generation)
        self._whenadopted_n = len(self._sol_noisy_fits[sol_tuple])

    def _close_visit(self, end_gen: int, n_at_visit_end: int, end_evals: int):
        """Close the current visit and append a VisitRecord to self.visits."""
        sol_fits = self._sol_noisy_fits[self._cur_sol]

        whenadopted_fits = sol_fits[:self._whenadopted_n]
        whendiscarded_fits = sol_fits[:n_at_visit_end]

        self.visits.append(VisitRecord(
            solution=self._cur_solution,
            true_fitness=self._cur_true_fitness,
            noisy_fitness=self._cur_noisy_fitness,
            noisy_solution=self._cur_noisy_sol,
            iterations=self._cur_count,
            evals=end_evals - self._cur_evals_before,
            start_gen=self._cur_start_gen,
            end_gen=end_gen,
            whenadopted_median=median(whenadopted_fits) if whenadopted_fits else None,
            whenadopted_count=len(whenadopted_fits),
            whendiscarded_median=median(whendiscarded_fits) if whendiscarded_fits else None,
            whendiscarded_count=len(whendiscarded_fits),
        ))
        self._last_closed_evals = end_evals

    # ==============================
    # Trajectory Data Extraction (with caching)
    # ==============================

    def _build_trajectory_cache(self):
        """
        Assemble trajectory data from all VisitRecords (closed + current open visit).

        Called once after the run ends, then cached. Cache is invalidated by clear().
        """
        if self._trajectory_cache is not None:
            return self._trajectory_cache

        # Include all closed visits plus a snapshot of the current open visit (if any)
        all_visits = list(self.visits)
        if self._cur_sol is not None:
            sol_fits = self._sol_noisy_fits[self._cur_sol]
            whenadopted_fits = sol_fits[:self._whenadopted_n]
            whendiscarded_fits = sol_fits[:self._cur_sol_n_at_gen_end]
            all_visits.append(VisitRecord(
                solution=self._cur_solution,
                true_fitness=self._cur_true_fitness,
                noisy_fitness=self._cur_noisy_fitness,
                noisy_solution=self._cur_noisy_sol,
                iterations=self._cur_count,
                evals=self._prev_evals - self._cur_evals_before,
                start_gen=self._cur_start_gen,
                end_gen=self._last_generation,
                whenadopted_median=median(whenadopted_fits) if whenadopted_fits else None,
                whenadopted_count=len(whenadopted_fits),
                whendiscarded_median=median(whendiscarded_fits) if whendiscarded_fits else None,
                whendiscarded_count=len(whendiscarded_fits),
            ))

        self._trajectory_cache = {
            'representative_solutions': [v.solution for v in all_visits],
            'representative_true_fitnesses': [v.true_fitness for v in all_visits],
            'representative_noisy_fitnesses': [v.noisy_fitness for v in all_visits],
            'representative_noisy_solutions': [v.noisy_solution for v in all_visits],
            'solution_iterations': [v.iterations for v in all_visits],
            'solution_evals': [v.evals for v in all_visits],
            'solution_transitions': [
                (tuple(all_visits[i].solution), tuple(all_visits[i + 1].solution))
                for i in range(len(all_visits) - 1)
            ],
            'representative_estimated_true_fits_whenadopted': [
                v.whenadopted_median for v in all_visits
            ],
            'representative_estimated_true_fits_whendiscarded': [
                v.whendiscarded_median for v in all_visits
            ],
            'count_estimated_fits_whenadopted': [v.whenadopted_count for v in all_visits],
            'count_estimated_fits_whendiscarded': [v.whendiscarded_count for v in all_visits],
        }
        return self._trajectory_cache

    @property
    def representative_solutions(self) -> List[List]:
        """Representative solutions in order of adoption, including revisits."""
        return self._build_trajectory_cache()['representative_solutions']

    @property
    def representative_true_fitnesses(self) -> List[float]:
        """True (noise-free) fitness for each representative solution at adoption."""
        return self._build_trajectory_cache()['representative_true_fitnesses']

    @property
    def representative_noisy_fitnesses(self) -> List[float]:
        """Observed (noisy) fitness for each representative solution at adoption."""
        return self._build_trajectory_cache()['representative_noisy_fitnesses']

    @property
    def representative_noisy_solutions(self) -> List[Optional[List]]:
        """Noisy (perturbed) solution at the time each representative was adopted."""
        return self._build_trajectory_cache()['representative_noisy_solutions']

    @property
    def solution_iterations(self) -> List[int]:
        """Consecutive generation count for each visit."""
        return self._build_trajectory_cache()['solution_iterations']

    @property
    def solution_evals(self) -> List[int]:
        """Evaluations consumed during each visit."""
        return self._build_trajectory_cache()['solution_evals']

    @property
    def solution_transitions(self) -> List[Tuple[Tuple, Tuple]]:
        """List of (from_solution, to_solution) tuples for each transition."""
        return self._build_trajectory_cache()['solution_transitions']

    @property
    def representative_estimated_true_fits_whenadopted(self) -> List[Optional[float]]:
        """
        Estimated true fitness at adoption: median of all noisy evaluations of the
        representative solution up to and including the adoption generation.
        """
        return self._build_trajectory_cache()['representative_estimated_true_fits_whenadopted']

    @property
    def representative_estimated_true_fits_whendiscarded(self) -> List[Optional[float]]:
        """
        Estimated true fitness at discard: median of all noisy evaluations of the
        representative solution up to and including the visit-end generation.
        Includes the adoption generation's evaluations plus any evaluations during
        the visit, but excludes the generation in which the solution was replaced.
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
        Extract trajectory data for plotting (backwards compatibility).

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
        self.visits.clear()
        self._gen_evals.clear()
        self._sol_noisy_fits.clear()
        self._trajectory_cache = None
        self.current_generation = 0
        self._last_true_fitness = None
        self._last_generation = 0
        self._cur_sol = None
        self._cur_solution = None
        self._cur_count = 0
        self._cur_start_gen = 0
        self._cur_true_fitness = 0.0
        self._cur_noisy_fitness = 0.0
        self._cur_noisy_sol = None
        self._cur_evals_before = 0
        self._whenadopted_n = 0
        self._cur_sol_n_at_gen_end = 0
        self._prev_evals = 0
        self._last_closed_evals = 0
        self._total_gens = 0

    def __len__(self):
        """Return total number of generations logged."""
        return self._total_gens
