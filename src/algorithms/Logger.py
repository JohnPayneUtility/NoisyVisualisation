# IMPORTS
import hashlib
import pickle
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _compute_boxplot_stats(fits: List[float]) -> Optional[Tuple[float, float, float, float, float]]:
    """Return (min, Q1, median, Q3, max) or None if fewer than 2 values."""
    if len(fits) < 2:
        return None
    arr = np.array(fits, dtype=float)
    return (
        float(np.min(arr)),
        float(np.percentile(arr, 25)),
        float(np.median(arr)),
        float(np.percentile(arr, 75)),
        float(np.max(arr)),
    )

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
# Fit History Backends
# ==============================

class _MemFitHistory:
    """
    In-memory noisy-fitness history.
    Default backend when no NVMe path is configured.
    """
    def __init__(self):
        self._data: Dict[tuple, List[float]] = defaultdict(list)

    def batch_append(self, updates: Dict[tuple, List[float]]):
        """Append floats for multiple solutions at once."""
        for sol_tuple, fits in updates.items():
            self._data[sol_tuple].extend(fits)

    def get_fits(self, sol_tuple: tuple, end: int = None) -> List[float]:
        """Return the noisy-fit history for a solution, optionally sliced to [:end]."""
        fits = self._data[sol_tuple]
        return fits[:end] if end is not None else list(fits)

    def get_len(self, sol_tuple: tuple) -> int:
        """Return the number of noisy fits recorded for a solution."""
        return len(self._data[sol_tuple])

    def clear(self):
        self._data.clear()

    def close(self):
        pass  # nothing to close


class LMDBFitHistory:
    """
    LMDB-backed noisy-fitness history for NVMe offload.

    Stores one entry per unique solution (keyed by SHA-256 hash of the solution
    tuple), with the value being a packed array of double-precision floats.
    All writes for a generation are batched into a single LMDB transaction.

    Performance flags (sync=False, writemap=True) trade crash-durability for
    speed — acceptable since this is transient working storage that is deleted
    after each run.
    """
    # Virtual address space reservation for LMDB memory map (not physical allocation).
    # Linux does not commit physical pages until they are actually written.
    _DEFAULT_MAP_SIZE = 100 * 1024 ** 3  # 100 GB

    def __init__(self, path: str, map_size: int = _DEFAULT_MAP_SIZE):
        try:
            import lmdb
        except ImportError:
            raise ImportError(
                "lmdb package is required for NVMe storage. "
                "Install it with:  pip install lmdb"
            )
        import os
        os.makedirs(path, exist_ok=True)
        self._env = lmdb.open(
            str(path),
            map_size=map_size,
            sync=False,       # Skip fsync — faster; OK for transient storage
            writemap=True,    # Write directly through memory map — faster writes
            max_readers=128,  # Allow parallel read-only access from worker processes
        )

    @staticmethod
    def _encode_key(sol_tuple: tuple) -> bytes:
        """
        SHA-256 hash of the pickled solution tuple.
        32-byte key: well within LMDB limits and collision-safe for any realistic
        number of solutions (P(collision) < 10^-60 at 1M unique solutions).
        """
        return hashlib.sha256(pickle.dumps(sol_tuple, protocol=2)).digest()

    @staticmethod
    def _pack(floats: List[float]) -> bytes:
        n = len(floats)
        return struct.pack(f'{n}d', *floats)

    @staticmethod
    def _unpack(data: bytes) -> List[float]:
        n = len(data) // 8
        return list(struct.unpack(f'{n}d', data))

    def batch_append(self, updates: Dict[tuple, List[float]]):
        """Append floats for multiple solutions in a single LMDB transaction."""
        with self._env.begin(write=True) as txn:
            for sol_tuple, new_fits in updates.items():
                key = self._encode_key(sol_tuple)
                existing = txn.get(key)
                if existing:
                    current = self._unpack(existing)
                    current.extend(new_fits)
                else:
                    current = list(new_fits)
                txn.put(key, self._pack(current))

    def get_fits(self, sol_tuple: tuple, end: int = None) -> List[float]:
        """Return the noisy-fit history for a solution, optionally sliced to [:end]."""
        key = self._encode_key(sol_tuple)
        with self._env.begin() as txn:
            data = txn.get(key)
        if not data:
            return []
        fits = self._unpack(data)
        return fits[:end] if end is not None else fits

    def get_len(self, sol_tuple: tuple) -> int:
        """Return the number of noisy fits recorded for a solution."""
        key = self._encode_key(sol_tuple)
        with self._env.begin() as txn:
            data = txn.get(key)
        return len(data) // 8 if data else 0

    def clear(self):
        """No-op — the LMDB directory is deleted externally after close()."""
        pass

    def close(self):
        """Close the LMDB environment. The directory should be deleted afterwards."""
        self._env.close()


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
    Kept for backwards compatibility with fitness functions that call log_noisy_eval.
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
    This reduces generation-level memory from O(N_generations) to O(N_visits).

    Noisy fitness history is stored in a fit-history backend:
    - _MemFitHistory  (default): in-memory defaultdict — fast, no dependencies
    - LMDBFitHistory  (opt-in):  LMDB on NVMe — frees RAM for large/long runs

    Set nvme_path to a directory on the NVMe device to enable LMDB offload.
    Each run should use a unique path (e.g. include seed in the path) so that
    parallel runs do not share an environment.

    Noisy fits are batched per generation: log_noisy_eval writes to a transient
    per-generation buffer (_gen_evals), which is flushed to the fit-history backend
    at the start of each log_generation call (one LMDB transaction per generation,
    not one per evaluation).
    """

    # Closed visit records — one per solution change
    visits: List[VisitRecord] = field(default_factory=list)

    # Path to NVMe directory for LMDB-backed fit history.
    # None = use in-memory _MemFitHistory.
    nvme_path: Optional[str] = None

    # Track current generation for evaluation logging
    current_generation: int = 0

    # Whether to store population snapshots when representative solution changes
    # (reserved for future use; not yet implemented)
    record_population: bool = False

    # Cache for trajectory data (built once at end of run, invalidated on clear())
    _trajectory_cache: Optional[Dict] = field(default=None, init=False, repr=False)

    # Fit-history backend (created in __post_init__)
    _fit_history: Any = field(default=None, init=False, repr=False)

    # Current-generation buffer: sol_tuple -> [(noisy_fit, noisy_sol, true_fit)]
    # Flushed to _fit_history at the start of each log_generation call, then cleared.
    _gen_evals: Dict = field(default_factory=dict, init=False, repr=False)

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
    # Length of fit history for cur_sol at adoption time — defines whenadopted window
    _whenadopted_n: int = field(default=0, init=False, repr=False)
    # Length of fit history for cur_sol at the END of the most recently completed
    # generation — used as the whendiscarded boundary when the visit closes, so
    # that the closing generation's evals of the old solution are excluded.
    _cur_sol_n_at_gen_end: int = field(default=0, init=False, repr=False)
    # Evals count from the previous log_generation call — used as end_evals when
    # a visit is closed (the visit ended at the previous generation)
    _prev_evals: int = field(default=0, init=False, repr=False)
    # Cumulative evals at the end of the most recently closed visit
    _last_closed_evals: int = field(default=0, init=False, repr=False)
    # Total generations logged (returned by __len__)
    _total_gens: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Initialise the fit-history backend based on nvme_path."""
        if self.nvme_path is not None:
            self._fit_history = LMDBFitHistory(self.nvme_path)
        else:
            self._fit_history = _MemFitHistory()

    # ==============================
    # Core Logging Methods
    # ==============================

    def log_generation(self, generation: int, population, best_solution,
                       best_fitness: float, evals: int = None):
        """
        Log the state at the end of a generation.

        Detects changes in the best solution inline. A VisitRecord is only persisted
        when the best solution changes — not on every generation.

        The generation buffer (_gen_evals) is flushed to the fit-history backend
        first so that get_len() / get_fits() reflect the current generation's data
        when _open_visit reads _whenadopted_n.
        """
        self._total_gens += 1
        self.current_generation = generation
        evals = evals or 0

        sol_tuple = tuple(best_solution)

        # Flush this generation's noisy fits to the backend BEFORE any reads
        self._flush_gen_evals()

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
            # evals and fit-history count (the current generation belongs to the new visit)
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
        self._cur_sol_n_at_gen_end = self._fit_history.get_len(self._cur_sol)
        self._prev_evals = evals
        self._last_generation = generation

        # Clear the generation buffer — fit data flushed, eval info no longer needed
        self._gen_evals.clear()

    def log_noisy_eval(self, original, noisy, true_fitness, noisy_fitness,
                       generation: int = None):
        """
        Log a noisy evaluation.

        Writes only to the transient per-generation buffer (_gen_evals).
        Noisy fits are flushed to the fit-history backend (in-memory or LMDB)
        at the start of each log_generation call, batched per generation.

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

        if key not in self._gen_evals:
            self._gen_evals[key] = []
        self._gen_evals[key].append((noisy_fitness, noisy_sol, true_fitness))

    # ==============================
    # Internal Helpers
    # ==============================

    def _flush_gen_evals(self):
        """Flush noisy fits from the generation buffer to the fit-history backend."""
        updates = {
            key: [r[0] for r in records]
            for key, records in self._gen_evals.items()
            if records
        }
        if updates:
            self._fit_history.batch_append(updates)

    def _get_eval_info(self, sol_tuple: Tuple, best_fitness: float
                       ) -> Tuple[float, Optional[List]]:
        """
        Look up true_fitness and noisy_sol for a solution from the current-gen buffer.

        Uses the last record for that solution (corresponding to the fitness value
        currently held by the DEAP individual after the final evaluate() call).
        Falls back to (best_fitness, None) if no record exists in this generation.
        For posterior noise (noisy_sol is None), returns the original solution.
        """
        records = self._gen_evals.get(sol_tuple, [])
        if not records:
            return best_fitness, None
        _, noisy_sol, true_fitness = records[-1]
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
        # Fit history has already been flushed for this generation, so get_len
        # correctly reflects all evals up to and including the adoption generation
        self._whenadopted_n = self._fit_history.get_len(sol_tuple)

    def _close_visit(self, end_gen: int, n_at_visit_end: int, end_evals: int):
        """Close the current visit and append a VisitRecord to self.visits."""
        whenadopted_fits = self._fit_history.get_fits(self._cur_sol,
                                                      end=self._whenadopted_n)
        whendiscarded_fits = self._fit_history.get_fits(self._cur_sol,
                                                        end=n_at_visit_end)
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
            whenadopted_fits = self._fit_history.get_fits(self._cur_sol,
                                                          end=self._whenadopted_n)
            whendiscarded_fits = self._fit_history.get_fits(self._cur_sol,
                                                            end=self._cur_sol_n_at_gen_end)
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
            'representative_fitness_boxplot_stats': [
                _compute_boxplot_stats(self._fit_history.get_fits(tuple(v.solution)))
                for v in all_visits
            ],
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

    @property
    def representative_fitness_boxplot_stats(self) -> List[Optional[Tuple[float, float, float, float, float]]]:
        """5-number summary (min, Q1, median, Q3, max) of all noisy evaluations per visited solution."""
        return self._build_trajectory_cache()['representative_fitness_boxplot_stats']

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
        """
        Clear all logged data and invalidate cache.

        Closes the fit-history backend (important for LMDB — releases the file
        handle so the caller can safely delete the LMDB directory afterwards).
        Resets to an in-memory backend so the logger is safe to use again if needed.
        """
        self._fit_history.clear()
        self._fit_history.close()
        self._fit_history = _MemFitHistory()  # safe fallback if logger is reused

        self.visits.clear()
        self._gen_evals.clear()
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
