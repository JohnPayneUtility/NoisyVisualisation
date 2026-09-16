"""Registry of persisted-data compatibility fixtures (plan §7.5).

This is a compatibility fixture set, not a collection of old backups. Historical data access
outranks shim removal: if an artefact ever needs an old dotted path, that forwarder stays
permanently, documented as a deliberate compatibility surface.

`test_historical_pickles.py` runs at Stage 1 (baseline), Stage 8 (module locations moved) and
Stage 12 (shims removed), and is what gates those stages.

EXTENDING THIS REGISTRY
Add one `Fixture(...)` entry; the test body does not change. When a new experiment family, payload
structure or schema version appears, preserve a small representative artefact and register it.
Coverage worth adding as artefacts become available:

    * single-objective runs with coping methods (resampling / median)
    * multi-objective runs (the columns holding DEAP individuals)
    * LON and CoLON rows
    * continuous problems (Rastrigin / BiRastrigin)
    * any schema version where columns are added or removed

Current coverage is narrower than Revision 7 anticipated: it described 18 `temp2026*` warehouse
snapshots, none of which exist any more. They were point-in-time copies of the same
single-objective warehouse schema, already represented by the live warehouse, so no genuinely
distinct historical schema or experiment family is currently available to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WORKSPACE = Path("/workspace")

# Column contracts, measured at Stage 1 on the unmodified tree. Renaming any of these columns
# during the restructuring would break the dashboard and every historical artefact (risk R7).
ALGO_RESULTS_COLUMNS = (
    "problem_name", "problem_type", "problem_goal", "dimensions", "opt_global", "mean_value",
    "mean_weight", "PID", "experiment_name", "experiment_description", "fit_func", "noise",
    "penalty", "algo_type", "algo_name", "n_gens", "n_evals", "stop_trigger", "n_unique_sols",
    "final_fit", "max_fit", "min_fit", "seed", "seed_signature", "peak_ram_mb", "payload_path",
    "run_id", "parent_run_id", "rep_sols", "rep_fits", "rep_noisy_fits",
    "rep_estimated_fits_whenadopted", "rep_estimated_fits_whendiscarded",
    "count_estimated_fits_whenadopted", "count_estimated_fits_whendiscarded", "sol_iterations",
    "sol_iterations_evals", "sol_transitions", "rep_noisy_sols", "rep_fitness_boxplot_stats",
    "alternative_rep_sols", "alternative_rep_fits",
)

STAGING_SO_COLUMNS = (
    "problem_name", "problem_type", "problem_goal", "dimensions", "opt_global", "mean_value",
    "mean_weight", "PID", "experiment_name", "experiment_description", "fit_func", "noise",
    "penalty", "algo_type", "algo_name", "n_gens", "n_evals", "stop_trigger", "n_unique_sols",
    "final_fit", "max_fit", "min_fit", "seed", "seed_signature", "peak_ram_mb", "payload_path",
    "run_id", "parent_run_id",
)

LON_RESULTS_COLUMNS = (
    "problem_name", "problem_type", "problem_goal", "dimensions", "opt_global", "PID", "LON_Algo",
    "n_flips_mut", "n_flips_pert", "compression_val", "n_local_optima", "local_optima",
    "fitness_values", "edges", "optima_feasibility", "neighbour_feasibility", "visit_counts",
    "visit_proportions",
)

PAYLOAD_KEYS = (
    "rep_sols", "rep_true_fits", "rep_noisy_fits", "rep_noisy_sols",
    "rep_estimated_true_fits_whenadopted", "rep_estimated_true_fits_whendiscarded",
    "count_estimated_fits_whenadopted", "count_estimated_fits_whendiscarded",
    "rep_fitness_boxplot_stats", "alternative_rep_sols", "alternative_rep_fits", "sol_iterations",
    "sol_iterations_evals", "sol_transitions",
)


@dataclass(frozen=True)
class Fixture:
    """One persisted artefact and the structure it must retain."""

    id: str
    kind: str                      # "dataframe" or "payload"
    path: str | None = None        # relative to /workspace
    glob: str | None = None        # alternative to path
    select: str | None = None      # "oldest" or "newest", for globs
    required: bool = True
    expected_columns: tuple | None = None
    expected_keys: tuple | None = None
    min_rows: int | None = None
    notes: str = ""

    def resolve(self) -> Path | None:
        """The concrete file this fixture refers to today, or None if absent."""
        if self.path is not None:
            candidate = WORKSPACE / self.path
            return candidate if candidate.is_file() else None

        matches = sorted(WORKSPACE.glob(self.glob), key=lambda p: p.stat().st_mtime)
        if not matches:
            return None
        return matches[0] if self.select == "oldest" else matches[-1]


REGISTRY = [
    Fixture(
        id="warehouse-so",
        kind="dataframe",
        path="data/warehouse/algo_results.pkl",
        expected_columns=ALGO_RESULTS_COLUMNS,
        min_rows=38430,
        notes="Live single-objective warehouse; 1.27 GB, rewritten whole on every run (P9/R6).",
    ),
    Fixture(
        id="warehouse-lon",
        kind="dataframe",
        path="data/warehouse/lon_results.pkl",
        expected_columns=LON_RESULTS_COLUMNS,
        min_rows=28,
        notes="Live LON/CoLON warehouse, including the feasibility columns.",
    ),
    Fixture(
        id="staging-so",
        kind="dataframe",
        path="data/temp/results.pkl",
        expected_columns=STAGING_SO_COLUMNS,
        min_rows=1,
        notes="Per-run SO staging file. There is no data/temp/algo_results.pkl.",
    ),
    Fixture(
        id="staging-lon",
        kind="dataframe",
        path="data/temp/lon_results.pkl",
        expected_columns=LON_RESULTS_COLUMNS,
        min_rows=1,
        notes="Per-run LON staging file.",
    ),
    Fixture(
        id="payload-oldest",
        kind="payload",
        glob="data/temp/payloads/*.pkl",
        select="oldest",
        required=False,
        expected_keys=PAYLOAD_KEYS,
        notes="Oldest surviving STN trajectory payload.",
    ),
    Fixture(
        id="payload-newest",
        kind="payload",
        glob="data/temp/payloads/*.pkl",
        select="newest",
        required=False,
        expected_keys=PAYLOAD_KEYS,
        notes="Newest STN trajectory payload.",
    ),
]

# Module paths that must never appear inside a persisted artefact. §3.5 found none; if one ever
# does, the corresponding compatibility shim stays permanently rather than being deleted.
FORBIDDEN_MODULE_PREFIXES = ("src", "noisyvis", "run_helpers", "deap.creator", "omegaconf", "__main__")
