"""STN trajectory payloads: the per-seed pickles written by single-objective runs (plan Stage 7).

Moved from `run.py`. Each SO seed writes its heavy trajectory data to its own pickle, and the
results row keeps only the path; `enrich_df_with_payloads` reads the pickles back into the columns
the dashboard warehouse expects.

The persisted `payload_path` contract (Stage 6):
    a payload under PROJECT_ROOT   -> stored relative, e.g. "data/temp/payloads/stn_payload_seed1_sig….pkl"
    a payload outside PROJECT_ROOT -> stored absolute
    every read                     -> resolved through PROJECT_ROOT (an absolute path stays absolute)

Payload keys, their order, the file name and the pickle protocol are part of the persisted schema.
Multi-objective runs have no STN payloads.
"""

import pickle
from pathlib import Path

import pandas as pd

from noisyvis.results.paths import PROJECT_ROOT


def build_stn_payload(logger) -> dict:
    """The heavy per-seed trajectory data, taken from a finished run's ExperimentLogger."""
    return {
        "rep_sols": logger.representative_solutions,
        "rep_true_fits": logger.representative_true_fitnesses,
        "rep_noisy_fits": logger.representative_noisy_fitnesses,
        "rep_noisy_sols": logger.representative_noisy_solutions,
        "rep_estimated_true_fits_whenadopted": logger.representative_estimated_true_fits_whenadopted,
        "rep_estimated_true_fits_whendiscarded": logger.representative_estimated_true_fits_whendiscarded,
        "count_estimated_fits_whenadopted": logger.count_estimated_fits_whenadopted,
        "count_estimated_fits_whendiscarded": logger.count_estimated_fits_whendiscarded,
        "rep_fitness_boxplot_stats": logger.representative_fitness_boxplot_stats,
        "alternative_rep_sols": logger.alternative_rep_sols,
        "alternative_rep_fits": logger.alternative_rep_fits,
        "sol_iterations": logger.solution_iterations,
        "sol_iterations_evals": logger.solution_evals,
        "sol_transitions": logger.solution_transitions,
    }


def write_stn_payload(payload: dict, payload_dir, seed: int, seed_signature) -> Path:
    """Write a payload under a name unique per seed and signature (safe for parallel workers)."""
    payload_dir = Path(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"stn_payload_seed{seed}_sig{seed_signature}.pkl"
    with open(payload_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload_path


def persisted_payload_path(payload_path: Path) -> str:
    """The `payload_path` stored in a results row: relative to PROJECT_ROOT when inside it."""
    return (
        str(payload_path.relative_to(PROJECT_ROOT))
        if payload_path.is_relative_to(PROJECT_ROOT)
        else str(payload_path)
    )


# Temp function for dashboard transition
def enrich_df_with_payloads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rehydrate heavy trajectory columns from payload pickles so that
    algo_results.pkl remains dashboard-compatible.
    """
    df = df.copy()

    # Initialise columns (important so Pandas knows they exist)
    df["rep_sols"] = None
    df["rep_fits"] = None
    df["rep_noisy_fits"] = None
    df["rep_estimated_fits_whenadopted"] = None
    df["rep_estimated_fits_whendiscarded"] = None
    df["count_estimated_fits_whenadopted"] = None
    df["count_estimated_fits_whendiscarded"] = None
    df["sol_iterations"] = None
    df["sol_iterations_evals"] = None
    df["sol_transitions"] = None
    df["rep_noisy_sols"] = None
    df["rep_fitness_boxplot_stats"] = None
    df["alternative_rep_sols"] = None
    df["alternative_rep_fits"] = None

    for idx, row in df.iterrows():
        payload_path = row.get("payload_path")
        if not payload_path:
            continue

        try:
            with open(PROJECT_ROOT / payload_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load payload {payload_path}: {e}")
            continue

        # Map payload keys → dashboard column names
        df.at[idx, "rep_sols"] = payload.get("rep_sols", [])
        df.at[idx, "rep_fits"] = payload.get("rep_true_fits", [])
        df.at[idx, "rep_noisy_fits"] = payload.get("rep_noisy_fits", [])
        df.at[idx, "rep_estimated_fits_whenadopted"] = payload.get("rep_estimated_true_fits_whenadopted", [])
        df.at[idx, "rep_estimated_fits_whendiscarded"] = payload.get("rep_estimated_true_fits_whendiscarded", [])
        df.at[idx, "count_estimated_fits_whenadopted"] = payload.get("count_estimated_fits_whenadopted", [])
        df.at[idx, "count_estimated_fits_whendiscarded"] = payload.get("count_estimated_fits_whendiscarded", [])
        df.at[idx, "sol_iterations"] = payload.get("sol_iterations", [])
        df.at[idx, "sol_iterations_evals"] = payload.get("sol_iterations_evals", [])
        df.at[idx, "sol_transitions"] = payload.get("sol_transitions", [])
        df.at[idx, "rep_noisy_sols"] = payload.get("rep_noisy_sols", [])
        df.at[idx, "rep_fitness_boxplot_stats"] = payload.get("rep_fitness_boxplot_stats", [])
        df.at[idx, "alternative_rep_sols"] = payload.get("alternative_rep_sols", [])
        df.at[idx, "alternative_rep_fits"] = payload.get("alternative_rep_fits", [])

    return df
