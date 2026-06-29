"""
Functions for transforming raw data into dashboard-ready DataFrames.

This module contains pure transformation functions that take a DataFrame
and return a new transformed DataFrame. Each function has a single
responsibility and doesn't modify the input.
"""

import numpy as np
import pandas as pd
from typing import List

from .column_config import (
    LIST_COLUMNS,
    DISPLAY1_COLUMNS,
    DISPLAY2_DROP_COLUMNS,
    DISPLAY2_DEDUP_KEYS,
    LON_HIDDEN_COLUMNS,
)


def _compute_n_misjudgements(row) -> int:
    """
    Count misjudgements in a run: steps where the representative solution's
    true fitness moved in the wrong direction (worse than the previous step).

    For maximisation: fits[i+1] < fits[i] is a misjudgement.
    For minimisation: fits[i+1] > fits[i] is a misjudgement.
    """
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    # MO rows and old SO rows (pre-Logger rename) have rep_fits=NaN, not a list.
    # NaN is truthy so `not rep_fits` doesn't catch it — guard with isinstance.
    # TODO: compute MO equivalent from noisy_pf_true_hypervolumes (HV decreased steps)
    if not isinstance(rep_fits, (list, np.ndarray)) or len(rep_fits) < 2:
        return 0
    problem_goal = row.get('problem_goal', 'maximise') if hasattr(row, 'get') else row['problem_goal']
    minimising = str(problem_goal)[:3].lower() == 'min'
    if minimising:
        return sum(1 for i in range(len(rep_fits) - 1) if rep_fits[i + 1] > rep_fits[i])
    return sum(1 for i in range(len(rep_fits) - 1) if rep_fits[i + 1] < rep_fits[i])


def _compute_n_increasing_noise(row) -> int:
    """
    Count steps where the noise magnitude |true_fit - noisy_fit| grew
    relative to the previous step's noise magnitude, within a run.
    """
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    rep_noisy_fits = row.get('rep_noisy_fits') if hasattr(row, 'get') else row['rep_noisy_fits']
    if not isinstance(rep_fits, (list, np.ndarray)) or not isinstance(rep_noisy_fits, (list, np.ndarray)):
        return 0
    if len(rep_fits) != len(rep_noisy_fits) or len(rep_fits) < 2:
        return 0
    diffs = [abs(t - n) for t, n in zip(rep_fits, rep_noisy_fits)]
    return sum(1 for i in range(len(diffs) - 1) if diffs[i + 1] > diffs[i])


def _compute_n_comparison_misjudgements(row) -> int:
    """
    Count steps where the true fitness moved in the wrong direction (worse than
    the previous step) while the noisy fitness moved in the right direction
    (better than the previous step) — i.e. the noisy signal would have favoured
    adopting a solution that was actually worse.

    For maximisation: true_fit[i+1] < true_fit[i] AND noisy_fit[i+1] > noisy_fit[i].
    For minimisation: true_fit[i+1] > true_fit[i] AND noisy_fit[i+1] < noisy_fit[i].
    """
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    rep_noisy_fits = row.get('rep_noisy_fits') if hasattr(row, 'get') else row['rep_noisy_fits']
    if not isinstance(rep_fits, (list, np.ndarray)) or not isinstance(rep_noisy_fits, (list, np.ndarray)):
        return 0
    if len(rep_fits) != len(rep_noisy_fits) or len(rep_fits) < 2:
        return 0
    problem_goal = row.get('problem_goal', 'maximise') if hasattr(row, 'get') else row['problem_goal']
    minimising = str(problem_goal)[:3].lower() == 'min'
    count = 0
    for i in range(len(rep_fits) - 1):
        if minimising:
            true_worse = rep_fits[i + 1] > rep_fits[i]
            noisy_better = rep_noisy_fits[i + 1] < rep_noisy_fits[i]
        else:
            true_worse = rep_fits[i + 1] < rep_fits[i]
            noisy_better = rep_noisy_fits[i + 1] > rep_noisy_fits[i]
        if true_worse and noisy_better:
            count += 1
    return count


def _compute_n_constraint_misjudgements(row) -> int:
    """
    Count visits where the representative solution's true fitness is negative
    (a constraint violation), within a run.
    """
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    if not isinstance(rep_fits, (list, np.ndarray)) or len(rep_fits) == 0:
        return 0
    return sum(1 for fit in rep_fits if fit < 0)


def _evals_to_visit(rep_fits, sol_iters_evals, idx) -> object:
    """
    Compute cumulative evaluations consumed up to and including a given visit index.

    Uses rep_fits (true fitness per visit) and sol_iterations_evals (evals consumed
    per visit) to validate the data, then sums evals through idx.

    Returns None if required data is missing or mismatched in length.
    """
    # MO rows and old SO rows (pre-Logger rename) have rep_fits=NaN, not a list.
    # NaN is truthy so `not rep_fits` doesn't catch it — guard with isinstance.
    # TODO: compute MO equivalent from true_pf_hypervolumes + eval counts per PF snapshot
    #       (requires recording evals alongside each HV measurement in run_mo.py)
    if not isinstance(rep_fits, (list, np.ndarray)) or not isinstance(sol_iters_evals, (list, np.ndarray)):
        return None
    if not rep_fits or not sol_iters_evals or len(rep_fits) != len(sol_iters_evals):
        return None

    return int(sum(sol_iters_evals[:idx + 1]))


def _compute_evals_to_best(row, fits_col='rep_fits') -> object:
    """
    Compute cumulative evaluations at the visit where the best fitness was first achieved.

    The best visit is the one with the highest fitness for maximisation problems
    and the lowest for minimisation problems. fits_col selects which per-visit
    fitness signal to use ('rep_fits' for true fitness, 'rep_noisy_fits' for noisy).
    """
    fits = row[fits_col]
    sol_iters_evals = row['sol_iterations_evals']
    problem_goal = row.get('problem_goal', 'maximise')

    if not isinstance(fits, (list, np.ndarray)) or not fits:
        return None

    goal = str(problem_goal)[:3].lower() if problem_goal else 'max'
    best_idx = int(np.argmin(fits)) if goal == 'min' else int(np.argmax(fits))
    return _evals_to_visit(fits, sol_iters_evals, best_idx)


def _compute_evals_to_final(row, fits_col='rep_fits') -> object:
    """
    Compute cumulative evaluations at the visit representing the final solution found.

    The final visit is always the last entry in fits_col, so this is equivalent
    to the total evaluations consumed across all recorded visits.
    """
    fits = row[fits_col]
    sol_iters_evals = row['sol_iterations_evals']

    if not isinstance(fits, (list, np.ndarray)) or not fits:
        return None

    return _evals_to_visit(fits, sol_iters_evals, len(fits) - 1)


def create_df_no_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a copy of the algorithm DataFrame with list columns removed.

    Used for 2D performance plots where list/trajectory data is not needed,
    only summary statistics like final_fit, max_fit, etc.

    Computes the scalar evals_to_best/evals_to_final columns (and their noisy
    counterparts) before dropping list columns.

    Args:
        df: Raw algorithm results DataFrame

    Returns:
        DataFrame with list columns removed (unique_sols, unique_fits,
        noisy_fits, sol_iterations, sol_transitions, pareto_* columns, etc.)
        plus new scalar columns evals_to_best, evals_to_final, evals_to_best_noisy,
        evals_to_final_noisy, final_fit_noisy, max_fit_noisy, min_fit_noisy.
    """
    df_no_lists = df.copy()

    # Compute scalar metrics from list columns before they are dropped
    if 'rep_fits' in df_no_lists.columns and 'sol_iterations_evals' in df_no_lists.columns:
        df_no_lists['evals_to_best'] = df_no_lists.apply(_compute_evals_to_best, axis=1)
        df_no_lists['evals_to_final'] = df_no_lists.apply(_compute_evals_to_final, axis=1)
    if 'rep_noisy_fits' in df_no_lists.columns and 'sol_iterations_evals' in df_no_lists.columns:
        df_no_lists['evals_to_best_noisy'] = df_no_lists.apply(
            lambda row: _compute_evals_to_best(row, fits_col='rep_noisy_fits'), axis=1)
        df_no_lists['evals_to_final_noisy'] = df_no_lists.apply(
            lambda row: _compute_evals_to_final(row, fits_col='rep_noisy_fits'), axis=1)
    if 'rep_noisy_fits' in df_no_lists.columns:
        df_no_lists['final_fit_noisy'] = df_no_lists['rep_noisy_fits'].apply(
            lambda fits: fits[-1] if isinstance(fits, (list, np.ndarray)) and len(fits) else None)
        df_no_lists['max_fit_noisy'] = df_no_lists['rep_noisy_fits'].apply(
            lambda fits: max(fits) if isinstance(fits, (list, np.ndarray)) and len(fits) else None)
        df_no_lists['min_fit_noisy'] = df_no_lists['rep_noisy_fits'].apply(
            lambda fits: min(fits) if isinstance(fits, (list, np.ndarray)) and len(fits) else None)
    if 'rep_fits' in df_no_lists.columns:
        df_no_lists['n_misjudgements'] = df_no_lists.apply(_compute_n_misjudgements, axis=1)
    if 'rep_fits' in df_no_lists.columns and 'rep_noisy_fits' in df_no_lists.columns:
        df_no_lists['n_increasing_noise'] = df_no_lists.apply(_compute_n_increasing_noise, axis=1)
        df_no_lists['n_comparison_misjudgements'] = df_no_lists.apply(_compute_n_comparison_misjudgements, axis=1)
    if 'rep_fits' in df_no_lists.columns:
        df_no_lists['n_constraint_misjudgements'] = df_no_lists.apply(_compute_n_constraint_misjudgements, axis=1)

    df_no_lists.drop(
        LIST_COLUMNS,
        axis=1,
        errors='ignore',  # Don't fail if columns missing
        inplace=True
    )
    return df_no_lists


def create_display1_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the problem selection table DataFrame (Table 1).

    Shows unique problems with their metadata for the user to select from.
    Each row represents a unique problem configuration.

    Args:
        df: Raw algorithm results DataFrame

    Returns:
        Deduplicated DataFrame with columns:
        - problem_type, problem_goal, problem_name
        - dimensions, opt_global, fit_func, PID
    """
    # Select only the columns we need, handling missing columns gracefully
    available_cols = [col for col in DISPLAY1_COLUMNS if col in df.columns]
    display1_df = df[available_cols].copy()
    return display1_df.drop_duplicates()


def create_display2_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the algorithm selection table DataFrame (Table 2).

    Shows unique algorithm configurations without run-specific data.
    Each row represents a unique (PID, algo_type, algo_name, noise, fit_func)
    combination that the user can select to view results for.

    Args:
        df: Raw algorithm results DataFrame

    Returns:
        Deduplicated DataFrame with algorithm configuration columns,
        excluding run-specific data like seeds, trajectory data, etc.
    """
    # Count runs per unique configuration before dropping columns
    available_keys = [k for k in DISPLAY2_DEDUP_KEYS if k in df.columns]
    if available_keys:
        run_counts = df.groupby(available_keys).size().reset_index(name='no_runs')

    display2_df = df.copy()
    display2_df.drop(
        DISPLAY2_DROP_COLUMNS,
        axis=1,
        errors='ignore',
        inplace=True
    )

    # Deduplicate by algorithm configuration keys
    if available_keys:
        display2_df = display2_df.drop_duplicates(subset=available_keys)
        display2_df = display2_df.merge(run_counts, on=available_keys, how='left')

    return display2_df


def get_lon_display_columns(df_lon: pd.DataFrame) -> List[str]:
    """
    Get the columns to display in the LON table.

    Filters out columns that contain complex data structures (like
    local_optima, edges) that shouldn't be shown in the table view.

    Args:
        df_lon: LON results DataFrame

    Returns:
        List of column names to display (excludes hidden columns like
        local_optima, fitness_values, edges, etc.)
    """
    return [col for col in df_lon.columns if col not in LON_HIDDEN_COLUMNS]
