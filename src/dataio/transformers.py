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


def _compute_evals_to_best(row) -> object:
    """
    Compute cumulative evaluations at the visit where the best fitness was first achieved.

    Uses rep_fits (true fitness per visit) and sol_iterations_evals (evals consumed
    per visit). The best visit is the one with the highest fitness for maximisation
    problems and the lowest for minimisation problems.

    Returns None if required data is missing or mismatched in length.
    """
    rep_fits = row['rep_fits']
    sol_iters_evals = row['sol_iterations_evals']
    problem_goal = row.get('problem_goal', 'maximise')

    if not rep_fits or not sol_iters_evals or len(rep_fits) != len(sol_iters_evals):
        return None

    goal = str(problem_goal)[:3].lower() if problem_goal else 'max'
    best_idx = int(np.argmin(rep_fits)) if goal == 'min' else int(np.argmax(rep_fits))
    return int(sum(sol_iters_evals[:best_idx + 1]))


def create_df_no_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a copy of the algorithm DataFrame with list columns removed.

    Used for 2D performance plots where list/trajectory data is not needed,
    only summary statistics like final_fit, max_fit, etc.

    Computes the scalar evals_to_best column before dropping list columns.

    Args:
        df: Raw algorithm results DataFrame

    Returns:
        DataFrame with list columns removed (unique_sols, unique_fits,
        noisy_fits, sol_iterations, sol_transitions, pareto_* columns, etc.)
        plus a new scalar column evals_to_best.
    """
    df_no_lists = df.copy()

    # Compute evals_to_best before list columns are dropped
    if 'rep_fits' in df_no_lists.columns and 'sol_iterations_evals' in df_no_lists.columns:
        df_no_lists['evals_to_best'] = df_no_lists.apply(_compute_evals_to_best, axis=1)

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
