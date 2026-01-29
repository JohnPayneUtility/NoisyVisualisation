"""
Functions for transforming raw data into dashboard-ready DataFrames.

This module contains pure transformation functions that take a DataFrame
and return a new transformed DataFrame. Each function has a single
responsibility and doesn't modify the input.
"""

import pandas as pd
from typing import List

from .column_config import (
    LIST_COLUMNS,
    DISPLAY1_COLUMNS,
    DISPLAY2_DROP_COLUMNS,
    DISPLAY2_DEDUP_KEYS,
    LON_HIDDEN_COLUMNS,
)


def create_df_no_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a copy of the algorithm DataFrame with list columns removed.

    Used for 2D performance plots where list/trajectory data is not needed,
    only summary statistics like final_fit, max_fit, etc.

    Args:
        df: Raw algorithm results DataFrame

    Returns:
        DataFrame with list columns removed (unique_sols, unique_fits,
        noisy_fits, sol_iterations, sol_transitions, pareto_* columns, etc.)
    """
    df_no_lists = df.copy()
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
    display2_df = df.copy()
    display2_df.drop(
        DISPLAY2_DROP_COLUMNS,
        axis=1,
        errors='ignore',
        inplace=True
    )

    # Deduplicate by algorithm configuration keys
    available_keys = [k for k in DISPLAY2_DEDUP_KEYS if k in display2_df.columns]
    if available_keys:
        display2_df = display2_df.drop_duplicates(subset=available_keys)

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
