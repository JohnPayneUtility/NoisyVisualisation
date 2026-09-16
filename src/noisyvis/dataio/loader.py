"""
Functions for loading raw data from pickle files.

This module handles the raw data loading from disk, with error handling
and path validation. The loaded DataFrames are then transformed by
the transformers module.
"""

import pandas as pd
from pathlib import Path
from typing import Union


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


def load_algo_results(
    path: Union[str, Path] = 'data/dashboard_dw/algo_results.pkl'
) -> pd.DataFrame:
    """
    Load algorithm results from pickle file.

    Args:
        path: Path to the algo_results.pkl file. Defaults to the
              standard location in data/dashboard_dw/.

    Returns:
        DataFrame with algorithm run results containing columns like:
        - algo_name, algo_type, noise, fit_func
        - unique_sols, unique_fits, noisy_fits (trajectory data)
        - pareto_solutions, pareto_fitnesses (MO data)
        - final_fit, max_fit, min_fit (summary statistics)

    Raises:
        DataLoadError: If file not found or invalid format
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Algorithm results file not found: {path}")

    try:
        df = pd.read_pickle(path)
        # Ensure opt_global is float (matches original behavior from Dashboard.py line 63)
        if 'opt_global' in df.columns:
            df['opt_global'] = df['opt_global'].astype(float)
        return df
    except Exception as e:
        raise DataLoadError(f"Failed to load algorithm results from {path}: {e}")


def load_lon_results(
    path: Union[str, Path] = 'data/dashboard_dw/lon_results.pkl'
) -> pd.DataFrame:
    """
    Load Local Optima Network (LON) results from pickle file.

    Args:
        path: Path to the lon_results.pkl file. Defaults to the
              standard location in data/dashboard_dw/.

    Returns:
        DataFrame with LON data containing columns like:
        - PID, problem_name, problem_type
        - local_optima, fitness_values, edges
        - optima_feasibility, neighbour_feasibility

    Raises:
        DataLoadError: If file not found or invalid format
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"LON results file not found: {path}")

    try:
        return pd.read_pickle(path)
    except Exception as e:
        raise DataLoadError(f"Failed to load LON results from {path}: {e}")
