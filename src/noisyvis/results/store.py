"""
The result warehouse: writer and reader together (plan §5.2, §6.3).

Merges the former `io/ExperimentsHelpers.py` (write side, used by the run scripts and LONs) and
`dataio/loader.py` (read side, used by the dashboard). Bodies are unchanged: the append is still
an unlocked load/concat/dump of the whole pickle (B7, deferred), and column names and pickle
format are untouched (R7).
"""

import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Union

from .paths import WAREHOUSE_DIR


# ----------------------------------------------------------------------------- write side

def save_or_append_results(df, filename):
    """
    Save DataFrame to pickle file, appending if file exists.
    """
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'rb') as f:
            existing_df = pickle.load(f)
        # Append new data
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df

    # Save combined data
    with open(filename, 'wb') as f:
        pickle.dump(combined_df, f)

def convert_to_split_edges_format(lon_data):
    """
    Convert LON data to split edges format.
    """
    # This is a placeholder implementation
    return lon_data


# ------------------------------------------------------------------------------ read side

class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


def load_algo_results(
    path: Union[str, Path] = WAREHOUSE_DIR / 'algo_results.pkl'
) -> pd.DataFrame:
    """
    Load algorithm results from pickle file.

    Args:
        path: Path to the algo_results.pkl file. Defaults to the
              standard location in data/warehouse/ under the project root.

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
    path: Union[str, Path] = WAREHOUSE_DIR / 'lon_results.pkl'
) -> pd.DataFrame:
    """
    Load Local Optima Network (LON) results from pickle file.

    Args:
        path: Path to the lon_results.pkl file. Defaults to the
              standard location in data/warehouse/ under the project root.

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
