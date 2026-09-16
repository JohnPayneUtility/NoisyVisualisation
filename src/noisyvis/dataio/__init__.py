"""
Data loading and transformation module for the dashboard.

This module provides a clean interface for loading and transforming
the data needed by the dashboard. It separates data concerns from
the dashboard presentation logic.

Usage:
    from src.data import DashboardData

    # Load all data with default paths
    data = DashboardData.load()

    # Access the data
    df = data.df                    # Full algorithm results
    df_lon = data.df_lon            # LON results
    display1_df = data.display1_df  # Problem selection table
    display2_df = data.display2_df  # Algorithm selection table

    # Or load from custom paths
    data = DashboardData.load(
        algo_path='path/to/algo_results.pkl',
        lon_path='path/to/lon_results.pkl'
    )
"""

from dataclasses import dataclass
from typing import List
import pandas as pd

from .loader import load_algo_results, load_lon_results, DataLoadError
from .transformers import (
    create_df_no_lists,
    create_display1_df,
    create_display2_df,
    get_lon_display_columns,
)
from .column_config import (
    LON_HIDDEN_COLUMNS,
    DISPLAY2_HIDDEN_COLUMNS,
    DISPLAY1_COLUMNS,
    LIST_COLUMNS,
    DISPLAY2_DROP_COLUMNS,
    DISPLAY2_DEDUP_KEYS,
)


@dataclass
class DashboardData:
    """
    Container for all dashboard data.

    This class bundles together all the DataFrames and configuration
    needed by the dashboard, providing a single point of access.

    Attributes:
        df: Full algorithm results DataFrame with all columns
        df_lon: LON (Local Optima Network) results DataFrame
        df_no_lists: Algorithm results without list columns (for 2D plots)
        display1_df: Problem selection table (deduplicated problems)
        display2_df: Algorithm selection table (deduplicated configurations)
        lon_display_columns: Column names to show in LON table
    """

    # Raw data
    df: pd.DataFrame
    df_lon: pd.DataFrame

    # Transformed data
    df_no_lists: pd.DataFrame
    display1_df: pd.DataFrame
    display2_df: pd.DataFrame

    # Column configuration
    lon_display_columns: List[str]

    @classmethod
    def load(
        cls,
        algo_path: str = 'data/dashboard_dw/algo_results.pkl',
        lon_path: str = 'data/dashboard_dw/lon_results.pkl',
    ) -> 'DashboardData':
        """
        Load and transform all dashboard data.

        This is the main entry point for loading data. It handles both
        the raw data loading and all necessary transformations.

        Args:
            algo_path: Path to algorithm results pickle file.
                      Defaults to 'data/dashboard_dw/algo_results.pkl'
            lon_path: Path to LON results pickle file.
                     Defaults to 'data/dashboard_dw/lon_results.pkl'

        Returns:
            DashboardData instance with all data loaded and transformed

        Raises:
            DataLoadError: If either file cannot be loaded
        """
        # Load raw data
        df = load_algo_results(algo_path)
        df_lon = load_lon_results(lon_path)

        # Transform for different uses
        df_no_lists = create_df_no_lists(df)
        display1_df = create_display1_df(df)
        display2_df = create_display2_df(df)
        lon_display_columns = get_lon_display_columns(df_lon)

        return cls(
            df=df,
            df_lon=df_lon,
            df_no_lists=df_no_lists,
            display1_df=display1_df,
            display2_df=display2_df,
            lon_display_columns=lon_display_columns,
        )


# Public API
__all__ = [
    # Main data container
    'DashboardData',
    # Exception
    'DataLoadError',
    # Column configuration constants
    'LON_HIDDEN_COLUMNS',
    'DISPLAY2_HIDDEN_COLUMNS',
    'DISPLAY1_COLUMNS',
    'LIST_COLUMNS',
    'DISPLAY2_DROP_COLUMNS',
    'DISPLAY2_DEDUP_KEYS',
    # Individual loader functions (for advanced use)
    'load_algo_results',
    'load_lon_results',
    # Individual transformer functions (for advanced use)
    'create_df_no_lists',
    'create_display1_df',
    'create_display2_df',
    'get_lon_display_columns',
]
