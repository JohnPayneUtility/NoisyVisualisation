"""
Main layout assembly for the Dashboard.

This module imports components from other layout files and assembles
them into the complete dashboard layout.
"""
from dash import html

from .stores import create_all_stores
from .components import (
    create_problem_selection_tabs,
    create_2d_plot_tabs,
    create_algorithm_table,
    create_pareto_front_section,
    create_multiobjective_options_section,
    create_stn_options_section,
    create_lon_options_section,
    create_plot_options_section,
    create_opacity_options_section,
    create_axis_options_section,
    create_main_plot_section,
)


def create_layout(display2_df, display2_hidden_cols):
    """
    Create the complete dashboard layout.

    Args:
        display2_df: DataFrame for algorithm selection table.
        display2_hidden_cols: List of column names to hide in table2.

    Returns:
        html.Div: The complete dashboard layout.
    """
    # Build the layout by assembling all components
    children = []

    # Header
    children.append(html.H2("LON/STN Dashboard", style={'textAlign': 'center'}))

    # Hidden stores for state management
    children.extend(create_all_stores())

    # Problem selection tabs
    children.append(create_problem_selection_tabs())

    # 2D performance plot tabs
    children.append(create_2d_plot_tabs())

    # Algorithm selection table (Table 2)
    children.extend(create_algorithm_table(display2_df, display2_hidden_cols))

    # Pareto front plotting section
    children.extend(create_pareto_front_section())

    # Multiobjective options
    children.extend(create_multiobjective_options_section())

    # STN run options
    children.extend(create_stn_options_section())

    # LON options
    children.extend(create_lon_options_section())

    # General plotting options
    children.extend(create_plot_options_section())

    # Opacity options
    children.extend(create_opacity_options_section())

    # Axis range options
    children.extend(create_axis_options_section())

    # Main plot and info display
    children.extend(create_main_plot_section())

    return html.Div(children)
