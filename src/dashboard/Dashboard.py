import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px  # for continuous color scales
import networkx as nx
import numpy as np
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import ClassicalMDS
from sklearn.manifold import TSNE
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os

from .DashboardHelpers import *
from ..problems.FitnessFunctions import *
from ..problems.ProblemScripts import load_problem_KP
from .DimensionalityReduction import *
from .layout import create_layout, TAB_STYLE, TAB_SELECTED_STYLE
from .layout.stores import LON_TABLE_SELECTED_PID_STORE

# Visualization module imports
from ..visualization import (
    parse_callback_inputs,
    PlotConfig,
    generate_run_summary_string,
    add_stn_trajectories,
    add_mo_fronts,
    add_prior_noise_stn_v4,
    add_prior_noise_stn_v5,
    add_lon_nodes,
    add_lon_edges,
    debug_mo_counts,
    style_nodes,
    calculate_positions,
    calculate_lon_statistics,
    build_all_traces,
    create_axis_settings,
    create_figure,
)

# Plotting module imports - using registry for dynamic dispatch
from ..plotting import get_pareto_plot
from ..plotting.performance import plot2d_line, plot2d_box, plot2d_line_mo, plot2d_box_mo, plot2d_line_evals, plot2d_box_evals, plot2d_box_misjudgements_so

# ==========
# Data Loading
# ==========
from ..dataio import DashboardData, DISPLAY2_HIDDEN_COLUMNS, LON_HIDDEN_COLUMNS
from ..dataio.transformers import create_display2_df
from ..dataio.column_config import DISPLAY1_COLUMNS

# Load all dashboard data using the data module
data = DashboardData.load()

# Unpack for backward compatibility with existing callbacks
# These variable names are used throughout the dashboard code
df = data.df                              # Full algorithm results
df_LONs = data.df_lon                     # LON results
df_no_lists = data.df_no_lists            # Algorithm results without list columns
display1_df = data.display1_df            # Problem selection table
display2_df = data.display2_df            # Algorithm selection table
LON_display_columns = data.lon_display_columns  # Columns to show in LON table

# Column configuration for table displays
display2_hidden_cols = DISPLAY2_HIDDEN_COLUMNS
LON_hidden_cols = LON_HIDDEN_COLUMNS

# Unique experiment names for the top-level filter dropdown
experiment_names = sorted(df['experiment_name'].dropna().unique().tolist()) if 'experiment_name' in df.columns else []
# ==========
# Main Dashboard App
# ==========

app = dash.Dash(__name__, suppress_callback_exceptions=True)
# app = dash.Dash(__name__) # Don't suppress exceptions

# ---------- Layout Definition ----------
# The layout is defined in the layout module for better organization.
# Style constants (tab_style, tab_selected_style) are imported from layout module.
# Local aliases for backward compatibility with callbacks that use lowercase names
tab_style = TAB_STYLE
tab_selected_style = TAB_SELECTED_STYLE

app.layout = create_layout(display2_df, display2_hidden_cols, display1_df, df_LONs, LON_display_columns, experiment_names)

# ---------- Fit function -> x-axis label mapping ----------
FIT_FUNC_XAXIS_LABELS = {
    'OneMax_fitness': 'sigma (s.d. of gaussian Noise N(0, sigma))',
    'OneMax_prior_bitflip_fitness': 'p (probability of single bit flip (p/n))',
    'OneMax_prior_mult_bitflip_fitness': 'k (number of bit flips)',
    'OneMax_prior_pq_bitwise_fitness': 'q (bitwise flip probability q/n), probability of applying noise 1/n',
    'OneMax_prior_1q_bitwise_fitness': 'q (bitwise flip probability q/n)',
    'eval_noisy_kp_v1': 'd, where d x mean(W) is s.d. of noise',
    'eval_noisy_kp_v2': 'd, where d x mean(W) is s.d. of noise',
    'eval_noisy_kp_prior_bitflip': 'p (probability of single bit flip (p/n))',
    'eval_noisy_kp_prior_mult_bitflip': 'k (number of bit flips)',
    'eval_noisy_kp_pq_prior_bitwise': 'q (bitwise flip probability q/n), probability of applying noise 1/n',
    'eval_noisy_kp_1q_prior_bitwise': 'q (bitwise flip probability q/n)',
    'rastrigin_eval': 'sigma (s.d. of gaussian Noise N(0, sigma))',
    'birastrigin_eval': 'sigma (s.d. of gaussian Noise N(0, sigma))',
}

def _get_so_xaxis_label(fit_func):
    """Return the x-axis label for the given fit_func, or None if unset."""
    if not fit_func:
        return None
    return FIT_FUNC_XAXIS_LABELS.get(fit_func, fit_func)


def _get_problem_goal(opt_goal):
    """Return the problem_goal ('maximise' or 'minimise') from the opt_goal store."""
    return opt_goal or 'maximise'

# ------------------------------
# Callbacks: Update Selection Stores
# ------------------------------

@app.callback(
    Output("table1", "data"),
    Input("experiment-selector", "value"),
)
def update_table1_for_experiment(experiment_name):
    filtered = df[df['experiment_name'] == experiment_name] if experiment_name and 'experiment_name' in df.columns else df
    available_cols = [col for col in DISPLAY1_COLUMNS if col in filtered.columns]
    return filtered[available_cols].drop_duplicates().to_dict("records")


@app.callback(
    Output("table1-selected-store", "data"),
    Input("table1", "selected_rows"),
    Input("experiment-selector", "value"),
    prevent_initial_call=True
)
def update_table1_store(selected_rows, _experiment_name):
    if ctx.triggered_id == "experiment-selector":
        return []
    return selected_rows

@app.callback(
    Output("table2-selected-store", "data"),
    Input("table2", "selected_rows"),
    prevent_initial_call=True
)
def update_table2_store(selected_rows):
    return selected_rows

# ------------------------------
# Callback: Filter Table 2 Based on Selections
# ------------------------------

# Update data store filtered by specific problem
@app.callback(
    Output("data-problem-specific", "data"),
    Input("table1-selected-store", "data"),
    Input("experiment-selector", "value"),
    State("table1", "data"),
)
def filter_table2(selection1, experiment_name, table1_current_data):
    exp_df = df[df['experiment_name'] == experiment_name] if experiment_name and 'experiment_name' in df.columns else df
    exp_display2_df = create_display2_df(exp_df)

    union = set()
    if selection1 and table1_current_data:
        for idx in selection1:
            if idx < len(table1_current_data):
                row = table1_current_data[idx]
                union.add((row['PID'], row['fit_func']))
    if not union:
        return exp_display2_df.to_dict("records")
    else:
        mask = exp_display2_df.apply(
            lambda r: (r['PID'], r['fit_func']) in union, axis=1
        )
        return exp_display2_df[mask].to_dict("records")

# Update table 2 to use problem specific data store
@app.callback(
    Output("table2", "data"),
    Input("data-problem-specific", "data")
)
def update_table2(data):
    if data is None:
        return []
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    return df.to_dict('records')

@app.callback(
    [Output("optimum", "data"),
     Output("PID", "data"),
     Output("opt_goal", 'data'),
     Output("fit_func_store", 'data')],
    [Input("data-problem-specific", "data"),
     Input(LON_TABLE_SELECTED_PID_STORE, "data")],
    State("table1-selected-store", "data"),
)
def update_table2(data, lon_table_pid, table1_selection):
    # Primary: use problem table selection if available
    if data is not None and table1_selection:
        df = pd.DataFrame(data)
        optimum = df["opt_global"].iloc[0]
        PID = df["PID"].iloc[0]
        opt_goal = df["problem_goal"].iloc[0]
        fit_func = df["fit_func"].iloc[0]
        return optimum, PID, opt_goal, fit_func

    # Fallback: no problem selected in table 1, use PID from selected LON row
    if lon_table_pid is not None:
        return None, lon_table_pid, None, None

    return None, None, None, None

# ------------------------------
# Callback: Display Selected Rows from Table 2
# ------------------------------

@app.callback(
    Output("table2-selected-output", "children"),
    Input("table2", "selected_rows"),
    State("table2", "data")
)
def update_table2_selected(selected_rows, table2_data):
    if not selected_rows:
        return "No rows selected in Table 2."
    selected_data = [table2_data[i] for i in selected_rows]
    return f"Table 2 selected rows: {selected_data}"

# ------------------------------
# 2D Plot
# ------------------------------
# ---------- Generate data for 2D performance plot by filtering main data with table 2 selection ----------
filter_columns = [col for col in display2_df.columns if col in df_no_lists.columns]
@app.callback(
    Output('plot_2d_data', 'data'),
    Input('table2', 'derived_virtual_data')
)
def update_filtered_view(filtered_data):
    # If no filtering is applied, return the full data.
    if not filtered_data:
        return df_no_lists.to_dict('records')
    
    # Convert the filtered data to a DataFrame.
    df_filtered = pd.DataFrame(filtered_data)
    # print("df_filtered:")
    # print(df_filtered.head())
    
    mask = pd.Series(True, index=df_no_lists.index)
    for col in filter_columns:
        if col in df_filtered.columns:
            allowed_values = df_filtered[col].unique()
            # print(f"Filtering column {col} with allowed values: {allowed_values}")
            # If any allowed value is null (None or np.nan), allow null rows.
            if any(pd.isnull(allowed_values)):
                mask &= (df_no_lists[col].isin(allowed_values) | df_no_lists[col].isnull())
            else:
                mask &= df_no_lists[col].isin(allowed_values)
    
    df_result = df_no_lists[mask]
    # print("Filtered result (first few rows):")
    # print(df_result.head())
    return df_result.to_dict('records')

# ---------- 2D plot callbacks ----------
# Plot data table
@app.callback(
    Output('plot_2d_data_table', 'data'),
    Input('plot_2d_data', 'data')
)
def display_stored_data(data):
    return data

# 2D line plot
@app.callback(
    Output('2DLinePlot', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('opt_goal', 'data'),
    Input('plot-theme', 'value'),
)
def display_stored_data(data, fitness_mode, fit_func, opt_goal, plot_theme):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    problem_goal = _get_problem_goal(opt_goal)
    plot_df = pd.DataFrame(data)
    return plot2d_line(plot_df, fitness_mode=fitness_mode or 'best', problem_goal=problem_goal, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')
# 2D box plot
@app.callback(
    Output('2DBoxPlot', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('opt_goal', 'data'),
    Input('plot-theme', 'value'),
)
def display_stored_data(data, fitness_mode, fit_func, opt_goal, plot_theme):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    problem_goal = _get_problem_goal(opt_goal)
    plot_df = pd.DataFrame(data)
    return plot2d_box(plot_df, fitness_mode=fitness_mode or 'best', problem_goal=problem_goal, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# 2D line plot (multi-objective)
@app.callback(
    Output('2DLinePlotMO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('plot-theme', 'value'),
)
def display_stored_data_mo_line(data, plot_theme):
    plot_df = pd.DataFrame(data)
    plot = plot2d_line_mo(plot_df, colorscale=plot_theme or 'Viridis')
    return plot

# 2D box plot (multi-objective)
@app.callback(
    Output('2DBoxPlotMO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('plot-theme', 'value'),
)
def display_stored_data_mo_box(data, plot_theme):
    plot_df = pd.DataFrame(data)
    plot = plot2d_box_mo(plot_df, colorscale=plot_theme or 'Viridis')
    return plot

# 2D line plot (evals, single-objective)
@app.callback(
    Output('2DLinePlotEvalsSO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('line-evals-show-std', 'value'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('plot-theme', 'value'),
)
def display_line_evals_so(data, std_checkbox, fitness_mode, fit_func, plot_theme):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    plot_df = pd.DataFrame(data)
    show_std = bool(std_checkbox and 'show' in std_checkbox)
    return plot2d_line_evals(plot_df, fitness_mode=fitness_mode or 'final', show_std=show_std, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# 2D box plot (evals, single-objective)
@app.callback(
    Output('2DBoxPlotEvalsSO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('plot-theme', 'value'),
)
def display_box_evals_so(data, fitness_mode, fit_func, plot_theme):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    plot_df = pd.DataFrame(data)
    return plot2d_box_evals(plot_df, fitness_mode=fitness_mode or 'final', xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# 2D box plot (misjudgements, single-objective)
@app.callback(
    Output('2DBoxPlotMisjudgementsSO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('fit_func_store', 'data'),
    Input('plot-theme', 'value'),
)
def display_box_misjudgements_so(data, fit_func, plot_theme):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    plot_df = pd.DataFrame(data)
    return plot2d_box_misjudgements_so(plot_df, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# ---------- Performance summary table ----------
@app.callback(
    Output('performance-summary-table', 'children'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('opt_goal', 'data'),
)
def update_performance_summary_table(data, fitness_mode, fit_func, opt_goal):
    if not fit_func:
        return html.P(
            "Select a problem from the table above to see the performance summary.",
            style={'color': '#888', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    if not data:
        return html.P("No data available.", style={'color': '#888', 'fontStyle': 'italic'})

    problem_goal = _get_problem_goal(opt_goal)
    minimising = problem_goal == 'minimise'

    plot_df = pd.DataFrame(data)
    if fitness_mode == 'final':
        fit_col = 'final_fit'
        label = 'Final Fitness'
    elif minimising:
        fit_col = 'min_fit'
        label = 'Best Fitness'
    else:
        fit_col = 'max_fit'
        label = 'Best Fitness'

    required_cols = {'algo_name', 'noise', fit_col}
    if not required_cols.issubset(plot_df.columns):
        return html.P("Required columns not available for summary.", style={'color': '#888'})

    df_sub = plot_df[['algo_name', 'noise', fit_col]].copy()
    stats = df_sub.groupby(['noise', 'algo_name'])[fit_col].agg(['median', 'std']).reset_index()

    algos = sorted(stats['algo_name'].unique())
    noise_levels = sorted(stats['noise'].unique())

    rows = []
    highlight_cells = []  # (row_index, col_id) for best algorithm per noise level

    for i, noise in enumerate(noise_levels):
        row = {'Noise Level': noise}
        noise_stats = stats[stats['noise'] == noise]
        best_median = None
        best_algos = []
        for algo in algos:
            algo_row = noise_stats[noise_stats['algo_name'] == algo]
            if algo_row.empty:
                row[algo] = '-'
            else:
                med = algo_row['median'].values[0]
                std = algo_row['std'].values[0]
                std_str = f'{std:.3f}' if pd.notna(std) else 'N/A'
                row[algo] = f'{med:.3f} ± {std_str}'
                if best_median is None or (minimising and med < best_median) or (not minimising and med > best_median):
                    best_median = med
                    best_algos = [algo]
                elif med == best_median:
                    best_algos.append(algo)
        rows.append(row)
        for algo in best_algos:
            highlight_cells.append((i, algo))

    columns = [{'name': 'Noise Level', 'id': 'Noise Level'}] + [{'name': a, 'id': a} for a in algos]

    style_data_conditional = [
        {
            'if': {'row_index': row_idx, 'column_id': col_id},
            'backgroundColor': '#d4edda',
            'fontWeight': 'bold',
        }
        for row_idx, col_id in highlight_cells
    ]

    highlight_description = (
        "Green highlight indicates the lowest median fitness for that noise level."
        if minimising else
        "Green highlight indicates the highest median fitness for that noise level."
    )

    table = dash_table.DataTable(
        data=rows,
        columns=columns,
        style_data_conditional=style_data_conditional,
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'padding': '6px 12px', 'fontFamily': 'monospace'},
        style_table={'marginBottom': '8px'},
    )

    return html.Div([
        html.H5(f"Performance Summary: Median {label} ± Std Dev by Noise Level",
                style={'marginTop': '16px', 'marginBottom': '4px'}),
        html.P(highlight_description,
               style={'color': '#555', 'fontSize': '12px', 'marginBottom': '8px'}),
        table,
        html.Hr(),
    ])


# ---------- Mann-Whitney U-test pairwise table ----------
@app.callback(
    Output('mann-whitney-table', 'children'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('opt_goal', 'data'),
)
def update_mann_whitney_table(data, fitness_mode, fit_func, opt_goal):
    from scipy.stats import mannwhitneyu

    if not fit_func:
        return html.P(
            "Select a problem from the table above to see the Mann-Whitney U-test results.",
            style={'color': '#888', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    if not data:
        return html.P("No data available.", style={'color': '#888', 'fontStyle': 'italic'})

    problem_goal = _get_problem_goal(opt_goal)
    plot_df = pd.DataFrame(data)
    if fitness_mode == 'final':
        fit_col = 'final_fit'
        label = 'Final Fitness'
    elif problem_goal == 'minimise':
        fit_col = 'min_fit'
        label = 'Best Fitness'
    else:
        fit_col = 'max_fit'
        label = 'Best Fitness'

    required_cols = {'algo_name', 'noise', fit_col}
    if not required_cols.issubset(plot_df.columns):
        return html.P("Required columns not available for Mann-Whitney tests.", style={'color': '#888'})

    algos = sorted(plot_df['algo_name'].unique())
    noise_levels = sorted(plot_df['noise'].unique())

    def build_tab_content(noise):
        noise_df = plot_df[plot_df['noise'] == noise]
        rows = []
        style_data_conditional = []

        for row_idx, algo_row in enumerate(algos):
            row = {'Algorithm': algo_row}
            samples_row = noise_df[noise_df['algo_name'] == algo_row][fit_col].dropna().values
            for algo_col in algos:
                if algo_row == algo_col:
                    row[algo_col] = '-'
                else:
                    samples_col = noise_df[noise_df['algo_name'] == algo_col][fit_col].dropna().values
                    if len(samples_row) < 2 or len(samples_col) < 2:
                        row[algo_col] = 'N/A'
                    else:
                        _, p = mannwhitneyu(samples_row, samples_col, alternative='two-sided')
                        row[algo_col] = f'{p:.4f}'
                        if p < 0.05:
                            style_data_conditional.append({
                                'if': {'row_index': row_idx, 'column_id': algo_col},
                                'backgroundColor': '#d4edda',
                                'fontWeight': 'bold',
                            })
            rows.append(row)

        columns = [{'name': 'Algorithm', 'id': 'Algorithm'}] + [{'name': a, 'id': a} for a in algos]

        return dash_table.DataTable(
            data=rows,
            columns=columns,
            style_data_conditional=style_data_conditional,
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0', 'textAlign': 'center'},
            style_cell={'textAlign': 'center', 'padding': '6px 12px', 'fontFamily': 'monospace'},
            style_table={'marginBottom': '8px'},
        )

    tabs = dcc.Tabs(
        children=[
            dcc.Tab(
                label=f'Noise = {noise}',
                children=[build_tab_content(noise)],
                style=tab_style,
                selected_style=tab_selected_style,
            )
            for noise in noise_levels
        ]
    )

    return html.Div([
        html.H5(f"Mann-Whitney U-Test: Pairwise p-values ({label}, two-sided)",
                style={'marginTop': '16px', 'marginBottom': '4px'}),
        html.P("Green highlight indicates p < 0.05 (statistically significant difference). Each tab shows one noise level.",
               style={'color': '#555', 'fontSize': '12px', 'marginBottom': '8px'}),
        tabs,
        html.Hr(),
    ])


# ---------- Evaluations summary table ----------
@app.callback(
    Output('evals-summary-table', 'children'),
    Input('plot_2d_data', 'data'),
    Input('fit_func_store', 'data'),
    Input('so-fitness-mode', 'value'),
)
def update_evals_summary_table(data, fit_func, fitness_mode):
    if not fit_func:
        return html.P(
            "Select a problem from the table above to see the evaluations summary.",
            style={'color': '#888', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    if not data:
        return html.P("No data available.", style={'color': '#888', 'fontStyle': 'italic'})

    plot_df = pd.DataFrame(data)

    use_best = fitness_mode == 'best' and 'evals_to_best' in plot_df.columns
    eval_col = 'evals_to_best' if use_best else 'n_evals'

    if not {'algo_name', 'noise', eval_col}.issubset(plot_df.columns):
        return html.P("Required columns not available for evaluations summary.", style={'color': '#888'})

    df_sub = plot_df[['algo_name', 'noise', eval_col]].dropna(subset=[eval_col]).copy()
    stats = df_sub.groupby(['noise', 'algo_name'])[eval_col].agg(['median', 'std']).reset_index()

    algos = sorted(stats['algo_name'].unique())
    noise_levels = sorted(stats['noise'].unique())

    rows = []
    highlight_cells = []

    for i, noise in enumerate(noise_levels):
        row = {'Noise Level': noise}
        noise_stats = stats[stats['noise'] == noise]
        best_median = None
        best_algos = []
        for algo in algos:
            algo_row = noise_stats[noise_stats['algo_name'] == algo]
            if algo_row.empty:
                row[algo] = '-'
            else:
                med = algo_row['median'].values[0]
                std = algo_row['std'].values[0]
                std_str = f'{std:.3f}' if pd.notna(std) else 'N/A'
                row[algo] = f'{med:.1f} ± {std_str}'
                if best_median is None or med < best_median:
                    best_median = med
                    best_algos = [algo]
                elif med == best_median:
                    best_algos.append(algo)
        rows.append(row)
        for algo in best_algos:
            highlight_cells.append((i, algo))

    columns = [{'name': 'Noise Level', 'id': 'Noise Level'}] + [{'name': a, 'id': a} for a in algos]

    style_data_conditional = [
        {
            'if': {'row_index': row_idx, 'column_id': col_id},
            'backgroundColor': '#d4edda',
            'fontWeight': 'bold',
        }
        for row_idx, col_id in highlight_cells
    ]

    table = dash_table.DataTable(
        data=rows,
        columns=columns,
        style_data_conditional=style_data_conditional,
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'padding': '6px 12px', 'fontFamily': 'monospace'},
        style_table={'marginBottom': '8px'},
    )

    evals_label = 'Evaluations to Best Found Fitness' if use_best else 'Runtime (n_evals)'
    return html.Div([
        html.H5(f"Evaluations Summary: Median {evals_label} ± Std Dev by Noise Level",
                style={'marginTop': '16px', 'marginBottom': '4px'}),
        html.P("Green highlight indicates the lowest median evaluations for that noise level.",
               style={'color': '#555', 'fontSize': '12px', 'marginBottom': '8px'}),
        table,
        html.Hr(),
    ])


# ---------- Evaluations Mann-Whitney U-test pairwise table ----------
@app.callback(
    Output('evals-mann-whitney-table', 'children'),
    Input('plot_2d_data', 'data'),
    Input('fit_func_store', 'data'),
    Input('so-fitness-mode', 'value'),
)
def update_evals_mann_whitney_table(data, fit_func, fitness_mode):
    from scipy.stats import mannwhitneyu

    if not fit_func:
        return html.P(
            "Select a problem from the table above to see the evaluations Mann-Whitney U-test results.",
            style={'color': '#888', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    if not data:
        return html.P("No data available.", style={'color': '#888', 'fontStyle': 'italic'})

    plot_df = pd.DataFrame(data)

    use_best = fitness_mode == 'best' and 'evals_to_best' in plot_df.columns
    eval_col = 'evals_to_best' if use_best else 'n_evals'

    if not {'algo_name', 'noise', eval_col}.issubset(plot_df.columns):
        return html.P("Required columns not available for evaluations Mann-Whitney tests.", style={'color': '#888'})

    algos = sorted(plot_df['algo_name'].unique())
    noise_levels = sorted(plot_df['noise'].unique())

    def build_tab_content(noise):
        noise_df = plot_df[plot_df['noise'] == noise]
        rows = []
        style_data_conditional = []

        for row_idx, algo_row in enumerate(algos):
            row = {'Algorithm': algo_row}
            samples_row = noise_df[noise_df['algo_name'] == algo_row][eval_col].dropna().values
            for algo_col in algos:
                if algo_row == algo_col:
                    row[algo_col] = '-'
                else:
                    samples_col = noise_df[noise_df['algo_name'] == algo_col][eval_col].dropna().values
                    if len(samples_row) < 2 or len(samples_col) < 2:
                        row[algo_col] = 'N/A'
                    else:
                        _, p = mannwhitneyu(samples_row, samples_col, alternative='two-sided')
                        row[algo_col] = f'{p:.4f}'
                        if p < 0.05:
                            style_data_conditional.append({
                                'if': {'row_index': row_idx, 'column_id': algo_col},
                                'backgroundColor': '#d4edda',
                                'fontWeight': 'bold',
                            })
            rows.append(row)

        columns = [{'name': 'Algorithm', 'id': 'Algorithm'}] + [{'name': a, 'id': a} for a in algos]

        return dash_table.DataTable(
            data=rows,
            columns=columns,
            style_data_conditional=style_data_conditional,
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0', 'textAlign': 'center'},
            style_cell={'textAlign': 'center', 'padding': '6px 12px', 'fontFamily': 'monospace'},
            style_table={'marginBottom': '8px'},
        )

    tabs = dcc.Tabs(
        children=[
            dcc.Tab(
                label=f'Noise = {noise}',
                children=[build_tab_content(noise)],
                style=tab_style,
                selected_style=tab_selected_style,
            )
            for noise in noise_levels
        ]
    )

    evals_label = 'Evaluations to Best Found Fitness' if use_best else 'Runtime n_evals'
    return html.Div([
        html.H5(f"Mann-Whitney U-Test: Pairwise p-values ({evals_label}, two-sided)",
                style={'marginTop': '16px', 'marginBottom': '4px'}),
        html.P("Green highlight indicates p < 0.05 (statistically significant difference). Each tab shows one noise level.",
               style={'color': '#555', 'fontSize': '12px', 'marginBottom': '8px'}),
        tabs,
        html.Hr(),
    ])


# ---------- Render 2D plot content in tabbed view ----------
@app.callback(
    Output('2DPlotTabContent', 'children'),
    Input('2DPlotTabSelection', 'value')
)
def render_content_2DPlot_tab(tab):
    if tab == 'p1':
        return html.Div([
            dcc.Graph(id='2DLinePlot'),
        ])
    elif tab == 'p2':
        return html.Div([
            # dcc.Graph(id='2DBoxPlot'),
            dcc.Graph(id='2DBoxPlot', style={'width': '800px', 'height': '600px'}),
        ])
    elif tab == 'p3':
        return html.Div([
            dcc.Graph(id='2DLinePlotMO'),
        ])
    elif tab == 'p4':
        return html.Div([
            dcc.Graph(id='2DBoxPlotMO', style={'width': '800px', 'height': '600px'}),
        ])
    elif tab == 'p6':
        return html.Div([
            dcc.Checklist(
                id='line-evals-show-std',
                options=[{'label': ' Show standard deviation (symmetric around mean)', 'value': 'show'}],
                value=[],
            ),
            dcc.Graph(id='2DLinePlotEvalsSO'),
        ])
    elif tab == 'p7':
        return html.Div([
            dcc.Graph(id='2DBoxPlotEvalsSO', style={'width': '800px', 'height': '600px'}),
        ])
    elif tab == 'p8':
        return html.Div([
            dcc.Graph(id='2DBoxPlotMisjudgementsSO', style={'width': '800px', 'height': '600px'}),
        ])
    elif tab == 'p5':
        return html.Div([
            dash_table.DataTable(
                id='plot_2d_data_table',
                columns=[{'name': col, 'id': col} for col in df_no_lists.columns],
                page_size=10,
                data=[],
                style_table={
                    'maxWidth': '100%',  # limits the table to the width of the container
                    'overflowX': 'auto'  # adds a scrollbar if needed
                },
            )
        ])
# ------------------------------
# LON Plots
# ------------------------------

# Store PID from selected LON row so it is accessible from the initial layout
@app.callback(
    Output(LON_TABLE_SELECTED_PID_STORE, 'data'),
    Input("LON_table", "selected_rows"),
    State("LON_table", "data"),
    prevent_initial_call=True
)
def update_lon_table_selected_pid(selected_rows, lon_table_data):
    if selected_rows and lon_table_data:
        return lon_table_data[selected_rows[0]].get("PID")
    return None

# Filter LON dataframe by selected problem using table 1 selection
@app.callback(
    Output('LON_data', 'data'),
    Input("LON_table", "selected_rows"),
    State("LON_table", "data")
)
def update_filtered_view(selected_rows, LON_table_data):
    if not selected_rows:
        blank_df = pd.DataFrame(columns=df_LONs.columns)
        return blank_df.to_dict('records')

    selected_data = [LON_table_data[i] for i in selected_rows]

    # columns needed for plotting; feasibility may be absent for regular LONs
    LON_plotting_cols = [
        'local_optima', 'fitness_values', 'edges',
        'optima_feasibility', 'neighbour_feasibility',
    ]

    df_filtered = pd.DataFrame(selected_data)
    mask = pd.Series(True, index=df_LONs.index)
    for col in LON_display_columns:
        if col in df_filtered.columns:
            allowed = df_filtered[col].unique()
            if any(pd.isnull(allowed)):
                mask &= (df_LONs[col].isin(allowed) | df_LONs[col].isnull())
            else:
                mask &= df_LONs[col].isin(allowed)

    df_result = df_LONs[mask]
    # tolerate missing feasibility columns
    df_result = df_result.loc[:, [c for c in LON_plotting_cols if c in df_result.columns]]
    rows = df_result.to_dict('records')

    combined = {
        "local_optima": [],
        "fitness_values": [],
        "edges": {},
        # feasibility maps (string-keyed for JSON)
        "opt_feas_map": {},    # "1,0,1,..." -> 0/1
        "neigh_feas_map": {},  # "1,0,1,..." -> float in [0,1]
    }

    def key_str_from_opt(opt):
        # opt is a list/tuple of bits
        return ",".join(str(int(x)) for x in opt)

    for row in rows:
        los = row.get("local_optima", [])
        fvs = row.get("fitness_values", [])
        feas_raw = row.get("optima_feasibility")
        feas_list = feas_raw if isinstance(feas_raw, list) else [0] * len(los)
        neigh_raw = row.get("neighbour_feasibility")
        neigh_list = neigh_raw if isinstance(neigh_raw, list) else [0.0] * len(los)

        combined["local_optima"].extend(los)
        combined["fitness_values"].extend(fvs)

        # merge edges (weights) — keep tuple internally then convert in split-format helper
        for (source, target), weight in row.get("edges", {}).items():
            source = tuple(source)
            target = tuple(target)
            combined["edges"][(source, target)] = combined["edges"].get((source, target), 0) + weight

        # fill feasibility maps (string keys for JSON safety)
        for opt, feas, neigh in zip(los, feas_list, neigh_list):
            k = key_str_from_opt(opt)
            combined["opt_feas_map"].setdefault(k, int(feas))
            combined["neigh_feas_map"].setdefault(k, float(neigh))

    # your helper expects only core keys; convert & then attach the maps
    payload_for_split = {
        "local_optima": combined["local_optima"],
        "fitness_values": combined["fitness_values"],
        "edges": combined["edges"],
    }
    dict_result_SE = convert_to_split_edges_format(payload_for_split)

    # attach JSON-safe feasibility maps
    dict_result_SE["opt_feas_map"] = combined["opt_feas_map"]
    dict_result_SE["neigh_feas_map"] = combined["neigh_feas_map"]

    return dict_result_SE

@app.callback(
    Output('STN_data', 'data'),
    Input("table2", "selected_rows"),
    State("table2", "data")
)
def update_filtered_view(selected_rows, table2_data):
    if not selected_rows:
        blank_df = pd.DataFrame(columns=df.columns)
        return blank_df.to_dict('records')

    # rows selected in table2 (deduped algorithm rows)
    selected_data = [table2_data[i] for i in selected_rows]
    df_selected = pd.DataFrame(selected_data)

    # Only filter by the identity keys for an algorithm series
    KEYS = ["PID", "fit_func", "algo_type", "algo_name", "noise", "experiment_name"]

    mask = pd.Series(True, index=df.index)
    for col in KEYS:
        # skip if missing for any reason
        if col not in df_selected.columns or col not in df.columns:
            continue
        allowed = df_selected[col].dropna().unique()
        mask &= df[col].isin(allowed)

    df_result = df[mask]

    print(
        f"[STN filter] selected_rows={selected_rows} -> matched_rows={len(df_result)}",
        flush=True
    )
    return df_result.to_dict('records')

@app.callback(
    [Output('STN_data_processed', 'data'),
     Output('STN_series_labels', 'data'),
     Output('noisy_fitnesses_data', 'data'),
     Output('STN_MO_data', 'data'),
     Output('STN_MO_series_labels', 'data'),
     Output('MO_data_PPP', 'data')],
    [Input('STN_data', 'data'),
     Input('mo_plot_type', 'value')],
)
def process_STN_data(df, mo_plot_type, group_cols=['algo_name', 'noise']):
    print('Processing data...', flush=True)
    df = pd.DataFrame(df)
    STN_data, STN_series, Noise_data = [], [], []
    MO_data, MO_series = [], []
    MO_data_PPP = []

    if df.empty:
        return STN_data, STN_series, Noise_data, MO_data, MO_series, MO_data_PPP

    # default/fallback if somehow empty
    mode = mo_plot_type or 'npnhv'

    grouped = df.groupby(group_cols)
    
    required = {'rep_sols','rep_fits','rep_noisy_fits','sol_iterations','sol_transitions'}
    has_required = required.issubset(df.columns)

    for group_key, group_df in grouped:
        runs = []

        for _, row in group_df.iterrows():
            if not has_required:
                continue
            if row['rep_sols'] is None:
                continue

            runs.append([
                row['rep_sols'],
                row['rep_fits'],
                row['rep_noisy_fits'],
                row['sol_iterations'],
                row['sol_transitions'],
                [],                                             # index 5: noisy_sol_variants removed
                row.get('rep_fitness_boxplot_stats', []),       # index 6: replaces noisy_variant_fitnesses
                row.get('rep_noisy_sols', []),
                row.get('rep_estimated_fits_whenadopted', []),
                row.get('rep_estimated_fits_whendiscarded', []),
                row.get('count_estimated_fits_whenadopted', []),
                row.get('count_estimated_fits_whendiscarded', []),
                row.get('sol_iterations_evals', []),      # index 12: evals-based iterations
                row.get('alternative_rep_sols', []),      # index 13: alt representation solutions
                row.get('alternative_rep_fits', []),      # index 14: alt representation fitnesses
            ])

        STN_data.append(runs)
        STN_series.append(group_key)

        # --- MO runs (mode-dependent) ---
        mo_runs = []
        mo_runs_full = []
        for _, row in group_df.iterrows():
            pareto_solutions = (row.get('pareto_solutions') or [])
            nps_hv_noisy         = (row.get('noisy_pf_noisy_hypervolumes') or [])
            nps_hv_true          = (row.get('noisy_pf_true_hypervolumes') or [])
            true_pareto_solutions = (row.get('true_pareto_solutions') or [])
            tps_hv_true         = (row.get('true_pf_hypervolumes') or [])
            nps_noisy_fits      = (row.get('pareto_fitnesses') or [])
            nps_clean_fits      = (row.get('pareto_true_fitnesses') or [])
            tps_clean_fits      = (row.get('true_pareto_fitnesses') or [])

            Gmax = min(len(pareto_solutions), len(nps_hv_noisy))

            fronts = []
            fronts_full = []
            for g in range(Gmax):
                if mode == 'bpbhv': # Both pareto front sets & both metrics
                    fronts.append({
                        'front1':   true_pareto_solutions[g],
                        'front2':   pareto_solutions[g],
                        'metric1':  tps_hv_true[g],
                        'metric2':  nps_hv_noisy[g],
                        'gen_idx':  g,
                    })
                elif mode == 'tpthv':
                    fronts.append({
                        'front1':   true_pareto_solutions[g],
                        'front2':   None,
                        'metric1':  tps_hv_true[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                elif mode == 'npthv':
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  nps_hv_true[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                elif mode == 'npbhv':
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  nps_hv_true[g],
                        'metric2':  nps_hv_noisy[g],
                        'gen_idx':  g,
                    })
                elif mode == 'tpbhv':
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  tps_hv_true[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                else:  # npnhv
                    fronts.append({
                        'front1':   pareto_solutions[g],
                        'front2':   None,
                        'metric1':  nps_hv_noisy[g],
                        'metric2':  None,
                        'gen_idx':  g,
                    })
                fronts_full.append({
                    'algo_front_solutions': pareto_solutions[g],
                    'algo_front_noisy_fitnesses': nps_noisy_fits[g],
                    'algo_front_clean_fitnesses': nps_clean_fits[g],
                    'algo_front_noisy_hypervolume': nps_hv_noisy[g],
                    'algo_front_clean_hypervolume': nps_hv_true[g],
                    'clean_front_solutions': true_pareto_solutions[g],
                    'clean_front_fitnesses': tps_clean_fits[g],
                    'clean_front_hypervolume': tps_hv_true[g],
                    'gen_idx':  g,
                })
            if fronts:
                mo_runs.append(fronts)
            if fronts_full:
                mo_runs_full.append(fronts_full)

        print(f'generations in data: {Gmax}', flush=True)
        MO_data.append(mo_runs)
        MO_series.append(group_key)
        MO_data_PPP.append(mo_runs_full)
    print('Data processing complete', flush=True)
    return STN_data, STN_series, Noise_data, MO_data, MO_series, MO_data_PPP

@app.callback(
    Output('run-print-info', 'style'),
    Input('show-text-info', 'value')
)
def toggle_run_print_info(value):
    from .layout.styles import MONOSPACE_STYLE
    if value and 'show' in value:
        return MONOSPACE_STYLE
    return {**MONOSPACE_STYLE, 'display': 'none'}


@app.callback(
    Output('print_STN_series_labels', "children"),
    Input('STN_series_labels', 'data')
)
def update_table2_selected(series_list):
    if not series_list:
        return "No rows selected in Table 2."
    # series_labels = [series_list[i] for i in series_list]
    return f"Plotted series: {series_list}"

# ==========
# main plot callbacks
# ==========

# callback for custom axis values
@app.callback(
    Output('axis-values', 'data'),
    [Input('custom_x_min', 'value'),
     Input('custom_x_max', 'value'),
     Input('custom_y_min', 'value'),
     Input('custom_y_max', 'value'),
     Input('custom_z_min', 'value'),
     Input('custom_z_max', 'value'),
     Input('log-z-axis', 'value')]
)
def clean_axis_values(custom_x_min, custom_x_max, custom_y_min, custom_y_max, custom_z_min, custom_z_max, log_z):
    def clean(val):
        # If the input is empty, an empty string, or None, return None.
        # Otherwise, assume it's a valid number (or you could cast to float/int).
        if val in [None, ""]:
            return None
        return val  # or float(val) if needed
    
    return {
        "custom_x_min": clean(custom_x_min),
        "custom_x_max": clean(custom_x_max),
        "custom_y_min": clean(custom_y_min),
        "custom_y_max": clean(custom_y_max),
        "custom_z_min": clean(custom_z_min),
        "custom_z_max": clean(custom_z_max),
        "log_z": 'log_z' in (log_z or []),
    }

def _build_stn_stats_table(stn_algo_data, stn_labels, problem_goal='maximise'):
    """Build the STN stats DataTable from per-algorithm trajectory data."""
    if not stn_algo_data:
        return html.Div()

    def fmt(values):
        if not values:
            return 'N/A'
        if len(values) == 1:
            return str(values[0]) if values[0] is not None else 'N/A'
        return '[' + ','.join(str(v) if v is not None else 'N/A' for v in values) + ']'

    minimising = str(problem_goal or 'maximise')[:3].lower() == 'min'

    def count_misjudgements(entry):
        fits = entry[1] if entry and len(entry) > 1 and entry[1] is not None else []
        if minimising:
            return sum(1 for i in range(len(fits) - 1) if fits[i + 1] > fits[i])
        return sum(1 for i in range(len(fits) - 1) if fits[i + 1] < fits[i])

    rows = []
    for idx, selected_trajectories, all_run_trajectories in stn_algo_data:
        label = stn_labels[idx] if stn_labels and idx < len(stn_labels) else [f'Algo {idx}']
        algo_name = label[0] if len(label) > 0 else f'Algo {idx}'
        noise = label[1] if len(label) > 1 else '?'
        algo_name = f'{algo_name} noise={noise}'

        nodes_per_run = [
            len(entry[0]) for entry in selected_trajectories
            if entry and len(entry) > 0 and entry[0] is not None
        ]

        all_node_counts = [
            len(entry[0]) for entry in all_run_trajectories
            if entry and len(entry) > 0 and entry[0] is not None
        ]
        mean_nodes = round(sum(all_node_counts) / len(all_node_counts)) if all_node_counts else 'N/A'

        avg_evals_per_run = []
        for entry in selected_trajectories:
            sol_evals = entry[12] if entry and len(entry) > 12 else None
            if sol_evals:
                avg_evals_per_run.append(round(sum(sol_evals) / len(sol_evals)))
            else:
                avg_evals_per_run.append(None)

        all_evals_flat = [
            e for entry in all_run_trajectories
            if entry and len(entry) > 12 and entry[12]
            for e in entry[12]
        ]
        mean_avg_evals = round(sum(all_evals_flat) / len(all_evals_flat)) if all_evals_flat else 'N/A'

        misjudgements_per_run = [count_misjudgements(e) for e in selected_trajectories]
        all_misjudgement_counts = [count_misjudgements(e) for e in all_run_trajectories]
        mean_misjudgements = round(sum(all_misjudgement_counts) / len(all_misjudgement_counts)) \
            if all_misjudgement_counts else 'N/A'

        rows.append({
            'algo': algo_name,
            'nodes_rendered': fmt(nodes_per_run),
            'mean_nodes_all': str(mean_nodes),
            'avg_evals_rendered': fmt(avg_evals_per_run),
            'mean_avg_evals_all': str(mean_avg_evals),
            'misjudgements_rendered': fmt(misjudgements_per_run),
            'mean_misjudgements_all': str(mean_misjudgements),
        })

    if not rows:
        return html.Div()

    return dash_table.DataTable(
        columns=[
            {'name': 'Algorithm', 'id': 'algo'},
            {'name': 'Nodes (rendered runs)', 'id': 'nodes_rendered'},
            {'name': 'Mean nodes (all runs)', 'id': 'mean_nodes_all'},
            {'name': 'Avg evals/node (rendered runs)', 'id': 'avg_evals_rendered'},
            {'name': 'Avg evals/node (all runs)', 'id': 'mean_avg_evals_all'},
            {'name': 'Misjudgements (rendered runs)', 'id': 'misjudgements_rendered'},
            {'name': 'Mean misjudgements (all runs)', 'id': 'mean_misjudgements_all'},
        ],
        data=rows,
        style_table={'width': '1300px'},
        style_cell={'textAlign': 'center', 'padding': '8px'},
        style_header={'fontWeight': 'bold'},
    )


# print mode: uncheck info panel, set scale defaults
@app.callback(
    [Output('annotation-options', 'value'),
     Output('axes-text-scale', 'value'),
     Output('annotation-text-scale', 'value')],
    Input('annotation-options', 'value'),
    prevent_initial_call=True,
)
def handle_print_mode(annotation_options):
    options = annotation_options or []
    if 'print-mode' in options:
        options = [o for o in options if o != 'annotate-info-panel']
        return options, 1.2, 2
    return options, dash.no_update, dash.no_update


# callback for main plot
@app.callback(
    [Output('trajectory-plot', 'figure'),
     Output('run-print-info', 'children'),
     Output('stn-stats-table', 'children'),
     Output('lon-stats-table', 'children')],
    [Input("optimum", "data"),
     Input("PID", "data"),
     Input("opt_goal", "data"),
     Input('options', 'value'),
     Input('run-options', 'value'),
     Input('STN_lower_fit_limit', 'value'),
     Input('LON-fit-percent', 'value'),
     Input('LON-options', 'value'),
     Input('LON-node-colour-mode', 'value'), # CoLON colour
     Input('LON-edge-colour-feas', 'value'), # CoLON colour
     Input('lmds-multiplier', 'value'),
     Input('NLON_fit_func', 'value'),
     Input('NLON_intensity', 'value'),
     Input('NLON_samples', 'value'),
     Input('layout', 'value'),
     Input('plotType', 'value'),
     Input('hover-info', 'value'),
     Input('azimuth_deg', 'value'),
     Input('elevation_deg', 'value'),
     Input('STN_data_processed', 'data'),
     Input('STN_series_labels', 'data'),
     Input('run-index', 'value'),
     Input('run-selector', 'value'),
     Input('LON_data', 'data'),
     Input('axis-values', 'data'),
     Input('opacity_noise_bar', 'value'),
     Input('LON_node_opacity', 'value'),
     Input('LON_edge_opacity', 'value'),
     Input('STN_node_opacity', 'value'),
     Input('STN_edge_opacity', 'value'),
     Input('STN-node-min', 'value'),
     Input('STN-node-max', 'value'),
     Input('LON-node-min', 'value'),
     Input('LON-node-max', 'value'),
     Input('LON-edge-size-slider', 'value'),
     Input('STN-edge-size-slider', 'value'),
     Input('noisy_fitnesses_data', 'data'),
     Input('stn-plot-type', 'value'),
     Input('STN_MO_data', 'data'),
     Input('STN_MO_series_labels', 'data'),
     Input('stn-node-size-metric', 'value'),
     Input('annotation-options', 'value'),
     Input('fit_func_store', 'data'),
     Input('info-panel-x', 'value'),
     Input('info-panel-y', 'value'),
     Input('axes-text-scale', 'value'),
     Input('annotation-text-scale', 'value'),
     Input('plot-theme', 'value')]
)
def update_plot(optimum, PID, opt_goal, options, run_options, STN_lower_fit_limit,
                LO_fit_percent, LON_options, LON_node_colour_mode, LON_edge_colour_feas,
                lmds_multiplier, NLON_fit_func, NLON_intensity, NLON_samples, layout_value, plot_type,
                hover_info_value, azimuth_deg, elevation_deg, all_trajectories_list, STN_labels,
                run_start_index, n_runs_display, local_optima, axis_values,
                opacity_noise_bar, LON_node_opacity, LON_edge_opacity, STN_node_opacity, STN_edge_opacity,
                STN_node_min, STN_node_max, LON_node_min, LON_node_max,
                LON_edge_size_slider, STN_edge_size_slider, noisy_fitnesses_list,
                stn_plot_type, STN_MO_data, STN_MO_series_labels, stn_node_size_metric,
                annotation_options, fit_func, info_panel_x, info_panel_y,
                axes_text_scale, annotation_text_scale, plot_theme):
    """
    Main visualization callback - orchestrates the visualization pipeline.

    This refactored callback delegates to specialized modules in src/visualization/
    for graph building, node styling, positioning, and trace creation.
    """
    print('\033[1m\033[31mCreating new Plot...\033[0m', flush=True)

    # ==========
    # STEP 1: Parse all callback inputs into configuration object
    # ==========
    config = parse_callback_inputs(
        optimum=optimum,
        pid=PID,
        opt_goal=opt_goal,
        options=options or [],
        run_options=run_options or [],
        stn_lower_fit_limit=STN_lower_fit_limit,
        lo_fit_percent=LO_fit_percent,
        lon_options=LON_options or [],
        lon_node_colour_mode=LON_node_colour_mode,
        lon_edge_colour_feas=LON_edge_colour_feas or [],
        lmds_multiplier=lmds_multiplier,
        nlon_fit_func=NLON_fit_func,
        nlon_intensity=NLON_intensity,
        nlon_samples=NLON_samples,
        layout_value=layout_value,
        plot_type=plot_type,
        hover_info_value=hover_info_value,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        run_start_index=run_start_index,
        n_runs_display=n_runs_display,
        axis_values=axis_values or {},
        opacity_noise_bar=opacity_noise_bar,
        lon_node_opacity=LON_node_opacity,
        lon_edge_opacity=LON_edge_opacity,
        stn_node_opacity=STN_node_opacity,
        stn_edge_opacity=STN_edge_opacity,
        stn_node_min=STN_node_min,
        stn_node_max=STN_node_max,
        lon_node_min=LON_node_min,
        lon_node_max=LON_node_max,
        lon_edge_size_slider=LON_edge_size_slider,
        stn_edge_size_slider=STN_edge_size_slider,
        stn_plot_type=stn_plot_type,
        node_size_metric=stn_node_size_metric or 'generations',
        colorscale=plot_theme or 'Viridis',
    )

    # Apply viridis color palette for series if enabled
    if config.stn.use_viridis:
        n_series = len(STN_labels or STN_MO_series_labels or [])
        if n_series > 0:
            positions = [i / max(n_series - 1, 1) for i in range(n_series)]
            config.algo_colors = [px.colors.sample_colorscale(config.colorscale, p)[0] for p in positions]

    # ==========
    # STEP 2: Initialize graph and node mappings
    # ==========
    G = nx.MultiDiGraph()
    stn_node_mapping = {}
    lon_node_mapping = {}
    debug_summaries = []
    stn_algo_data = []
    node_noise = {}
    fitness_dict = {}

    # ==========
    # STEP 3: Build graph - add nodes and edges
    # ==========

    # If 'evaluations' node size metric is selected, swap sol_iterations (index 3)
    # with sol_iterations_evals (index 12) in each run entry before graph building.
    if config.node_size_metric == 'evaluations' and all_trajectories_list:
        for series_entries in all_trajectories_list:
            for entry in series_entries:
                if len(entry) > 12 and entry[12]:
                    entry[3] = entry[12]

    if config.stn_plot_type == 'multiobjective':
        # Multi-objective mode
        print('ADDING NODES IN MULTIOBJECTIVE MODE')
        for idx, mo_runs in enumerate(STN_MO_data or []):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]
            selected_runs = []
            if config.n_runs_display > 0:
                selected_runs.extend(mo_runs[config.run_start_index:config.run_start_index + config.n_runs_display])
            add_mo_fronts(G, selected_runs, edge_color, idx, config.noisy_node_color)

        debug_summary_component = html.Div("None implemented for MO")

    elif config.stn_plot_type == 'prior_v4' and all_trajectories_list:
        # Prior noise STN V4 mode
        print('ADDING NODES IN PRIOR NOISE STN V4 MODE')
        optimisation_goal = opt_goal[:3].lower() if opt_goal else 'max'

        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]

            selected_trajectories = []
            if config.n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[config.run_start_index:config.run_start_index + config.n_runs_display])
            if config.show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if config.show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if config.show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if config.show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            add_prior_noise_stn_v4(
                G, selected_trajectories, edge_color, idx, config.noisy_node_color,
                dedup=config.stn.dedup_prior_noise,
                show_alt_rep=config.stn.show_alt_rep,
                show_alt_rep_no_fit=config.stn.show_alt_rep_no_fit,
                stn_node_min=config.node_size.stn_min,
            )

            stn_algo_data.append((idx, selected_trajectories, all_run_trajectories))
            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)

    elif config.stn_plot_type == 'prior_v5' and all_trajectories_list:
        # Prior noise STN V5 mode
        print('ADDING NODES IN PRIOR NOISE STN V5 MODE')
        optimisation_goal = opt_goal[:3].lower() if opt_goal else 'max'

        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]

            selected_trajectories = []
            if config.n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[config.run_start_index:config.run_start_index + config.n_runs_display])
            if config.show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if config.show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if config.show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if config.show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            add_prior_noise_stn_v5(
                G, selected_trajectories, edge_color, idx, config.noisy_node_color,
                dedup=config.stn.dedup_prior_noise,
                show_alt_rep=config.stn.show_alt_rep,
                show_alt_rep_no_fit=config.stn.show_alt_rep_no_fit,
                stn_node_min=config.node_size.stn_min,
            )

            stn_algo_data.append((idx, selected_trajectories, all_run_trajectories))
            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)

    elif all_trajectories_list:
        # Single-objective STN mode
        optimisation_goal = opt_goal[:3].lower() if opt_goal else 'max'

        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]

            selected_trajectories = []
            if config.n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[config.run_start_index:config.run_start_index + config.n_runs_display])
            if config.show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if config.show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if config.show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if config.show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            stn_node_mapping = add_stn_trajectories(
                G, selected_trajectories, edge_color, idx, stn_node_mapping, config
            )

            stn_algo_data.append((idx, selected_trajectories, all_run_trajectories))
            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        # Create debug summary component
        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)
    else:
        debug_summary_component = html.Div("No trajectory data available.")

    stn_stats_table = _build_stn_stats_table(stn_algo_data, STN_labels, problem_goal=opt_goal)

    print('STN TRAJECTORIES ADDED')
    debug_mo_counts(G, by="run_idx", label="[MO]", list_fronts=True, max_list=50)

    # ==========
    # STEP 4: Add LON nodes and edges (if provided)
    # ==========
    opt_feas_map = {}
    neigh_feas_map = {}

    if local_optima:
        # Extract CoLON colour maps
        if isinstance(local_optima, dict):
            opt_feas_map = local_optima.get("opt_feas_map", {}) or {}
            neigh_feas_map = local_optima.get("neigh_feas_map", {}) or {}

        # Convert and filter local optima data
        local_optima_processed = convert_to_single_edges_format(local_optima)
        local_optima_processed = filter_local_optima(local_optima_processed, config.lon.fit_percent)
        if config.lon.filter_negative:
            local_optima_processed = filter_negative_LO(local_optima_processed)

        # Add LON nodes
        lon_node_mapping, node_noise = add_lon_nodes(
            G, local_optima_processed, lon_node_mapping, config, PID
        )
        fitness_dict = {node: data['fitness'] for node, data in G.nodes(data=True)}

        # Add LON edges
        add_lon_edges(G, local_optima_processed, lon_node_mapping, config, opt_feas_map)
        print('LOCAL OPTIMA ADDED')

    # ==========
    # STEP 5: Apply node styling (sizes and colors)
    # ==========
    style_nodes(G, config, opt_feas_map, neigh_feas_map)

    # ==========
    # STEP 6: Calculate statistics
    # ==========
    calculate_lon_statistics(G, verbose=True)

    # ==========
    # STEP 7: Calculate node positions
    # ==========
    pos = calculate_positions(G, config.layout_type, config.stn_plot_type, config.plot_3d, config.lon.lmds_multiplier)

    # ==========
    # STEP 8: Build visualization traces
    # ==========
    if config.plot_type in ('RegLon', 'NLon_box'):
        print('CREATING PLOT...')
        traces = build_all_traces(G, pos, config, node_noise, fitness_dict)

        # ==========
        # STEP 9: Configure axes and create figure
        # ==========
        xaxis_settings, yaxis_settings, zaxis_settings = create_axis_settings(
            G, pos, config, node_noise, axes_text_scale=axes_text_scale or 1.0)

        # Build scene annotations from enabled annotation options
        ann_font_size = round(11 * (annotation_text_scale or 1.0))
        scene_annotations = []
        if 'annotate-start-nodes' in (annotation_options or []):
            seen_positions = set()
            for node, attr in G.nodes(data=True):
                if attr.get('start_node') and node in pos:
                    x, y = pos[node][:2]
                    z = attr.get('fitness', 0)
                    pos_key = (round(x, 6), round(y, 6), round(z, 6))
                    if pos_key not in seen_positions:
                        seen_positions.add(pos_key)
                        scene_annotations.append(dict(
                            x=x, y=y, z=z,
                            text='Start node',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='black',
                            ax=80, ay=0,
                            font=dict(size=ann_font_size, color='black'),
                        ))

        if 'annotate-end-nodes' in (annotation_options or []):
            seen_positions = set()
            for node, attr in G.nodes(data=True):
                if attr.get('end_node') and node in pos:
                    if config.optimum is not None and attr.get('fitness') == config.optimum:
                        continue
                    x, y = pos[node][:2]
                    z = attr.get('fitness', 0)
                    pos_key = (round(x, 6), round(y, 6), round(z, 6))
                    if pos_key not in seen_positions:
                        seen_positions.add(pos_key)
                        scene_annotations.append(dict(
                            x=x, y=y, z=z,
                            text='End node',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='black',
                            ax=80, ay=0,
                            font=dict(size=ann_font_size, color='black'),
                        ))

        if 'annotate-mistakes' in (annotation_options or []):
            seen_positions = set()
            maximizing = (config.opt_goal or 'max')[:3].lower() == 'max'
            for u, v, edge_attr in G.edges(data=True):
                if not edge_attr.get('edge_type', '').startswith('STN'):
                    continue
                if v not in pos or 'Noisy' in v or G.nodes[v].get('type') == 'STN_ALT':
                    continue
                fit_u = G.nodes[u].get('fitness')
                fit_v = G.nodes[v].get('fitness')
                if fit_u is None or fit_v is None:
                    continue
                is_decline = fit_v < fit_u if maximizing else fit_v > fit_u
                if is_decline:
                    x, y = pos[v][:2]
                    z = fit_v
                    pos_key = (round(x, 6), round(y, 6), round(z, 6))
                    if pos_key not in seen_positions:
                        seen_positions.add(pos_key)
                        scene_annotations.append(dict(
                            x=x, y=y, z=z,
                            text='',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='red',
                            ax=20, ay=0,
                            font=dict(size=ann_font_size, color='red'),
                        ))

        if 'annotate-optimum' in (annotation_options or []) and config.optimum is not None:
            for node, attr in G.nodes(data=True):
                if attr.get('fitness') == config.optimum and 'Noisy' not in node and node in pos:
                    x, y = pos[node][:2]
                    z = attr.get('fitness', 0)
                    scene_annotations.append(dict(
                        x=x, y=y, z=z,
                        text='Global optimum',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor='black',
                        ax=40, ay=-40,
                        font=dict(size=ann_font_size, color='black'),
                    ))
                    break  # Only one global optimum

        fig = create_figure(traces, config, xaxis_settings, yaxis_settings, zaxis_settings,
                            scene_annotations=scene_annotations or None)

        # Add 2D info panel annotation in top-right corner
        annotation_opts = annotation_options or []
        print_mode = 'print-mode' in annotation_opts
        if 'annotate-info-panel' in annotation_opts or print_mode:
            lines = []
            if not print_mode:
                if PID:
                    lines.append(f'<b>PID:</b> {PID}')
                if fit_func:
                    lines.append(f'<b>Fitness:</b> {fit_func}')
                    lines.append('&#9679; True solution/fitness')
                    lines.append('&#9632; Noisy solution/fitness')
                    lines.append('<i>Node size = evals at node</i>')

            if STN_labels:
                if lines:
                    lines.append('')  # blank line separator
                for idx, label in enumerate(STN_labels):
                    algo_name = label[0] if len(label) > 0 else '?'
                    noise = label[1] if len(label) > 1 else '?'
                    color = config.algo_colors[idx % len(config.algo_colors)]
                    lines.append(f'<span style="color:{color}"><b>{algo_name}</b> noise={noise}</span>')
            if lines:
                annotation_kwargs = dict(
                    xref='paper', yref='paper',
                    x=(info_panel_x if info_panel_x is not None else 90) / 100,
                    y=(info_panel_y if info_panel_y is not None else 75) / 100,
                    xanchor='right', yanchor='top',
                    text='<br>'.join(lines),
                    showarrow=False,
                    align='right',
                    font=dict(size=ann_font_size),
                )
                if not print_mode:
                    annotation_kwargs.update(
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='grey',
                        borderwidth=1,
                    )
                fig.add_annotation(**annotation_kwargs)
    else:
        # Fallback for other plot types
        fig = go.Figure()

    # Build LON stats table
    lon_nodes = [n for n in G.nodes() if "Local Optimum" in n]
    total_optima = len(lon_nodes)
    if total_optima > 0:
        if opt_feas_map:
            feasible = sum(
                1 for n in lon_nodes
                if opt_feas_map.get(",".join(str(int(x)) for x in G.nodes[n].get('solution', []))) == 1
            )
            infeasible = sum(
                1 for n in lon_nodes
                if opt_feas_map.get(",".join(str(int(x)) for x in G.nodes[n].get('solution', []))) == 0
            )
            feas_display, infeas_display = str(feasible), str(infeasible)
        else:
            feas_display, infeas_display = 'N/A', 'N/A'
        maximizing = (config.opt_goal or 'max')[:3].lower() == 'max'
        best_fitness = max(G.nodes[n].get('fitness', float('-inf')) for n in lon_nodes) if maximizing \
            else min(G.nodes[n].get('fitness', float('inf')) for n in lon_nodes)
        global_opt_weight = sum(
            G.nodes[n].get('weight', 0)
            for n in lon_nodes
            if G.nodes[n].get('fitness') == best_fitness
        )
        lon_stats_table = dash_table.DataTable(
            columns=[
                {'name': 'Total Optima', 'id': 'total'},
                {'name': 'Feasible Optima', 'id': 'feasible'},
                {'name': 'Infeasible Optima', 'id': 'infeasible'},
                {'name': 'Global Optima Weight', 'id': 'go_weight'},
            ],
            data=[{
                'total': total_optima,
                'feasible': feas_display,
                'infeasible': infeas_display,
                'go_weight': global_opt_weight,
            }],
            style_table={'width': '700px'},
            style_cell={'textAlign': 'center', 'padding': '8px'},
            style_header={'fontWeight': 'bold'},
        )
    else:
        lon_stats_table = html.Div()

    return fig, debug_summary_component, stn_stats_table, lon_stats_table

@app.callback(
    Output("plotParetoFront", "figure"),
    [Input('MO_data_PPP', 'data'),
     Input('STN_MO_series_labels', 'data'),
     Input('paretoFrontPlotType', 'value'),
     Input('IndVsDist_IndType', 'value'),
     Input('IndVsDist_DistType', 'value'),
     Input('paretoPlotNumRuns', 'value'),
     Input('paretoPlotWindowSize', 'value')]
)
def updateParetoPlot(frontdata, series_labels, paretoFrontPlotType, IndVsDist_IndType, IndVsDist_DistType, nruns, windowSize):
    """
    Update the Pareto front plot based on the selected plot type.
    Uses the plotting registry for dynamic dispatch.
    """
    plot_func = get_pareto_plot(paretoFrontPlotType)
    if plot_func is None:
        return go.Figure()

    # Handle plot-specific arguments
    if paretoFrontPlotType == 'SubplotsMulti':
        return plot_func(frontdata, series_labels, nruns=nruns)
    elif paretoFrontPlotType == 'IndVsDist':
        return plot_func(frontdata, series_labels, distance_method=IndVsDist_DistType, nruns=nruns)
    elif paretoFrontPlotType == 'IGDVsDist':
        return plot_func(frontdata, series_labels, distance_method=IndVsDist_DistType, nruns=nruns)
    elif paretoFrontPlotType == 'MoveCorr':
        return plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType, window=windowSize)
    elif paretoFrontPlotType == 'Hist':
        return plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType)
    elif paretoFrontPlotType == 'Scatter':
        return plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType)
    else:
        # Default case: just pass frontdata and series_labels
        return plot_func(frontdata, series_labels)
    
# ==========
# RUN
# ==========

if __name__ == '__main__':
    # app.run_server(debug=True)
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8050, debug=True, use_reloader=False)