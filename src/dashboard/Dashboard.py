import dash
from dash import html, dcc, dash_table, Input, Output, State
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
from ..plotting.performance import plot2d_line, plot2d_box, plot2d_line_mo, plot2d_box_mo, plot2d_line_evals, plot2d_box_evals

# ==========
# Data Loading
# ==========
from ..dataio import DashboardData, DISPLAY2_HIDDEN_COLUMNS, LON_HIDDEN_COLUMNS

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

app.layout = create_layout(display2_df, display2_hidden_cols)

# ------------------------------
# Callback: Render Problem Tab Content
# ------------------------------

@app.callback(
    Output('problemTabsContent', 'children'),
    [Input('problemTabSelection', 'value'),
     Input('table1-selected-store', 'data'),
     Input('table1tab2-selected-store', 'data')]
)
def render_content_problem_tab(tab, stored_selection_tab1, stored_selection_tab2):
    if tab == 'p1':
        return html.Div([
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "alignItems": "center",
                    "padding": "10px",
                    "marginTop": "0px"
                },
                children=[
                    dash_table.DataTable(
                        id="table1",
                        data=display1_df.to_dict("records"),
                        columns=[{"name": col, "id": col} for col in display1_df.columns],
                        page_size=10,
                        filter_action="native",
                        row_selectable="single",
                        # Use stored selection from Tab 1
                        selected_rows=stored_selection_tab1 if stored_selection_tab1 else [],
                        style_table={"overflowX": "auto"},
                    )
                ]
            ),
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "alignItems": "center",
                    "padding": "10px",
                    "marginTop": "0px"
                },
                children=[
                    dash_table.DataTable(
                        id="LON_table",
                        data=df_LONs[LON_display_columns].to_dict("records"),
                        columns=[{"name": col, "id": col} for col in LON_display_columns],
                        page_size=10,
                        # filter_action="native",
                        row_selectable="single",
                        style_table={"overflowX": "auto"},
                    )
                ]
            )
        ])
    elif tab == 'p2':
        # ADD ANOTHER VERSION OF TAB 1 HERE WITH UPDATED NAMES
        return html.Div([
            html.H3('Content for Tab Two'),
            dash_table.DataTable(
                id="table1_tab2",
                data=display1_df.to_dict("records"),
                columns=[{"name": col, "id": col} for col in display1_df.columns],
                page_size=10,
                filter_action="native",
                row_selectable="single",
                selected_rows=stored_selection_tab2 if stored_selection_tab2 else [],
                style_table={"overflowX": "auto"},
            )
        ])

# ------------------------------
# Callbacks: Update Selection Stores
# ------------------------------

@app.callback(
    Output("table1-selected-store", "data"),
    Input("table1", "selected_rows"),
    prevent_initial_call=True
)
def update_table1_store(selected_rows):
    return selected_rows

@app.callback(
    Output("table1tab2-selected-store", "data"),
    Input("table1_tab2", "selected_rows"),
    prevent_initial_call=True
)
def update_table1tab2_store(selected_rows):
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
    [Input("table1-selected-store", "data"),
     Input("table1tab2-selected-store", "data")]
)
def filter_table2(selection1, selection2):
    union = set()
    # For Table 1 (Tab 1), use display1_df to retrieve PID and fit_func.
    if selection1:
        for idx in selection1:
            if idx < len(display1_df):
                row = display1_df.iloc[idx]
                union.add((row['PID'], row['fit_func']))
                # union.add(display1_df.iloc[idx]['PID'])  # Old: PID only
    # For Table 1 on Tab 2, also use display1_df.
    if selection2:
        for idx in selection2:
            if idx < len(display1_df):
                row = display1_df.iloc[idx]
                union.add((row['PID'], row['fit_func']))
                # union.add(display1_df.iloc[idx]['PID'])  # Old: PID only
    if not union:
        return display2_df.to_dict("records")
    else:
        # Filter by both PID and fit_func
        mask = display2_df.apply(
            lambda r: (r['PID'], r['fit_func']) in union, axis=1
        )
        filtered_df = display2_df[mask]
        # filtered_df = display2_df[display2_df['PID'].isin(union)]  # Old: PID only
        return filtered_df.to_dict("records")

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
    Output("opt_goal", 'data')],
    Input("data-problem-specific", "data")
)
def update_table2(data):
    if data is None:
        return None
    df = pd.DataFrame(data)
    optimum = df["opt_global"].iloc[0]
    PID = df["PID"].iloc[0]
    opt_goal = df["problem_goal"].iloc[0]
    return optimum, PID, opt_goal

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
filter_columns = [col for col in display2_df.columns]
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
    Input('plot_2d_data', 'data')
)
def display_stored_data(data):
    plot_df = pd.DataFrame(data)
    plot = plot2d_line(plot_df)
    return plot
# 2D box plot
@app.callback(
    Output('2DBoxPlot', 'figure'),
    Input('plot_2d_data', 'data')
)
def display_stored_data(data):
    plot_df = pd.DataFrame(data)
    plot = plot2d_box(plot_df)
    return plot

# 2D line plot (multi-objective)
@app.callback(
    Output('2DLinePlotMO', 'figure'),
    Input('plot_2d_data', 'data')
)
def display_stored_data_mo_line(data):
    plot_df = pd.DataFrame(data)
    plot = plot2d_line_mo(plot_df)
    return plot

# 2D box plot (multi-objective)
@app.callback(
    Output('2DBoxPlotMO', 'figure'),
    Input('plot_2d_data', 'data')
)
def display_stored_data_mo_box(data):
    plot_df = pd.DataFrame(data)
    plot = plot2d_box_mo(plot_df)
    return plot

# 2D line plot (evals, single-objective)
@app.callback(
    Output('2DLinePlotEvalsSO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('line-evals-show-std', 'value')
)
def display_line_evals_so(data, std_checkbox):
    plot_df = pd.DataFrame(data)
    show_std = bool(std_checkbox and 'show' in std_checkbox)
    plot = plot2d_line_evals(plot_df, show_std=show_std)
    return plot

# 2D box plot (evals, single-objective)
@app.callback(
    Output('2DBoxPlotEvalsSO', 'figure'),
    Input('plot_2d_data', 'data')
)
def display_box_evals_so(data):
    plot_df = pd.DataFrame(data)
    plot = plot2d_box_evals(plot_df)
    return plot

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

# Filter LON dataframe by selected problem using table 1 selection
@app.callback(
    Output('LON_data', 'data'),
    Input("LON_table", "selected_rows"),
    State("LON_table", "data")
)
def update_filtered_view(selected_rows, LON_table_data):
    if len(selected_rows) == 0:
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
        feas_list = row.get("optima_feasibility", [0]*len(los)) or [0]*len(los)
        neigh_list = row.get("neighbour_feasibility", [0.0]*len(los)) or [0.0]*len(los)

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
    KEYS = ["PID", "fit_func", "algo_type", "algo_name", "noise"]

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
     Input('custom_z_max', 'value')]
)
def clean_axis_values(custom_x_min, custom_x_max, custom_y_min, custom_y_max, custom_z_min, custom_z_max):
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
        "custom_z_max": clean(custom_z_max)
    }

# callback for main plot
@app.callback(
    [Output('trajectory-plot', 'figure'),
     Output('run-print-info', 'children')],
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
     Input('stn-node-size-metric', 'value')]
)
def update_plot(optimum, PID, opt_goal, options, run_options, STN_lower_fit_limit,
                LO_fit_percent, LON_options, LON_node_colour_mode, LON_edge_colour_feas,
                NLON_fit_func, NLON_intensity, NLON_samples, layout_value, plot_type,
                hover_info_value, azimuth_deg, elevation_deg, all_trajectories_list, STN_labels,
                run_start_index, n_runs_display, local_optima, axis_values,
                opacity_noise_bar, LON_node_opacity, LON_edge_opacity, STN_node_opacity, STN_edge_opacity,
                STN_node_min, STN_node_max, LON_node_min, LON_node_max,
                LON_edge_size_slider, STN_edge_size_slider, noisy_fitnesses_list,
                stn_plot_type, STN_MO_data, STN_MO_series_labels, stn_node_size_metric):
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
    )

    # ==========
    # STEP 2: Initialize graph and node mappings
    # ==========
    G = nx.MultiDiGraph()
    stn_node_mapping = {}
    lon_node_mapping = {}
    debug_summaries = []
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
                stn_node_min=config.node_size.stn_min,
            )

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
                stn_node_min=config.node_size.stn_min,
            )

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
    pos = calculate_positions(G, config.layout_type, config.stn_plot_type, config.plot_3d)

    # ==========
    # STEP 8: Build visualization traces
    # ==========
    if config.plot_type in ('RegLon', 'NLon_box'):
        print('CREATING PLOT...')
        traces = build_all_traces(G, pos, config, node_noise, fitness_dict)

        # ==========
        # STEP 9: Configure axes and create figure
        # ==========
        xaxis_settings, yaxis_settings, zaxis_settings = create_axis_settings(G, pos, config, node_noise)
        fig = create_figure(traces, config, xaxis_settings, yaxis_settings, zaxis_settings)
    else:
        # Fallback for other plot types
        fig = go.Figure()

    return fig, debug_summary_component

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