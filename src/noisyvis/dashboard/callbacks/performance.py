"""2D performance tab: the filtered plot data, the SO/MO performance figures, the misjudgement,
performance and evaluation summary tables with their Mann-Whitney tests, and the tab renderer."""

import math
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table, Input, Output

from ..instance import app
from ..data import df_no_lists, display2_df
from ..layout import TAB_STYLE, TAB_SELECTED_STYLE
from ..helpers import _filter_penalty, _get_so_xaxis_label, _get_problem_goal
from ...viz.plots.performance import (
    plot2d_line,
    plot2d_box,
    plot2d_line_mo,
    plot2d_box_mo,
    plot2d_line_evals,
    plot2d_box_evals,
    plot2d_box_penalty,
    plot2d_box_misjudgements_so,
    plot2d_box_advanced_misjudgements_so,
)


# Style constants (tab_style, tab_selected_style) are imported from layout module.
# Local aliases for backward compatibility with callbacks that use lowercase names
tab_style = TAB_STYLE
tab_selected_style = TAB_SELECTED_STYLE

# ---------- Generate data for 2D performance plot by filtering main data with table 2 selection ----------
filter_columns = [col for col in display2_df.columns if col in df_no_lists.columns]
def _filter_by_table2_selection(filtered_data):
    """Filter df_no_lists down to the rows matching table2's derived_virtual_data."""
    if not filtered_data:
        return df_no_lists

    df_filtered = pd.DataFrame(filtered_data)

    mask = pd.Series(True, index=df_no_lists.index)
    for col in filter_columns:
        if col in df_filtered.columns:
            allowed_values = df_filtered[col].unique()
            # If any allowed value is null (None or np.nan), allow null rows.
            if any(pd.isnull(allowed_values)):
                mask &= (df_no_lists[col].isin(allowed_values) | df_no_lists[col].isnull())
            else:
                mask &= df_no_lists[col].isin(allowed_values)

    return df_no_lists[mask]

@app.callback(
    Output('penalty-filter-dropdown', 'options'),
    Input('table2', 'derived_virtual_data')
)
def update_penalty_filter_options(filtered_data):
    base_df = _filter_by_table2_selection(filtered_data)
    if 'penalty' not in base_df.columns:
        return []
    values = sorted(base_df['penalty'].dropna().unique())
    return [{'label': str(v), 'value': v} for v in values]

@app.callback(
    Output('penalty-filter-dropdown', 'value'),
    Input('PID', 'data'),
    prevent_initial_call=True,
)
def reset_penalty_filter_on_problem_change(_pid):
    return None

@app.callback(
    Output('plot_2d_data', 'data'),
    Input('table2', 'derived_virtual_data'),
    Input('penalty-filter-dropdown', 'value'),
)
def update_filtered_view(filtered_data, penalty_value):
    df_result = _filter_penalty(_filter_by_table2_selection(filtered_data), penalty_value)
    return df_result.to_dict('records')

def _cap_noise(plot_df, cap):
    if cap and 'noise' in plot_df.columns:
        plot_df = plot_df[plot_df['noise'] <= cap]
    return plot_df

def _hide_series(plot_df, hidden):
    if hidden and 'algo_name' in plot_df.columns:
        plot_df = plot_df[~plot_df['algo_name'].isin(hidden)]
    return plot_df

def _resolve_evals_column(fitness_mode, plot_df):
    """Pick the evals column and label matching the so-fitness-mode dropdown."""
    if fitness_mode == 'best' and 'evals_to_best' in plot_df.columns:
        return 'evals_to_best', 'Evaluations to Best Found Fitness'
    if fitness_mode == 'final' and 'evals_to_final' in plot_df.columns:
        return 'evals_to_final', 'Evaluations to Final Found Fitness'
    if fitness_mode == 'best_noisy' and 'evals_to_best_noisy' in plot_df.columns:
        return 'evals_to_best_noisy', 'Evaluations to Best Found Noisy Fitness'
    if fitness_mode == 'final_noisy' and 'evals_to_final_noisy' in plot_df.columns:
        return 'evals_to_final_noisy', 'Evaluations to Final Found Noisy Fitness'
    return 'n_evals', 'Runtime (n_evals)'

def _format_scientific(val, decimals=1):
    """Format a number in scientific notation like '4.5E-2' (no zero-padded exponent)."""
    if pd.isna(val) or val == 0:
        return f'{0:.{decimals}f}E0'
    exponent = math.floor(math.log10(abs(val)))
    mantissa = val / (10 ** exponent)
    mantissa_str = f'{mantissa:.{decimals}f}'
    if abs(float(mantissa_str)) >= 10:
        exponent += 1
        mantissa_str = f'{val / (10 ** exponent):.{decimals}f}'
    return f'{mantissa_str}E{exponent}'


def _format_median_std(med, std, round_stats, use_scientific=False, med_decimals=3, std_decimals=3):
    """Format a median ± std cell.

    If use_scientific is True, both values are rendered in scientific notation
    (e.g. '4.5E-2'), overruling round_stats. Otherwise, both are rounded to
    1 decimal place if round_stats is True, else the given defaults.
    """
    if use_scientific:
        std_str = _format_scientific(std) if pd.notna(std) else 'N/A'
        return f'{_format_scientific(med)} ± {std_str}'
    if round_stats:
        med_decimals = std_decimals = 1
    std_str = f'{std:.{std_decimals}f}' if pd.notna(std) else 'N/A'
    return f'{med:.{med_decimals}f} ± {std_str}'


def _resolve_fit_column(fitness_mode, minimising):
    """Pick the fitness column and label matching the so-fitness-mode dropdown."""
    if fitness_mode == 'final':
        return 'final_fit', 'Final Fitness'
    if fitness_mode == 'final_noisy':
        return 'final_fit_noisy', 'Final Fitness (Noisy)'
    if fitness_mode == 'best_noisy':
        return ('min_fit_noisy' if minimising else 'max_fit_noisy'), 'Best Fitness (Noisy)'
    return ('min_fit' if minimising else 'max_fit'), 'Best Fitness'

@app.callback(
    Output('hide-series-dropdown', 'options'),
    Input('plot_2d_data', 'data'),
)
def update_hide_series_options(data):
    if not data:
        return []
    algo_names = pd.DataFrame(data)['algo_name'].dropna().unique()
    return [{'label': name, 'value': name} for name in sorted(algo_names)]

@app.callback(
    Output('advanced-misjudgement-algo-dropdown', 'options'),
    Output('advanced-misjudgement-algo-dropdown', 'value'),
    Input('plot_2d_data', 'data'),
)
def update_advanced_misjudgement_algo_options(data):
    if not data:
        return [], None
    algo_names = sorted(pd.DataFrame(data)['algo_name'].dropna().unique())
    options = [{'label': name, 'value': name} for name in algo_names]
    default_value = algo_names[0] if algo_names else None
    return options, default_value

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
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_line_so(data, fitness_mode, fit_func, opt_goal, plot_theme, noise_cap, hidden_series):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    problem_goal = _get_problem_goal(opt_goal)
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
    return plot2d_line(plot_df, fitness_mode=fitness_mode or 'best', problem_goal=problem_goal, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')
# 2D box plot
@app.callback(
    Output('2DBoxPlot', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('opt_goal', 'data'),
    Input('plot-theme', 'value'),
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_box_so(data, fitness_mode, fit_func, opt_goal, plot_theme, noise_cap, hidden_series):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    problem_goal = _get_problem_goal(opt_goal)
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
    return plot2d_box(plot_df, fitness_mode=fitness_mode or 'best', problem_goal=problem_goal, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# 2D line plot (multi-objective)
@app.callback(
    Output('2DLinePlotMO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('plot-theme', 'value'),
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_stored_data_mo_line(data, plot_theme, noise_cap, hidden_series):
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
    plot = plot2d_line_mo(plot_df, colorscale=plot_theme or 'Viridis')
    return plot

# 2D box plot (multi-objective)
@app.callback(
    Output('2DBoxPlotMO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('plot-theme', 'value'),
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_stored_data_mo_box(data, plot_theme, noise_cap, hidden_series):
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
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
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_line_evals_so(data, std_checkbox, fitness_mode, fit_func, plot_theme, noise_cap, hidden_series):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
    show_std = bool(std_checkbox and 'show' in std_checkbox)
    return plot2d_line_evals(plot_df, fitness_mode=fitness_mode or 'final', show_std=show_std, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# 2D box plot (evals, single-objective)
@app.callback(
    Output('2DBoxPlotEvalsSO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('plot-theme', 'value'),
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_box_evals_so(data, fitness_mode, fit_func, plot_theme, noise_cap, hidden_series):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
    return plot2d_box_evals(plot_df, fitness_mode=fitness_mode or 'final', xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# 2D box plot (penalty, single-objective)
@app.callback(
    Output('2DBoxPlotPenaltySO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('opt_goal', 'data'),
    Input('plot-theme', 'value'),
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_box_penalty_so(data, fitness_mode, fit_func, opt_goal, plot_theme, noise_cap, hidden_series):
    if not fit_func:
        return go.Figure()
    problem_goal = _get_problem_goal(opt_goal)
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
    return plot2d_box_penalty(plot_df, fitness_mode=fitness_mode or 'best', problem_goal=problem_goal, colorscale=plot_theme or 'Viridis')

# 2D box plot (misjudgements, single-objective)
@app.callback(
    Output('2DBoxPlotMisjudgementsSO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('fit_func_store', 'data'),
    Input('plot-theme', 'value'),
    Input('noise-cap-input', 'value'),
    Input('hide-series-dropdown', 'value'),
)
def display_box_misjudgements_so(data, fit_func, plot_theme, noise_cap, hidden_series):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None:
        return go.Figure()
    plot_df = _hide_series(_cap_noise(pd.DataFrame(data), noise_cap), hidden_series)
    return plot2d_box_misjudgements_so(plot_df, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# 2D box plot (advanced misjudgements, single-objective)
@app.callback(
    Output('2DBoxAdvancedMisjudgementsSO', 'figure'),
    Input('plot_2d_data', 'data'),
    Input('fit_func_store', 'data'),
    Input('plot-theme', 'value'),
    Input('noise-cap-input', 'value'),
    Input('advanced-misjudgement-algo-dropdown', 'value'),
)
def display_box_advanced_misjudgements_so(data, fit_func, plot_theme, noise_cap, selected_algo):
    xaxis_label = _get_so_xaxis_label(fit_func)
    if xaxis_label is None or not selected_algo:
        return go.Figure()
    plot_df = _cap_noise(pd.DataFrame(data), noise_cap)
    return plot2d_box_advanced_misjudgements_so(plot_df, algo_name=selected_algo, xaxis_title=xaxis_label, colorscale=plot_theme or 'Viridis')

# ---------- Misjudgements summary table ----------
@app.callback(
    Output('misjudgements-summary-table', 'children'),
    Input('plot_2d_data', 'data'),
    Input('fit_func_store', 'data'),
    Input('round-stats-checkbox', 'value'),
    Input('scientific-notation-checkbox', 'value'),
)
def update_misjudgements_summary_table(data, fit_func, round_stats_value, sci_value):
    round_stats = 'round' in (round_stats_value or [])
    use_scientific = 'sci' in (sci_value or [])
    if not fit_func:
        return html.P(
            "Select a problem from the table above to see the misjudgements summary.",
            style={'color': '#888', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    if not data:
        return html.P("No data available.", style={'color': '#888', 'fontStyle': 'italic'})

    plot_df = pd.DataFrame(data)

    if 'n_misjudgements' not in plot_df.columns:
        return html.P("No misjudgement data available.", style={'color': '#888', 'fontStyle': 'italic'})

    df_sub = plot_df[['algo_name', 'noise', 'n_misjudgements']].dropna(subset=['n_misjudgements'])
    if df_sub.empty:
        return html.P("No misjudgement data available.", style={'color': '#888', 'fontStyle': 'italic'})

    stats = df_sub.groupby(['noise', 'algo_name'])['n_misjudgements'].agg(['median', 'std']).reset_index()

    algos = sorted(stats['algo_name'].unique())
    noise_levels = sorted(stats['noise'].unique())

    rows = []
    for noise in noise_levels:
        row = {'Noise Level': noise}
        noise_stats = stats[stats['noise'] == noise]
        for algo in algos:
            algo_row = noise_stats[noise_stats['algo_name'] == algo]
            if algo_row.empty:
                row[algo] = '-'
            else:
                med = algo_row['median'].values[0]
                std = algo_row['std'].values[0]
                row[algo] = _format_median_std(med, std, round_stats, use_scientific)
        rows.append(row)

    columns = [{'name': 'Noise Level', 'id': 'Noise Level'}] + [{'name': a, 'id': a} for a in algos]

    table = dash_table.DataTable(
        data=rows,
        columns=columns,
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'padding': '6px 12px', 'fontFamily': 'monospace'},
        style_table={'marginBottom': '8px'},
    )

    return html.Div([
        html.H5("Misjudgements Summary: Median ± Std Dev by Noise Level",
                style={'marginTop': '16px', 'marginBottom': '4px'}),
        html.P("Median number of misjudgements per run across all runs, by algorithm and noise level.",
               style={'color': '#555', 'fontSize': '12px', 'marginBottom': '8px'}),
        table,
        html.Hr(),
    ])


# ---------- Performance summary table ----------
@app.callback(
    Output('performance-summary-table', 'children'),
    Input('plot_2d_data', 'data'),
    Input('so-fitness-mode', 'value'),
    Input('fit_func_store', 'data'),
    Input('opt_goal', 'data'),
    Input('round-stats-checkbox', 'value'),
    Input('scientific-notation-checkbox', 'value'),
)
def update_performance_summary_table(data, fitness_mode, fit_func, opt_goal, round_stats_value, sci_value):
    round_stats = 'round' in (round_stats_value or [])
    use_scientific = 'sci' in (sci_value or [])
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
    fit_col, label = _resolve_fit_column(fitness_mode, minimising)

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
                row[algo] = _format_median_std(med, std, round_stats, use_scientific)
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
    fit_col, label = _resolve_fit_column(fitness_mode, problem_goal == 'minimise')

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
    Input('round-stats-checkbox', 'value'),
    Input('scientific-notation-checkbox', 'value'),
)
def update_evals_summary_table(data, fit_func, fitness_mode, round_stats_value, sci_value):
    round_stats = 'round' in (round_stats_value or [])
    use_scientific = 'sci' in (sci_value or [])
    if not fit_func:
        return html.P(
            "Select a problem from the table above to see the evaluations summary.",
            style={'color': '#888', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    if not data:
        return html.P("No data available.", style={'color': '#888', 'fontStyle': 'italic'})

    plot_df = pd.DataFrame(data)

    eval_col, evals_label = _resolve_evals_column(fitness_mode, plot_df)

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
                row[algo] = _format_median_std(med, std, round_stats, use_scientific, med_decimals=1, std_decimals=3)
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

    eval_col, evals_label = _resolve_evals_column(fitness_mode, plot_df)

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
    elif tab == 'p10':
        return html.Div([
            dcc.Graph(id='2DBoxPlotPenaltySO', style={'width': '800px', 'height': '600px'}),
        ])
    elif tab == 'p8':
        return html.Div([
            dcc.Graph(id='2DBoxPlotMisjudgementsSO', style={'width': '800px', 'height': '600px'}),
        ])
    elif tab == 'p9':
        return html.Div([
            dcc.Graph(id='2DBoxAdvancedMisjudgementsSO', style={'width': '800px', 'height': '600px'}),
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
