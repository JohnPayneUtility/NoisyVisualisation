"""Data-selection callbacks: the experiment filter, the problem (Table 1) and algorithm (Table 2)
selection tables, their stores, and the problem stores (optimum, PID, goal, fitness function)."""

import pandas as pd
from dash import html, Input, Output, State, ctx

from ..instance import app
from ..data import df, experiment_descriptions
from ..tables import create_display2_df
from ..columns import DISPLAY1_COLUMNS
from ..layout.stores import LON_TABLE_SELECTED_PID_STORE
from ...problems.instances import get_knapsack_problem_stats, interpret_correlation


def _filter_by_experiment(data_df, selected):
    """Filter a dataframe by the experiment-selector value (single string, list, or None).
    '__null__' matches rows where experiment_name is NaN/None."""
    if not selected or 'experiment_name' not in data_df.columns:
        return data_df
    if isinstance(selected, str):
        selected = [selected]
    include_null = '__null__' in selected
    named = [n for n in selected if n != '__null__']
    if include_null and named:
        return data_df[data_df['experiment_name'].isna() | data_df['experiment_name'].isin(named)]
    if include_null:
        return data_df[data_df['experiment_name'].isna()]
    return data_df[data_df['experiment_name'].isin(named)]


@app.callback(
    Output("experiment-description-display", "children"),
    Input("experiment-selector", "value"),
)
def update_experiment_description(selected):
    if not selected:
        return ""
    names = [selected] if isinstance(selected, str) else selected
    items = []
    for name in names:
        if name == '__null__':
            continue
        desc = experiment_descriptions.get(name, '')
        if desc:
            items.append(html.Div([
                html.Span(name, style={'fontWeight': 'bold'}),
                html.Span(f": {desc}", style={'marginLeft': '4px'}),
            ], style={'marginBottom': '4px'}))
    return items or ""


@app.callback(
    Output('knapsack-info-display', 'children'),
    Input('PID', 'data'),
)
def update_knapsack_info(pid):
    if not pid:
        return ""
    stats = get_knapsack_problem_stats(pid)
    if stats is None:
        return html.Div("Not a knapsack problem — no additional information.", style={'fontStyle': 'italic'})

    def row(label, value, suffix=""):
        if isinstance(value, float):
            formatted = f"{value:,.0f}" if value.is_integer() else f"{value:.3g}"
        else:
            formatted = str(value)
        return html.Div([
            html.Span(f"{label}: ", style={'fontWeight': 'bold'}),
            html.Span(formatted + suffix),
        ])

    correlation = stats['value_weight_correlation']
    ratio_text = (
        f"{stats['max_ratio_value']:.3g} / {stats['max_ratio_weight']:.3g} "
        f"({stats['max_ratio']:.3g})"
    )

    summary_col = [
        row("Items", stats['n_items']),
        row("Capacity", stats['capacity']),
        row("Global optimum", stats['global_optimum']),
        row("Sum of values", stats['sum_values']),
        row("Sum of weights", stats['sum_weights']),
        row("Value/weight correlation", correlation, suffix=f" ({interpret_correlation(correlation)})"),
        row("Largest value/weight", ratio_text),
    ]
    values_col = [
        row("Max value", stats['max_value']),
        row("Min value", stats['min_value']),
        row("Avg value", stats['avg_value']),
        row("Std value", stats['std_value']),
    ]
    weights_col = [
        row("Max weight", stats['max_weight']),
        row("Min weight", stats['min_weight']),
        row("Avg weight", stats['avg_weight']),
        row("Std weight", stats['std_weight']),
    ]
    return html.Div([
        html.Div(summary_col, style={'marginBottom': '6px'}),
        html.Div([
            html.Div(values_col, style={'marginRight': '30px'}),
            html.Div(weights_col),
        ], style={'display': 'flex'}),
    ])


@app.callback(
    Output("table1", "data"),
    Input("experiment-selector", "value"),
)
def update_table1_for_experiment(experiment_name):
    filtered = _filter_by_experiment(df, experiment_name)
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
    exp_df = _filter_by_experiment(df, experiment_name)
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
def update_problem_stores(data, lon_table_pid, table1_selection):
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
