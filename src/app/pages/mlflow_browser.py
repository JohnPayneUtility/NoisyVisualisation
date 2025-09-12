# pages/experiments.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from dash import html, dash_table, dcc, Input, Output, State, callback
import dash
import mlflow

from src.io.mlflow_loader import list_experiments_df
from src.io.mlflow_loader import list_runs_df, select_present_columns  # <-- new imports

dash.register_page(__name__, path="/experiments", name="Experiments")

layout = html.Div([
    html.Div([
        html.H3("MLflow Experiments", style={"marginBottom": 0}),
        html.Div(id="tracking-uri", style={"fontFamily": "monospace", "fontSize": 12, "opacity": 0.7}),
    ], style={"margin": "12px"}),

    # Experiments table
    html.Div(id="experiments-table-wrap", style={"margin": "12px"}),

    # Runs table header + table
    html.Hr(),
    html.Div([
        html.H4("Runs in selected experiment", style={"margin": "12px 12px 0 12px"}),
        html.Div(id="runs-table-wrap", style={"margin": "12px"}),
    ]),

    # Poll to refresh occasionally
    # dcc.Interval(id="experiments-poll", interval=30_000, n_intervals=0),
])

# Columns for experiments table
EXP_COLS = [
    "experiment_id",
    "name",
    "artifact_location",
    "lifecycle_stage",
    "creation_time",
    "last_update_time",
]

# A compact default set of run columns (only those present will be shown)
RUN_COLS_DEFAULT = [
    # "run_id",
    "run_name",
    # "experiment_id",
    # "status",
    "start_time",
    "end_time",
    "duration",
    # "artifact_uri",
    # Common params/metrics you might have — shown only if present:
    "params.algo", "params.problem", "params.noise", "params.seed", "params.pop_size",
    "metrics.best", "metrics.hv", "metrics.mean",
]


@callback(
    Output("tracking-uri", "children"),
    Output("experiments-table-wrap", "children"),
    Input("tracking-uri", "id"),   # fires once when page loads
)
def render_experiments(_):
    # Ensure your file-store tracking URI (adjust depth if your path is different)
    repo_root = Path(__file__).resolve().parents[3]
    mlruns_dir = repo_root / "data" / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    tracking_uri = mlflow.get_tracking_uri()

    df = list_experiments_df()
    if df.empty:
        return (
            f"Tracking URI: {tracking_uri}",
            html.Div("No experiments found.", style={"opacity": 0.7}),
        )

    df_view = df[[c for c in EXP_COLS if c in df.columns]].copy()
    for c in ("creation_time", "last_update_time"):
        if c in df_view.columns and pd.api.types.is_datetime64_any_dtype(df_view[c]):
            df_view[c] = df_view[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    table = dash_table.DataTable(
        id="experiments-table",
        columns=[{"name": c, "id": c} for c in df_view.columns],
        data=df_view.to_dict("records"),
        row_selectable="single",
        selected_rows=[],                 # no selection initially
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "fontFamily": "monospace", "fontSize": 13},
        style_header={"fontWeight": "bold"},
    )

    return (f"Tracking URI: {tracking_uri}", table)


@callback(
    Output("runs-table-wrap", "children"),
    Input("experiments-table", "data"),
    Input("experiments-table", "selected_rows"),
)
def render_runs_table(exp_data, selected_rows):
    if not exp_data:
        return html.Div("No experiments loaded.", style={"opacity": 0.7})

    # If nothing selected, show hint
    if not selected_rows:
        return html.Div("Select an experiment above to see its runs.", style={"opacity": 0.7})

    # Map the selected row index to an experiment_id
    try:
        idx = selected_rows[0]
        exp_row = exp_data[idx]
        experiment_id = exp_row["experiment_id"]
    except Exception:
        return html.Div("Invalid selection.", style={"opacity": 0.7})

    runs = list_runs_df(experiment_id, order_by=["attributes.start_time DESC"])
    if runs.empty:
        return html.Div("No runs found for this experiment.", style={"opacity": 0.7})

    # Choose a compact set of columns that actually exist
    cols = select_present_columns(runs, RUN_COLS_DEFAULT)
    df_view = runs[cols].copy()

    # Pretty time columns if present
    for c in ("start_time", "end_time"):
        if c in df_view.columns and pd.api.types.is_datetime64_any_dtype(df_view[c]):
            df_view[c] = df_view[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df_view.columns],
        data=df_view.to_dict("records"),
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "fontFamily": "monospace", "fontSize": 13},
        style_header={"fontWeight": "bold"},
    )
