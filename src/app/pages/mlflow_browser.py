# pages/experiments.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from dash import html, dash_table, dcc, Input, Output, callback
import mlflow
from src.io.mlflow_loader import list_experiments_df

import dash
dash.register_page(__name__, path="/experiments", name="Experiments")

layout = html.Div([
    html.Div([
        html.H3("MLflow Experiments", style={"marginBottom": 0}),
        html.Div(id="tracking-uri", style={"fontFamily": "monospace", "fontSize": 12, "opacity": 0.7}),
    ], style={"margin": "12px"}),

    html.Div(id="experiments-table-wrap", style={"margin": "12px"}),

    dcc.Interval(id="experiments-poll", interval=30_000, n_intervals=0),
])

def _columns_present(df: pd.DataFrame):
    cols = [
        "experiment_id",
        "name",
        "artifact_location",
        "lifecycle_stage",
        "creation_time",
        "last_update_time",
    ]
    return [c for c in cols if c in df.columns]

cols = [
        "experiment_id",
        "name",
        "artifact_location",
        "lifecycle_stage",
        "creation_time",
        "last_update_time",
    ]

@callback(
    Output("tracking-uri", "children"),
    Output("experiments-table-wrap", "children"),
    Input("experiments-poll", "n_intervals"),
)
def _render_experiments(_):
    repo_root = Path(__file__).resolve().parents[3]
    mlruns_dir = repo_root / "data" / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    tracking_uri = mlflow.get_tracking_uri()

    df = list_experiments_df()
    if df.empty:
        return (
            f"Tracking URI: {tracking_uri}",
            html.Div("No experiments found. Check MLFLOW_TRACKING_URI and permissions.", style={"opacity": 0.7}),
        )

    df_view = df[cols].copy()

    # pretty timestamps
    for c in ("creation_time", "last_update_time"):
        if c in df_view.columns and pd.api.types.is_datetime64_any_dtype(df_view[c]):
            df_view[c] = df_view[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in cols],
        data=df_view.to_dict("records"),
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "fontFamily": "monospace", "fontSize": 13},
        style_header={"fontWeight": "bold"},
    )

    return (f"Tracking URI: {tracking_uri}", table)
