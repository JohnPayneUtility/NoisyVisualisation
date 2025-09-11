"""
mlflow_loader.py — basic MLflow utilities

This version is simplified to just list experiments.
"""
from __future__ import annotations

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# ------------------------------
# Experiments
# ------------------------------

def list_experiments(as_df: bool = True):
    """Return all MLflow experiments.

    Args:
        as_df: if True, return a pandas DataFrame, else a list of Experiment objects.
    """
    client = MlflowClient()
    experiments = client.search_experiments()
    if as_df:
        data = [{
            "experiment_id": e.experiment_id,
            "name": e.name,
            "artifact_location": e.artifact_location,
            "lifecycle_stage": e.lifecycle_stage,
            "creation_time": e.creation_time,
            "last_update_time": e.last_update_time,
        } for e in experiments]
        return pd.DataFrame(data)
    return experiments


if __name__ == "__main__":
    df = list_experiments()
    if df.empty:
        print("No experiments found.")
    else:
        print(df.to_string(index=False))

# ------------------------------
# Experiments listing (simple + Dash)
# ------------------------------

def list_experiments_df(include_deleted: bool = False) -> pd.DataFrame:
    """Return all MLflow experiments as a tidy DataFrame.

    Columns: experiment_id, name, artifact_location, lifecycle_stage,
             creation_time, last_update_time, tags (dict)
    """
    client = MlflowClient()
    from mlflow.entities import ViewType
    view = ViewType.ALL if include_deleted else ViewType.ACTIVE_ONLY
    exps = client.search_experiments(view_type=view)
    rows = []
    for e in exps:
        rows.append({
            "experiment_id": e.experiment_id,
            "name": e.name,
            "artifact_location": e.artifact_location,
            "lifecycle_stage": getattr(e, "lifecycle_stage", "active"),
            "creation_time": getattr(e, "creation_time", None),
            "last_update_time": getattr(e, "last_update_time", None),
            "tags": getattr(e, "tags", {}),
        })
    df = pd.DataFrame(rows)
    # Optional: nicer times
    for c in ("creation_time", "last_update_time"):
        if c in df.columns and pd.notnull(df[c]).any():
            try:
                df[c] = pd.to_datetime(df[c], unit="ms")
            except Exception:
                pass
    return df.sort_values("name").reset_index(drop=True)

