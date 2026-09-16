"""Each entry point logs MLflow runs to the tracking store it is meant to use (risk R24, Stage 7).

The harness's unreachable sentinel URI proves an entry point sets *some* tracking URI. It cannot
tell *which*: in a temp root, the LON scripts' import-time `file:<root>/data/mlruns` and their
config's `tracking_uri: "data/mlruns"` (relative to the cwd, which is the same root) are one
directory. Losing the in-`main` `mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)` would go
unnoticed.

So every entry point runs once with the config's tracking URI pointed somewhere distinct, and the
test checks where the runs actually landed:

    SO, MO           -> data/mlruns (results.paths.MLRUNS_DIR); the config value is ignored
    LON, LON-par,    -> the config's tracking_uri; nothing under data/mlruns
    CoLON

This pins the existing per-workflow semantics, so extracting the tracking code cannot make one
workflow inherit another's URI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness.run_isolated import run_isolated

CONFIG_STORE = "data/mlruns_from_config"
MODULE_STORE = "data/mlruns"

# script: (test config, store the runs must land in, runs expected there)
ENTRYPOINTS = {
    "run.py": ("so_onemax", MODULE_STORE, 2),  # parent sweep run + one child per seed
    "run_mo.py": ("mo_knapsack", MODULE_STORE, 1),
    "run_lon.py": ("lon_knapsack_noisy", CONFIG_STORE, 1),
    "run_lon_parallel.py": ("lon_knapsack_noisy", CONFIG_STORE, 1),
    "run_colon_parallel.py": ("colon_knapsack_noisy", CONFIG_STORE, 1),
}


def _run_ids(store: Path) -> list:
    """Run directories in an MLflow file store: <store>/<experiment id>/<run id>/meta.yaml."""
    if not store.is_dir():
        return []
    return sorted(
        path.parent.name
        for path in store.glob("*/*/meta.yaml")
        if not path.parent.parent.name.startswith(".")
    )


@pytest.mark.parametrize("script", list(ENTRYPOINTS))
def test_runs_land_in_the_intended_tracking_store(script):
    config_name, expected_store, expected_runs = ENTRYPOINTS[script]
    other_store = CONFIG_STORE if expected_store == MODULE_STORE else MODULE_STORE

    result = run_isolated(
        script,
        config_name,
        "none",
        overrides=["run.num_runs=1", "run.parallel=false", f"mlflow.tracking_uri={CONFIG_STORE}"],
        keep_root=True,
    )
    root = Path(result.root)
    try:
        landed = _run_ids(root / expected_store)
        stray = _run_ids(root / other_store)
        assert len(landed) == expected_runs, (
            f"{script}: expected {expected_runs} MLflow run(s) in {expected_store}, found {len(landed)}"
        )
        assert not stray, (
            f"{script}: MLflow runs were logged to {other_store}, which this workflow must not use: {stray}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
