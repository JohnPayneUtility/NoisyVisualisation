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

The per-experiment switch `mlflow.enabled` (default true) is pinned here too: disabled, an entry point
starts no MLflow run and writes no file into either store, while its warehouse, temp and Hydra outputs
are exactly those of an enabled run, apart from the MLflow run ids and the per-run resource timings.
Enabled runs all end FINISHED with the SO parent/child structure intact, switching within one
process leaves no run active, and a non-boolean value is rejected before anything is written.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.run_isolated import HarnessError, child_env, run_isolated

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


# ------------------------------------------------------------------ the mlflow.enabled switch

DISABLED = "++mlflow.enabled=false"
FINISHED = 3  # mlflow.entities.RunStatus.FINISHED, as written to meta.yaml

# script: (warehouse pickle, temp pickle) the run writes, whether or not MLflow records it
OUTPUTS = {
    "run.py": ("algo_results.pkl", "results.pkl"),
    "run_mo.py": ("algo_results.pkl", "results.pkl"),
    "run_lon.py": ("lon_results.pkl", "lon_results.pkl"),
    "run_lon_parallel.py": ("lon_results.pkl", "lon_results.pkl"),
    "run_colon_parallel.py": ("lon_results.pkl", "lon_results.pkl"),
}

# Compares two result pickles cell by cell (pickled bytes, so lists, arrays and NaN compare exactly),
# ignoring the MLflow run ids and the wall-clock/resource measurements. Runs in a subprocess: the
# pytest process never imports pandas or the experiment packages.
_COMPARE = r"""
import json, pickle, sys
import pandas as pd
VOLATILE = {"run_id", "parent_run_id", "run_wall_time_s", "run_cpu_time_s", "peak_ram_mb", "ram_baseline_mb"}
a, b = (pd.read_pickle(p) for p in sys.argv[1:3])
report = {"rows": [len(a), len(b)], "columns_equal": list(a.columns) == list(b.columns),
          "run_ids": [sorted({str(v) for v in df[c]}) for df in (a, b) for c in ("run_id", "parent_run_id") if c in df]}
cols = [c for c in a.columns if c not in VOLATILE]
report["diff"] = [] if len(a) != len(b) else [
    [i, c] for c in cols for i in range(len(a))
    if pickle.dumps(a[c].iloc[i], protocol=4) != pickle.dumps(b[c].iloc[i], protocol=4)
][:20]
report["compared_columns"] = len(cols)
print(json.dumps(report))
"""

_ROW_COUNT = "import sys, pandas as pd; print(len(pd.read_pickle(sys.argv[1])))"


def _files(store: Path) -> list:
    return sorted(str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()) if store.is_dir() else []


def _runs(store: Path) -> dict:
    """run id -> (status, parent run id or None), read from the file store's meta.yaml and tags."""
    runs = {}
    for meta in store.glob("*/*/meta.yaml") if store.is_dir() else []:
        if meta.parent.parent.name.startswith("."):
            continue
        status = next(int(line.split(":", 1)[1]) for line in meta.read_text().splitlines() if line.startswith("status:"))
        parent_tag = meta.parent / "tags" / "mlflow.parentRunId"
        runs[meta.parent.name] = (status, parent_tag.read_text().strip() if parent_tag.is_file() else None)
    return runs


def _python(root: Path, code: str, *args) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        cwd=str(root), env=child_env(root), capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return completed.stdout.strip().splitlines()[-1]


def _isolated_root(script: str, overrides: list) -> Path:
    return Path(run_isolated(script, ENTRYPOINTS[script][0], "none", overrides=overrides, keep_root=True).root)


@pytest.mark.parametrize("script", list(ENTRYPOINTS))
def test_disabled_logs_nothing_and_still_writes_the_warehouse(script):
    root = _isolated_root(
        script, ["run.num_runs=1", "run.parallel=false", f"mlflow.tracking_uri={CONFIG_STORE}", DISABLED]
    )
    try:
        for store in (MODULE_STORE, CONFIG_STORE):
            assert not _files(root / store), f"{script}: MLflow wrote {_files(root / store)} into {store}"
        warehouse, temp = OUTPUTS[script]
        assert (root / "data" / "warehouse" / warehouse).is_file(), f"{script}: no warehouse {warehouse}"
        assert (root / "data" / "temp" / temp).is_file(), f"{script}: no temp {temp}"
        if script == "run.py":
            ids = json.loads(_python(root, _COMPARE, *[root / "data" / "warehouse" / warehouse] * 2))["run_ids"]
            assert ids[:2] == [["None"], ["None"]], f"run_id/parent_run_id of a disabled SO run: {ids[:2]}"
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("script", ["run.py", "run_mo.py", "run_lon.py"])
def test_enabled_and_disabled_runs_write_identical_results(script):
    """Same seeds, MLflow on vs off: identical warehouse and temp rows; the enabled runs all FINISHED."""
    overrides = ["run.num_runs=2", "run.parallel=false"]
    on = _isolated_root(script, overrides)
    off = _isolated_root(script, overrides + [DISABLED])
    try:
        warehouse, temp = OUTPUTS[script]
        for rel in (f"data/warehouse/{warehouse}", f"data/temp/{temp}"):
            report = json.loads(_python(on, _COMPARE, on / rel, off / rel))
            assert report["rows"][0] == report["rows"][1] > 0, f"{script} {rel}: rows {report['rows']}"
            assert report["columns_equal"], f"{script} {rel}: column lists differ"
            assert not report["diff"], f"{script} {rel}: [row, column] cells differ: {report['diff']}"
        on_cfg, off_cfg = ((root / "data/outputs/.hydra/config.yaml").read_text().splitlines() for root in (on, off))
        assert on_cfg == [line for line in off_cfg if line.strip() != "enabled: false"], "Hydra configs differ"

        # No tracking_uri override here, so every workflow logs to data/mlruns.
        runs = _runs(on / MODULE_STORE)
        assert runs and all(status == FINISHED for status, _ in runs.values()), f"{script}: run states {runs}"
        if script == "run.py":
            parents = [rid for rid, (_, parent) in runs.items() if parent is None]
            children = [parent for _, parent in runs.values() if parent is not None]
            assert len(parents) == 1 and children == parents * 2, f"SO parent/child structure: {runs}"
        assert not _files(off / MODULE_STORE) and not _files(off / CONFIG_STORE)
    finally:
        shutil.rmtree(on, ignore_errors=True)
        shutil.rmtree(off, ignore_errors=True)


@pytest.mark.parametrize("order", ["true,false", "false,true"])
def test_switching_within_one_process_leaves_no_run_active(order):
    """A two-job multirun in one process: an MLflow run left active would make the next job's start_run fail."""
    root = _isolated_root(
        "run.py", ["hydra.mode=MULTIRUN", "run.num_runs=1", "run.parallel=false", f"++mlflow.enabled={order}"]
    )
    try:
        runs = _runs(root / MODULE_STORE)
        assert sorted(parent is None for _, parent in runs.values()) == [False, True], f"runs: {runs}"
        assert all(status == FINISHED for status, _ in runs.values()), f"run states: {runs}"
        assert _python(root, _ROW_COUNT, root / "data/warehouse/algo_results.pkl") == "2"
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("value", ["maybe", "1"])
def test_non_boolean_enabled_is_rejected(value):
    with pytest.raises(HarnessError) as failure:
        run_isolated("run.py", "so_onemax", "none", overrides=["run.num_runs=1", f"++mlflow.enabled={value}"])
    message = str(failure.value)
    root = Path(re.search(r"temp root kept at (\S+)", message).group(1))
    try:
        assert "mlflow.enabled must be true or false" in message, message[-2000:]
        assert not (root / "data" / "warehouse").exists() or not any((root / "data" / "warehouse").iterdir())
        assert not _files(root / MODULE_STORE)
    finally:
        shutil.rmtree(root, ignore_errors=True)
