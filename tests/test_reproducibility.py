"""The five reproducibility baselines (plan §5.5).

    Given the same config and seed, does the code still produce the same scientific behaviour
    after restructuring?

Each baseline runs the real entry point in an isolated temp root and compares the recorded
scientific outputs **exactly** -- no tolerances. A difference means the RNG call order or the
behaviour changed (the core invariant, §7.1), and must be investigated, never absorbed.

    SO-seq  run.py, parallel false: a loader-less OneMax and a tiny knapsack
    SO-par  the same configs, parallel true; compared keyed by seed, not completion order
    MO      run_mo.py: Pareto sets and hypervolume series
    LON     run_lon_parallel.py, forced sequential: aggregated optima/edge maps per compression
    CoLON   run_colon_parallel.py, forced sequential, with violation_fn: plus feasibility

LON and CoLON are sequential deliberately and permanently (risk R27): parallel LON output depends
on worker completion order, so it could not pass twice in a row. Do not try to make it determinstic.

Recording:
    NOISYVIS_RECORD_BASELINES=1          write missing baselines
    NOISYVIS_RECORD_BASELINES=overwrite  replace existing ones
A missing baseline in normal mode is a failure, never a skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from harness import canonical
from harness.run_isolated import run_isolated

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
RECORD_ENV = "NOISYVIS_RECORD_BASELINES"

BASELINES = {
    "so_seq": {
        "script": "run.py",
        "kind": "so",
        "configs": ["so_onemax", "so_knapsack"],
        "overrides": ["run.parallel=false"],
        "expect_parallel": False,
    },
    "so_par": {
        "script": "run.py",
        "kind": "so",
        "configs": ["so_onemax", "so_knapsack"],
        "overrides": ["run.parallel=true"],
        "expect_parallel": True,
    },
    "mo": {
        "script": "run_mo.py",
        "kind": "mo",
        "configs": ["mo_knapsack"],
        "overrides": ["run.parallel=false"],
        "expect_parallel": False,
    },
    "lon": {
        "script": "run_lon_parallel.py",
        "kind": "lon",
        "configs": ["lon_knapsack_noisy"],
        "overrides": ["run.parallel=false"],
        "expect_parallel": False,
    },
    "colon": {
        "script": "run_colon_parallel.py",
        "kind": "colon",
        "configs": ["colon_knapsack_noisy"],
        "overrides": ["run.parallel=false"],
        "expect_parallel": False,
    },
}


def record_mode() -> str | None:
    value = os.environ.get(RECORD_ENV, "").strip().lower()
    if value in ("", "0", "false", "no"):
        return None
    return "overwrite" if value == "overwrite" else "record"


@pytest.fixture(scope="session")
def run_baseline():
    """Execute a baseline once per session and cache its result."""
    cache: dict = {}

    def _run(name: str) -> dict:
        if name in cache:
            return cache[name]

        spec = BASELINES[name]
        runs = {}
        for config_name in spec["configs"]:
            result = run_isolated(
                spec["script"],
                config_name,
                spec["kind"],
                overrides=spec["overrides"],
            )
            runs[config_name] = result.extracted

        cache[name] = runs
        return runs

    return _run


def _check_isolation_and_mode(name: str, config_name: str, extracted: dict) -> None:
    """Every run must prove where it wrote and which mode it used."""
    spec = BASELINES[name]
    meta = extracted["meta"]

    assert meta["run_parallel"] is spec["expect_parallel"], (
        f"{name}/{config_name}: ran with run.parallel={meta['run_parallel']}, expected "
        f"{spec['expect_parallel']}. LON and CoLON baselines must be sequential (R27)."
    )

    if spec["kind"] in ("so", "mo"):
        expected_rows = meta["run_num_runs"]
    else:
        expected_rows = len(extracted["cases"])
    assert meta["warehouse_rows"] == expected_rows, (
        f"{name}/{config_name}: the isolated warehouse holds {meta['warehouse_rows']} rows, "
        f"expected {expected_rows}. The run may not have written where it was supposed to."
    )

    assert meta["mlruns_files"] > 0, (
        f"{name}/{config_name}: no MLflow files in the temp root; tracking was not redirected."
    )


def _observed(runs: dict) -> dict:
    return {config: canonical.jsonable(extracted["cases"]) for config, extracted in runs.items()}


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline(name, run_baseline, note):
    spec = BASELINES[name]
    runs = run_baseline(name)

    for config_name, extracted in runs.items():
        _check_isolation_and_mode(name, config_name, extracted)

    observed = _observed(runs)
    baseline_path = BASELINE_DIR / f"{name}.json"
    mode = record_mode()

    if mode and (mode == "overwrite" or not baseline_path.is_file()):
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            canonical.dumps(
                {
                    "schema": canonical.SCHEMA,
                    "baseline": name,
                    "script": spec["script"],
                    "kind": spec["kind"],
                    "overrides": spec["overrides"],
                    "configs": spec["configs"],
                    "cases": observed,
                }
            )
        )
        note(f"recorded baseline {name} ({baseline_path.name})")
        return

    if not baseline_path.is_file():
        pytest.fail(
            f"no baseline recorded for {name!r}. Record it on an unmodified tree with "
            f"{RECORD_ENV}=1; a missing baseline is never skipped, because it would silently "
            f"disable the migration gate."
        )

    if mode == "record":
        note(f"baseline {name} already exists; not overwritten (use {RECORD_ENV}=overwrite)")

    stored = json.loads(baseline_path.read_text())
    assert stored["script"] == spec["script"] and stored["overrides"] == spec["overrides"], (
        f"{name}: baseline was recorded with a different invocation "
        f"({stored['script']} {stored['overrides']}); re-record it deliberately."
    )

    difference = canonical.first_difference(stored["cases"], observed)
    assert difference is None, (
        f"{name}: scientific output changed against the recorded baseline.\n  {difference}\n"
        f"This means behaviour or RNG call order changed (plan §7.1). Investigate before "
        f"continuing; do not re-record to make it pass."
    )


def test_run_lon_matches_lon_baseline():
    """`run_lon.py` reproduces the LON baseline recorded from `run_lon_parallel.py`.

    `run_lon.py` builds the LON inline -- no worker function, its own merge loop -- and nothing else
    covers it behaviourally. Its merge, seeding and BinaryLON arguments match the sequential path of
    `run_lon_parallel.py`, so its aggregated maps must equal the LON baseline exactly. This compares
    against `lon.json` and never records.
    """
    spec = BASELINES["lon"]
    runs = {}
    for config_name in spec["configs"]:
        result = run_isolated("run_lon.py", config_name, spec["kind"], overrides=spec["overrides"])
        _check_isolation_and_mode("lon", config_name, result.extracted)
        runs[config_name] = result.extracted

    stored = json.loads((BASELINE_DIR / "lon.json").read_text())
    difference = canonical.first_difference(stored["cases"], _observed(runs))
    assert difference is None, (
        f"run_lon.py: output differs from the LON baseline.\n  {difference}\n"
        f"Investigate; never re-record lon.json to make this pass."
    )


def test_so_parallel_matches_sequential(run_baseline, note):
    """Observation, not a gate.

    §5.5 compares SO-par against its own baseline, because parallel and sequential runs need not
    agree. Whether they do is still worth knowing, so it is reported rather than asserted.
    """
    sequential = _observed(run_baseline("so_seq"))
    parallel = _observed(run_baseline("so_par"))

    difference = canonical.first_difference(sequential, parallel)
    if difference is None:
        note("SO parallel output is identical to sequential, per seed")
    else:
        note(f"SO parallel differs from sequential (expected to be possible): {difference}")
