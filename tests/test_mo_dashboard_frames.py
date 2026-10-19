"""Server-side multi-objective dashboard data (MO recording plan v6, Work Group 6).

`noisyvis.dashboard.mo_frames` is the only place the dashboard turns a result row's `mo_record` into plot
data, through MORunView, on the server (browser stores carry only a row id):

    legacy rows      skipped row by row: legacy MO payload present and no mo_record in that row (old and
                     new rows share one frame after a warehouse append, so both columns exist)
    row views        each row's record opened once, as an MORunView

The plot data built from those views is tested with its plots (test_mo_stn_* for the STN views,
test_pareto_dashboard_options for the Pareto tab).

It also documents where the removed recorder's per-entry lists cannot be reproduced from a record: tied
objective values (ParetoFront orders ties by insertion) and prior-noise observation identity (the
recorder merged equal (x, y) observations with different x~). Neither occurs in the frozen baselines.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.mo_legacy import legacy_columns
from harness.mo_runs import build, step_run
from noisyvis.dashboard.mo_frames import (
    LEGACY_MO_COLUMNS,
    row_view,
    without_legacy_mo_rows,
)
from noisyvis.results.mo_view import MORunView
from noisyvis.tracking.logger import clear_active_logger, get_active_logger


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


def view_of(name, prob):
    algo = build(name, prob=prob)
    step_run(algo)
    return algo, MORunView(algo.eval_log.finalise())


# ------------------------------------------------------------------------------ rows

def test_legacy_rows_are_skipped_row_by_row():
    """A frame mixing an SO row, a legacy MO row and a new MO row (as after appending new results to an
    old warehouse): only the legacy MO row goes; the legacy columns go; the index is the row id."""
    _, view = view_of("MoUMDA", "kp1")
    record = view_of("SEMO", "kp1")[0].eval_log.finalise()
    frame = pd.DataFrame([
        {"algo_type": "SO", "rep_sols": [[1, 0]], "seed": 1},
        {"algo_type": "MO", "pareto_solutions": [[[1, 0]]], "final_true_hv": 3.0, "seed": 2},
        {"algo_type": "MO", "mo_record": record, "seed": 3},
    ])
    assert {"pareto_solutions", "mo_record"} <= set(frame.columns)   # both columns exist globally
    kept = without_legacy_mo_rows(frame)
    assert list(kept["seed"]) == [1, 3] and list(kept.index) == [0, 1]
    assert not set(LEGACY_MO_COLUMNS) & set(kept.columns)
    assert row_view(kept, 0) is None and isinstance(row_view(kept, 1), MORunView)
    # without an mo_record column the legacy row still goes; a row with neither payload stays
    assert list(without_legacy_mo_rows(frame.drop(columns=["mo_record"]))["seed"]) == [1, 3]
    assert list(without_legacy_mo_rows(frame.drop(columns=["pareto_solutions", "final_true_hv"]))["seed"]) == [1, 2, 3]


# ------------------------------------------------------------------------------ documented limits

def test_tied_values_and_prior_noise_identity_are_the_documented_limits():
    """Where the removed recorder's lists cannot be reproduced from a record, and why:
    counting ones / counting zeros has integer objectives, so distinct genotypes tie and ParetoFront's
    insertion order among them is not recorded; under prior noise, observations with the same x and y
    but different x~ are distinct noisy-front members, which the recorder merged. Posterior knapsack
    runs (the frozen baselines' setting) have neither."""
    _, cocz = view_of("MoUMDA", "cocz1")
    assert legacy_columns(cocz)["tied_members"]
    _, prior = view_of("MoUMDA", "prior")
    distinct_xy = []
    for g in range(prior.n_generations):
        front = prior.current_noisy_front(g)
        xy = {(tuple(int(v) for v in x), tuple(y)) for x, y in zip(front.x, front.observed)}
        distinct_xy.append(len(front) - len(xy))
    assert legacy_columns(prior)["tied_members"] or any(distinct_xy)
    for name in ("MoUMDA", "NSGA2"):
        assert not legacy_columns(view_of(name, "kp1")[1])["tied_members"]
