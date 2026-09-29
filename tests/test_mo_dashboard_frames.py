"""Server-side multi-objective dashboard data (MO recording plan v6, Work Group 6).

`noisyvis.dashboard.mo_frames` is the only place the dashboard turns a result row's `mo_record` into plot
data, through MORunView, on the server (browser stores carry only a row id):

    legacy rows      skipped row by row: legacy MO payload present and no mo_record in that row (old and
                     new rows share one frame after a warehouse append, so both columns exist)
    plot entries     one per change of the current noisy front's genotype set, as the plots have always
                     consumed them: the legacy recorder's entries exactly (the frozen reference-#3
                     projection), now with the real generation (gen_idx) and the change position
                     (change_idx) as separate fields
    STN modes        the per-mode front/metric pairs, unchanged
    archives         the passive archives and lost clean solutions, available server-side

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
from harness.mo_runs import ALGORITHM_NAMES, build, step_run
from noisyvis.dashboard.mo_frames import (
    LEGACY_MO_COLUMNS,
    archive_snapshot,
    change_generations,
    front_entries,
    row_view,
    stn_entries,
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


# ------------------------------------------------------------------------------ plot entries

@pytest.mark.parametrize("prob", ["kp0", "kp1", "kpviol1"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_entries_are_the_legacy_entries_with_real_generations(name, prob):
    """Posterior noise: every entry equals the removed recorder's entry (the projection the frozen
    baselines are compared through, verified against the live recorder before its removal), members in
    order; gen_idx is the generation of the change and change_idx its position."""
    _, view = view_of(name, prob)
    legacy = legacy_columns(view)
    entries = front_entries(view)
    generations = change_generations(view)
    assert len(entries) == len(legacy["pareto_solutions"]) == len(generations)
    assert [e["gen_idx"] for e in entries] == generations and [e["change_idx"] for e in entries] == list(range(len(entries)))
    for k, e in enumerate(entries):
        assert e["algo_front_solutions"] == legacy["pareto_solutions"][k]
        assert e["algo_front_noisy_fitnesses"] == legacy["pareto_fitnesses"][k]
        assert e["algo_front_clean_fitnesses"] == legacy["pareto_true_fitnesses"][k]
        assert e["clean_front_solutions"] == legacy["true_pareto_solutions"][k]
        assert e["clean_front_fitnesses"] == legacy["true_pareto_fitnesses"][k]
        assert (e["algo_front_noisy_hypervolume"], e["algo_front_clean_hypervolume"], e["clean_front_hypervolume"]) == \
            (legacy["noisy_pf_noisy_hypervolumes"][k], legacy["noisy_pf_true_hypervolumes"][k],
             legacy["true_pf_hypervolumes"][k])


def test_stn_modes_pair_the_same_fronts_and_metrics():
    entry = {"algo_front_solutions": "noisy", "clean_front_solutions": "clean", "algo_front_noisy_hypervolume": 1.0,
             "algo_front_clean_hypervolume": 2.0, "clean_front_hypervolume": 3.0, "gen_idx": 7, "change_idx": 2}
    expected = {
        "npnhv": ("noisy", None, 1.0, None), "npthv": ("noisy", None, 2.0, None), "tpthv": ("clean", None, 3.0, None),
        "npbhv": ("noisy", None, 2.0, 1.0), "bpbhv": ("clean", "noisy", 3.0, 1.0),
        "bpbhv_algo_pov": ("noisy", "clean", 1.0, 3.0), "tpbhv": ("noisy", None, 3.0, None),
    }
    for mode, (front1, front2, metric1, metric2) in expected.items():
        [node] = stn_entries([entry], mode)
        assert node == {"front1": front1, "front2": front2, "metric1": metric1, "metric2": metric2,
                        "gen_idx": 7, "change_idx": 2}, mode
    assert stn_entries([entry], None)[0]["metric1"] == 1.0  # default npnhv


def test_archive_snapshot_serves_the_passive_archives():
    _, view = view_of("NSGA2", "prior")
    last = view.n_generations - 1
    snapshot = archive_snapshot(view, last)
    noisy, clean, lost = view.noisy_archive(last), view.clean_archive(last), view.lost_clean_solutions(last)
    assert sorted(map(tuple, snapshot["noisy_archive_solutions"])) == sorted(tuple(int(v) for v in x) for x in noisy.x)
    assert sorted(map(tuple, snapshot["clean_archive_solutions"])) == sorted(tuple(int(v) for v in x) for x in clean.x)
    assert sorted(map(tuple, snapshot["lost_clean_solutions"])) == sorted(tuple(int(v) for v in x) for x in lost.x)
    assert snapshot["noisy_archive__noisy"] == noisy.hv["noisy_archive__noisy"]
    assert snapshot["clean_archive__clean"] == clean.hv["clean_archive__clean"] and snapshot["gen_idx"] == last


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
