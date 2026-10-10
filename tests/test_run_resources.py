"""Per-run resource measurement (noisyvis.experiments.resources) and the result rows that carry it.

    measure_run          timings and RAM for one window; the peak is reset per window, so a later window
                         never inherits an earlier window's high-water mark; /proc failures degrade to NaN;
                         an algorithm exception propagates unchanged; no RNG is touched
    SO runner row        so_algo_data_single writes the four fields (the MO row is checked in test_mo_view)
    dashboard loading    historical rows without the fields load as NaN beside new SO and MO rows, and the
                         fields never become table-2 columns (which filter the 2D performance plots)

The reset test controls its memory with an anonymous mmap that is unmapped explicitly: it tests the reset,
not whether Python or glibc returns freed memory. Runs in-process; writes only under tmp_path.
"""

from __future__ import annotations

import math
import mmap
import os
import random

import numpy as np
import pandas as pd
import pytest

from noisyvis.experiments import resources
from noisyvis.experiments.resources import RESOURCE_COLUMNS, measure_run

MIB = 1024 * 1024
CAN_RESET = os.access("/proc/self/clear_refs", os.W_OK) and os.path.exists("/proc/self/status")
needs_reset = pytest.mark.skipif(not CAN_RESET, reason="per-window peak RSS needs Linux /proc/self/clear_refs")


def vm_hwm_mib():
    """The kernel's current VmHWM for this process, read directly (no reset)."""
    return resources._status_mib("VmHWM")


def touched_block(size):
    """An anonymous mapping of `size` bytes with every page written, so all of it is resident."""
    block = mmap.mmap(-1, size)
    for offset in range(0, size, mmap.PAGESIZE):
        block[offset] = 1
    return block


# ------------------------------------------------------------------ measure_run


@needs_reset
def test_every_field_is_a_finite_non_negative_float_and_peak_covers_baseline():
    with measure_run() as usage:
        sum(range(100_000))
    assert tuple(usage) == RESOURCE_COLUMNS
    for key, value in usage.items():
        assert type(value) is float and math.isfinite(value) and value >= 0, (key, value)
    assert usage["peak_ram_mb"] >= usage["ram_baseline_mb"]


@needs_reset
def test_peak_is_per_window_and_not_inherited_from_an_earlier_window():
    size = 256 * MIB
    with measure_run() as first:
        block = touched_block(size)
    block.close()  # munmap: the pages leave the process, independent of any allocator

    # Without a reset the kernel's high-water mark still holds the first window's peak (its RSS counters
    # are batched per CPU, so allow a few MiB against the 256 MiB signal) ...
    assert vm_hwm_mib() >= first["peak_ram_mb"] - 8
    with measure_run() as second:
        pass

    # ... the allocation raised the first window's peak ...
    assert first["peak_ram_mb"] >= first["ram_baseline_mb"] + 0.9 * size / MIB
    # ... and the second window's peak starts from its own baseline, not from that high-water mark.
    assert second["ram_baseline_mb"] <= second["peak_ram_mb"] <= second["ram_baseline_mb"] + 32
    assert second["peak_ram_mb"] <= first["peak_ram_mb"] - 0.5 * size / MIB


@needs_reset
def test_a_window_that_starts_with_retained_memory_reports_it_as_baseline():
    """Memory still resident when a window opens (kept by an allocator, or simply still referenced) is
    that window's baseline; it is part of the absolute peak but does not look like the window's growth."""
    size = 128 * MIB
    block = touched_block(size)
    try:
        with measure_run() as usage:
            pass
    finally:
        block.close()
    assert usage["ram_baseline_mb"] >= 0.9 * size / MIB
    assert usage["ram_baseline_mb"] <= usage["peak_ram_mb"] <= usage["ram_baseline_mb"] + 32


@pytest.mark.parametrize("broken", ["missing", "directory", "garbage"])
def test_unreadable_proc_gives_nan_ram_and_still_times_the_run(monkeypatch, tmp_path, broken):
    if broken == "missing":
        status, clear_refs = tmp_path / "no_status", tmp_path / "no_dir" / "clear_refs"
    elif broken == "directory":
        status = clear_refs = tmp_path
    else:
        status = tmp_path / "status"
        status.write_text("VmRSS:\tnot-a-number kB\nVmHWM:\n")
        clear_refs = tmp_path / "missing_dir" / "clear_refs"
    monkeypatch.setattr(resources, "_PROC_STATUS", str(status))
    monkeypatch.setattr(resources, "_PROC_CLEAR_REFS", str(clear_refs))

    with measure_run() as usage:
        sum(range(10_000))
    assert math.isnan(usage["peak_ram_mb"]) and math.isnan(usage["ram_baseline_mb"])
    assert math.isfinite(usage["run_wall_time_s"]) and usage["run_wall_time_s"] >= 0
    assert math.isfinite(usage["run_cpu_time_s"]) and usage["run_cpu_time_s"] >= 0


def test_peak_is_nan_when_it_cannot_be_reset(monkeypatch, tmp_path):
    """A peak read without a reset would be the process-lifetime mark; it is reported as NaN instead."""
    monkeypatch.setattr(resources, "_PROC_CLEAR_REFS", str(tmp_path / "missing_dir" / "clear_refs"))
    with measure_run() as usage:
        pass
    assert math.isnan(usage["peak_ram_mb"])


@pytest.mark.parametrize("proc", ["real", "broken"])
def test_an_algorithm_exception_propagates_unchanged(monkeypatch, tmp_path, proc):
    if proc == "broken":
        monkeypatch.setattr(resources, "_PROC_STATUS", str(tmp_path))
        monkeypatch.setattr(resources, "_PROC_CLEAR_REFS", str(tmp_path))

    class AlgorithmError(Exception):
        pass

    error = AlgorithmError("raised by the algorithm")
    with pytest.raises(AlgorithmError) as raised:
        with measure_run() as usage:
            raise error
    assert raised.value is error
    assert raised.value.__context__ is None  # nothing from the measurement chained onto it
    assert math.isfinite(usage["run_wall_time_s"]) and math.isfinite(usage["run_cpu_time_s"])


def test_measurement_touches_no_rng():
    random.seed(7)
    np.random.seed(7)
    before = (random.getstate(), repr(np.random.get_state()))
    with measure_run():
        pass
    assert (random.getstate(), repr(np.random.get_state())) == before


# ------------------------------------------------------------------ SO runner row


def test_the_so_runner_row_carries_the_per_run_resources(tmp_path):
    import noisyvis.algorithms
    import noisyvis.problems
    from noisyvis.experiments.runner import so_algo_data_single
    from noisyvis.tracking.logger import get_active_logger

    prob_info = {key: None for key in ("name", "type", "goal", "dimensions", "opt_global", "mean_value",
                                       "mean_weight", "PID", "experiment_name", "experiment_description")}
    algo_config = {"_target_": "noisyvis.algorithms.single_objective.MuPlusLamdaEA", "mu": 1, "lam": 1,
                   "mutate_function": "probFlipBit", "mutate_params": {"indpb": 0.1}}
    algo_params = {
        "sol_length": 10, "opt_weights": (1.0,), "eval_limit": 200, "gen_limit": None,
        "attr_function": noisyvis.algorithms.binary_attribute, "starting_solution": None,
        "target_stop": None, "no_improve_limit": None, "noisy_no_improve_limit": None,
        "fitness_function": (noisyvis.problems.OneMax_fitness, {"noise_intensity": 0}),
        "progress_print_interval": None, "record_population": False,
    }
    rows = [so_algo_data_single(prob_info, algo_config, algo_params, seed, payload_dir=str(tmp_path))[0]
            for seed in (1, 2)]
    assert get_active_logger() is None

    for row in rows:
        assert list(row)[list(row).index("seed_signature") + 1:][:4] == list(RESOURCE_COLUMNS)
        for key in RESOURCE_COLUMNS:
            assert type(row[key]) is float and math.isfinite(row[key]) and row[key] >= 0, (key, row[key])
        assert row["peak_ram_mb"] >= row["ram_baseline_mb"]
        assert row["n_evals"] == 200


# ------------------------------------------------------------------ dashboard loading


def _problem(fit_func="eval_noisy_kp_v1"):
    return dict(PID="kp_synthetic", problem_type="knapsack", problem_goal="maximise", problem_name="kp_10",
                dimensions=10, opt_global=30.0, fit_func=fit_func, experiment_name="exp",
                experiment_description=None)


def _so_row(seed, **resources_):
    return dict(_problem(), algo_type="(1+1)EA", algo_name="(1+1)EA", noise=1.0, seed=seed,
                seed_signature=seed * 11, n_gens=5, n_evals=6, stop_trigger="eval_limit", n_unique_sols=2,
                final_fit=3.0, max_fit=3.0, min_fit=1.0, penalty=1.0, rep_sols=[[0] * 10, [1] * 10],
                rep_fits=[1.0, 3.0], sol_iterations=[1, 1], **resources_)


def _mo_row(seed, **resources_):
    return dict(_problem(fit_func="eval_noisy_kp_v1_mo"), algo_type="SEMO", algo_name="SEMO", noise=1.0,
                seed=seed, seed_signature=seed * 13, n_gens=4, n_evals=5, stop_trigger="gen_limit",
                mo_record={"schema": "noisyvis.mo_record"}, final_hv_current_clean_front__clean=9.0 + seed,
                **resources_)


def _resources(seed):
    return dict(run_wall_time_s=0.1 * seed + 1e-7, run_cpu_time_s=0.09 * seed + 3e-8,
                peak_ram_mb=250.0 + seed + 1 / 3, ram_baseline_mb=225.0 + seed / 7)


def test_old_and_new_rows_load_together_and_resources_never_filter_table_2(tmp_path):
    from noisyvis.dashboard.mo_frames import without_legacy_mo_rows
    from noisyvis.dashboard.tables import create_df_no_lists, create_display2_df
    from noisyvis.results.store import load_algo_results, save_or_append_results

    warehouse = tmp_path / "algo_results.pkl"
    # Historical rows: an SO row with the legacy process-lifetime peak_ram_mb only, and one with none of it.
    historical = pd.DataFrame([_so_row(1, peak_ram_mb=900.5), _so_row(2)])
    save_or_append_results(historical, warehouse)
    # New rows appended by the runners: SO and MO, two seeds each, all four fields, distinct per seed.
    new = pd.DataFrame([_so_row(seed, **_resources(seed)) for seed in (3, 4)]
                       + [_mo_row(seed, **_resources(seed)) for seed in (1, 2)])
    save_or_append_results(new, warehouse)

    df = without_legacy_mo_rows(load_algo_results(warehouse))
    assert len(df) == 6
    old, so, mo = df.iloc[:2], df.iloc[2:4], df.iloc[4:]
    assert old[["run_wall_time_s", "run_cpu_time_s", "ram_baseline_mb"]].isna().all().all()
    assert old["peak_ram_mb"].iloc[0] == 900.5 and math.isnan(old["peak_ram_mb"].iloc[1])
    for part, seeds in ((so, (3, 4)), (mo, (1, 2))):
        expected = pd.DataFrame([_resources(seed) for seed in seeds], index=part.index)
        pd.testing.assert_frame_equal(part[list(RESOURCE_COLUMNS)], expected[list(RESOURCE_COLUMNS)])

    df_no_lists = create_df_no_lists(df)
    display2_df = create_display2_df(df)
    # Per-seed values stay in the per-run frame at full precision ...
    assert set(RESOURCE_COLUMNS) <= set(df_no_lists.columns)
    assert df_no_lists.loc[so.index, "run_wall_time_s"].tolist() == [_resources(s)["run_wall_time_s"] for s in (3, 4)]
    # ... and never reach table 2, whose columns become the 2D-plot filters.
    assert not set(RESOURCE_COLUMNS) & set(display2_df.columns)

    # Selecting every table-2 row keeps every run, as callbacks.performance._filter_by_table2_selection
    # does it: each column shared by display2_df and df_no_lists filters by the selected rows' values.
    filter_columns = [col for col in display2_df.columns if col in df_no_lists.columns]
    mask = pd.Series(True, index=df_no_lists.index)
    for col in filter_columns:
        allowed = display2_df[col].unique()
        if any(pd.isnull(allowed)):
            mask &= df_no_lists[col].isin(allowed) | df_no_lists[col].isnull()
        else:
            mask &= df_no_lists[col].isin(allowed)
    assert mask.all(), f"table 2 would drop runs: {df_no_lists.loc[~mask, ['algo_name', 'seed']]}"
    assert display2_df.set_index("algo_name")["no_runs"].to_dict() == {"(1+1)EA": 4, "SEMO": 2}
