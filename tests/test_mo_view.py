"""MORunView over persisted MO records (MO recording plan v6, Work Group 5).

The round trip runtime logger -> finalise() -> mo_record -> MORunView reproduces the Work Group 2-4
runtime API exactly, from the persisted structures alone (no replay, no HV recomputation):

    events and generations     x, x~, f(x), y, obs_id; gen_last_eval; generation_of / evals_in
    current fronts             population-based noisy (observations) and clean (genotypes) fronts and
                               their generation-resolution histories (re-appearances kept)
    passive archives           per generation, after any number of events, at any evaluation; membership
                               intervals; evaluation-resolution histories whose states are atomic
                               post-evaluation memberships
    hypervolume                all six metrics, as change points and dense per-generation trajectories
    lost_clean_solutions(g)    clean archive members the population no longer holds

It also checks the snapshot semantics, the NumPy-only import footprint, the result row the MO runner
now writes, and that the dashboard tables never carry mo_record.

Runs in-process (plus one clean subprocess) and writes nothing.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

import noisyvis.algorithms
import noisyvis.problems
from harness.mo_runs import A, ALGORITHM_NAMES, B, C, F_A, F_B, F_C, Lab, build, step_run
from noisyvis.results.mo_view import (
    ARCHIVE_METRICS,
    CURRENT_FRONT_METRICS,
    HV_METRICS,
    NOT_EXITED,
    MORunView,
)
from noisyvis.tracking.logger import clear_active_logger, get_active_logger

WORKSPACE = Path(__file__).resolve().parents[1]
PROBLEMS = ["kp0", "kp1", "kpviol1", "cocz1", "prior"]
SUMMARY_COLUMNS = [f"final_hv_{m}" for m in HV_METRICS] + [
    "n_generated_genotypes", "final_noisy_archive_size", "final_clean_archive_size"]


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


def run(name, prob):
    """A logged run, its view, and the genotype ids of the population at every generation."""
    algo = build(name, prob=prob)
    populations = []
    step_run(algo, lambda algo: populations.append(
        {algo.eval_log.orig_geno[algo.eval_log.resolve(ind)] for ind in algo.population}))
    return algo, MORunView(algo.eval_log.finalise()), populations


def ids(array):
    return tuple(int(v) for v in array)


# ------------------------------------------------------------------------------ round trip

@pytest.mark.parametrize("prob", PROBLEMS)
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_the_view_reproduces_the_runtime_log(name, prob):
    algo, view, populations = run(name, prob)
    log = algo.eval_log
    assert (view.n_evals, view.n_generations) == (len(log), log.n_generations)
    assert ids(view.gen_last_eval) == tuple(log.gen_last_eval)

    # events and the generation mapping
    events = view.events()
    for e in range(len(log)):
        want, got = log.event(e), view.event(e)
        assert (got.orig_geno, got.eval_geno, got.obs_id) == (want.orig_geno, want.eval_geno, want.obs_id)
        assert ids(got.x) == want.x and ids(got.x_tilde) == want.x_tilde
        assert tuple(got.true_obj) == tuple(float(v) for v in want.true_objectives)
        assert tuple(got.observed) == tuple(float(v) for v in want.observed)
        assert got.generation == int(np.searchsorted(log.gen_last_eval, e, side="right"))
        assert events.orig_geno[e] == want.orig_geno and events.obs_id[e] == want.obs_id
    for g in range(view.n_generations):
        span = view.evals_in(g)
        assert span.stop == log.gen_last_eval[g] == view.evals_through(g)
        assert all(view.generation_of(e) == g for e in span)

    # current fronts, per generation and as histories
    for g in range(view.n_generations):
        noisy, clean = view.current_noisy_front(g), view.current_clean_front(g)
        assert ids(noisy.obs_ids) == log.noisy_front_at(g) and ids(clean.geno_ids) == log.clean_front_at(g)
        for metric in CURRENT_FRONT_METRICS:
            snapshot = noisy if metric.startswith("current_noisy") else clean
            assert snapshot.hv[metric] == log.hv_trajectory(metric)[g]
    for which, gens, members in (("noisy", log.noisy_front_gens, log.noisy_front_members),
                                 ("clean", log.clean_front_gens, log.clean_front_members)):
        history = getattr(view, f"current_{which}_front_history")()
        assert history.resolution == "generation" and ids(history.starts) == tuple(gens)
        assert [ids(state.members) for state in history] == list(members)

    # passive archives: per generation, after every event, at every evaluation, intervals
    for g in range(view.n_generations):
        assert ids(view.noisy_archive(g).obs_ids) == log.noisy_archive_at(g)
        assert ids(view.clean_archive(g).geno_ids) == log.clean_archive_at(g)
    for n in range(len(log) + 1):
        assert ids(view.noisy_archive_after(n).members) == log.noisy_archive_after(n)
        assert ids(view.clean_archive_after(n).members) == log.clean_archive_after(n)
    assert ids(view.noisy_archive_at_eval(len(log) - 1).members) == log.noisy_archive_at_eval(len(log) - 1)
    for which in ("noisy", "clean"):
        replay, intervals = getattr(log, f"{which}_archive"), getattr(view, f"{which}_archive_intervals")()
        assert (ids(intervals.member), ids(intervals.enter_eval), ids(intervals.exit_eval), ids(intervals.exit_by)) \
            == (replay.members, replay.enter, replay.exit, replay.exit_by)
        history = getattr(view, f"{which}_archive_history")()
        assert history.resolution == "evaluation"
        for state, start in zip(history, history.starts):
            assert ids(state.members) == getattr(log, f"{which}_archive_after")(int(start) + 1)
        by_generation = getattr(view, f"{which}_archive_history")("generation")
        expected, previous = [], None
        for g in range(view.n_generations):
            members = getattr(log, f"{which}_archive_at")(g)
            if members != previous:
                expected.append((g, members))
                previous = members
        assert [(int(s), ids(state.members)) for s, state in zip(by_generation.starts, by_generation)] == expected

    # all six hypervolume metrics: dense trajectories and change points
    for metric in HV_METRICS:
        series = view.hv(metric)
        runtime = log.hv_trajectory(metric) if metric in CURRENT_FRONT_METRICS else log.archive_hv_trajectory(metric)
        assert list(series.by_generation) == runtime
        if metric in ARCHIVE_METRICS:
            evals, values = log._archives()["hv"][metric]
            assert ids(series.positions) == tuple(evals) and list(series.values) == values
            assert all(series.after(n) == log.archive_hv_after(metric, n) for n in range(len(log) + 1))

    # lost clean solutions: the archive members the population no longer holds
    for g in range(view.n_generations):
        lost = set(ids(view.lost_clean_solutions(g).geno_ids))
        archive = set(log.clean_archive_at(g))
        assert lost == archive - set(log.clean_front_at(g))
        assert not lost & populations[g] and (archive & populations[g]) <= set(log.clean_front_at(g))


# ------------------------------------------------------------------------------ constructed cases

def test_evaluation_states_are_atomic():
    """{A, B, C} then, at one evaluation, D enters and evicts A and B: the next state is {C, D}. No
    intermediate membership such as {B, C, D} is ever a state."""
    lab = Lab()
    d_bits = [0, 0, 0, 1]
    a = lab.evaluate(A, (50, 50), (50.0, 50.0))
    b = lab.evaluate(B, (40, 40), (40.0, 40.0))
    c = lab.evaluate(C, (30, 30), (30.0, 30.0))
    d = lab.evaluate(d_bits, (55, 35), (55.0, 35.0))            # dominates A and B, not C
    lab.observe([a, b, c, d])
    view = MORunView(lab.log.finalise())
    for which, member in (("noisy", lab.obs), ("clean", lambda ind: lab.geno(list(ind)))):
        history = getattr(view, f"{which}_archive_history")()
        states = [set(ids(state.members)) for state in history]
        assert ids(history.starts) == (0, 1, 2, 3)
        assert states == [{member(a)}, {member(a), member(b)}, {member(a), member(b), member(c)},
                          {member(c), member(d)}]
        assert set(ids(history.at(3).members)) == {member(c), member(d)}
        assert set(ids(history.at_generation(0).members)) == {member(c), member(d)}
    series = view.hv("noisy_archive__noisy")
    assert series.resolution == "evaluation" and len(series.positions) == 4 and series.at_eval(3) == series.values[-1]
    with pytest.raises(ValueError):
        view.hv("current_noisy_front__noisy").after(1)


def test_reappearing_memberships_are_later_history_states():
    """The Work Group 3 sequence: the clean front {A} -> {A, C} -> {A} again is three states."""
    lab = Lab()
    a, b = lab.evaluate(A, F_A, (80.0, 70.0)), lab.evaluate(B, F_B, (95.0, 55.0))
    lab.observe([a, b])
    c = lab.evaluate(C, F_C, (70.0, 90.0))
    lab.observe([a, b, c])
    lab.observe([a, b])
    history = MORunView(lab.log.finalise()).current_clean_front_history()
    assert ids(history.starts) == (0, 1, 2)
    assert [ids(state.members) for state in history] == [(lab.geno(A),), tuple(sorted((lab.geno(A), lab.geno(C)))),
                                                         (lab.geno(A),)]
    assert ids(history.at(1).members) == ids(history[1].members) and ids(history[-1].members) == (lab.geno(A),)
    for bad in (-1, 3):
        with pytest.raises(IndexError):
            history.at(bad) if bad < 0 else history.at_generation(bad)


def test_snapshots_never_fabricate_observations_for_clean_states():
    algo, view, _ = run("NSGA2", "prior")
    last = view.n_generations - 1
    for snapshot in (view.current_clean_front(last), view.clean_archive(last), view.lost_clean_solutions(last)):
        assert snapshot.obs_ids is None and snapshot.observed is None and snapshot.x_tilde is None
        assert snapshot.x.shape == (len(snapshot), view.sol_length) and np.isfinite(snapshot.true_obj).all()
    noisy = view.noisy_archive(last)
    assert noisy.observed.shape == (len(noisy), 2) and noisy.x_tilde.shape == noisy.x.shape
    assert noisy.kind == "noisy_archive" and noisy.generation == last and noisy.n_evals == view.n_evals
    assert set(noisy.hv) == {"noisy_archive__noisy", "noisy_archive__clean"}
    assert any(not np.array_equal(event.x, event.x_tilde) for event in map(view.event, range(view.n_evals)))
    with pytest.raises(ValueError):
        view.genotype(0)[0] = 1                                  # read-only
    for bad in (lambda: view.current_noisy_front(view.n_generations), lambda: view.event(view.n_evals),
                lambda: view.noisy_archive_after(view.n_evals + 1), lambda: view.hv("current_front")):
        with pytest.raises((IndexError, KeyError)):
            bad()


def test_the_view_is_numpy_only():
    """A clean interpreter unpickles a record, builds the view and queries every part of it without
    importing deap, dash, mlflow or pandas."""
    algo = build("MoUMDA_ParetoArchive", prob="prior")
    step_run(algo)
    script = (
        "import pickle, sys\n"
        "from noisyvis.results.mo_view import MORunView, HV_METRICS\n"
        "view = MORunView(pickle.loads(sys.stdin.buffer.read()))\n"
        "last = view.n_generations - 1\n"
        "view.current_noisy_front(last); view.current_clean_front(last); view.noisy_archive(last)\n"
        "view.clean_archive(last); view.lost_clean_solutions(last); view.events(); view.event(0)\n"
        "list(view.noisy_archive_history()); list(view.clean_archive_history('generation'))\n"
        "[view.hv(m).by_generation for m in HV_METRICS]; view.summary_scalars()\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'deap', 'dash', 'mlflow', 'pandas'}))\n"
    )
    done = subprocess.run([sys.executable, "-c", script], input=pickle.dumps(algo.eval_log.finalise()),
                          capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr.decode()[-2000:]
    assert done.stdout.decode().strip() == "[]"


# ------------------------------------------------------------------------------ the MO runner's row

def runner_row(target):
    """One seed through the real MO runner function on the harness knapsack config."""
    from noisyvis.experiments.config.workflows import resolve_mo_config
    from noisyvis.experiments.runner import mo_algo_data_single

    cfg = OmegaConf.create(yaml.safe_load((WORKSPACE / "tests" / "configs" / "mo_knapsack.yaml").read_text()))
    if target != "SEMO":
        cfg.algo = OmegaConf.create({"name": target, "type": target, "init_args": {
            "_target_": f"noisyvis.algorithms.multi_objective.{target}", "pop_size": 20, "select_size": 10}})
        cfg.run.max_gens, cfg.run.eval_limit = 12, None
    cfg = resolve_mo_config(cfg)
    fitness_fn = getattr(noisyvis.problems, cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)
    algo_params = {
        "sol_length": cfg.problem.dimensions, "opt_weights": tuple(cfg.problem.weights),
        "eval_limit": cfg.run.eval_limit, "starting_solution": None, "target_stop": None,
        "attr_function": getattr(noisyvis.algorithms, cfg.problem.attr_function),
        "gen_limit": cfg.run.max_gens, "stop_without_improvement_in_gens": None,
        "fitness_function": (fitness_fn, fit_params),
        "true_fitness_function": (fitness_fn, dict(fit_params, noise_intensity=0)),
        "ref_point": cfg.problem.get("ref_point", None), "verbose_rate": 0,
    }
    prob_info = {key: None for key in ("name", "type", "goal", "dimensions", "opt_global", "mean_value",
                                       "mean_weight", "PID", "experiment_name", "experiment_description")}
    return mo_algo_data_single(prob_info, cfg.algo.init_args, algo_params, seed=1)


def legacy_change_generations(view):
    """Generations of the legacy recorder's entries: where the noisy front's genotype set changed."""
    generations, last = [], None
    for g in range(view.n_generations):
        signature = frozenset(tuple(int(v) for v in x) for x in view.current_noisy_front(g).x)
        if signature != last:
            generations.append(g)
            last = signature
    return generations


@pytest.mark.parametrize("target", ["SEMO", "MoUMDA"])
def test_the_mo_runner_writes_mo_record_beside_the_legacy_columns(target):
    """The runner enables logging and finalises: its row holds mo_record and the nine summaries (equal to
    the view's), while the legacy columns are unchanged in meaning, reproduced entry by entry by the view,
    and carry no provenance tags."""
    row = runner_row(target)
    view = MORunView(row["mo_record"])
    assert {k: row[k] for k in SUMMARY_COLUMNS} == view.summary_scalars()
    assert (row["n_evals"], row["n_gens"] + 1) == (view.n_evals, view.n_generations)
    for front in row["pareto_solutions"] + row["true_pareto_solutions"]:
        assert all("_eval_tag" not in vars(ind) for ind in front)

    changes = legacy_change_generations(view)
    assert len(changes) == len(row["pareto_solutions"])
    for k, g in enumerate(changes):
        noisy, clean = view.current_noisy_front(g), view.current_clean_front(g)
        assert sorted((tuple(map(int, ind)), tuple(map(float, fit)))
                      for ind, fit in zip(row["pareto_solutions"][k], row["pareto_fitnesses"][k])) == \
            sorted((tuple(map(int, x)), tuple(map(float, y))) for x, y in zip(noisy.x, noisy.observed))
        assert sorted((tuple(map(int, ind)), tuple(map(float, fit)))
                      for ind, fit in zip(row["true_pareto_solutions"][k], row["true_pareto_fitnesses"][k])) == \
            sorted((tuple(map(int, x)), tuple(map(float, t))) for x, t in zip(clean.x, clean.true_obj))
        assert (row["noisy_pf_noisy_hypervolumes"][k], row["noisy_pf_true_hypervolumes"][k],
                row["true_pf_hypervolumes"][k]) == \
            (noisy.hv["current_noisy_front__noisy"], noisy.hv["current_noisy_front__clean"],
             clean.hv["current_clean_front__clean"])
    assert row["final_true_hv"] == view.hv("current_clean_front__clean").at_generation(changes[-1])
    assert row["n_gens_pareto_best"] == [b - a for a, b in zip(changes, changes[1:] + [view.n_generations])]

    pickle.loads(pickle.dumps(row["mo_record"]))


def test_dashboard_tables_never_carry_mo_record():
    """The table builders drop mo_record (DashboardData.load drops it before them too); the summaries
    stay in df_no_lists for the performance plots and out of the algorithm-selection table."""
    from noisyvis.dashboard.tables import create_df_no_lists, create_display2_df

    row = runner_row("MoUMDA")
    df = pd.DataFrame([row, dict(row, seed=2)])
    no_lists, display2 = create_df_no_lists(df), create_display2_df(df)
    assert "mo_record" not in no_lists.columns and "mo_record" not in display2.columns
    assert set(SUMMARY_COLUMNS) <= set(no_lists.columns)
    assert not set(SUMMARY_COLUMNS) & set(display2.columns)
    assert NOT_EXITED == -1
