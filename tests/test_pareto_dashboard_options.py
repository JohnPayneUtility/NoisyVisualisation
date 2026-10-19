"""The Pareto tab: its five front types, the three two-front "Noisy" plot types, and the data every plot reads.

    front types      the multi-objective STN's front kinds (FRONT_KEYS == mo_frames.STN_FRONT_KINDS):
                     noisy population, noisy population corrected, clean population, historical archive
                     noisy, historical archive clean
    entries          pareto_front_entries(view, kind): one per change of the kind's state (distinct genotypes
                     or stored HV, the STN rule), holding every member at that generation (noisy kinds: each
                     observation under its observed y; the others: each genotype under f(x)) and the stored
                     HV; the last entry carries the final clean archive (IGD's reference)
    pairs            pareto_pair_entries(view, a, b): both fronts wherever either changes (as the STN pairs)
    plots            every front, HV, IGD and movement plot reads the selected kind's members, objective
                     values and HV; distances are between distinct genotypes
    callback         updateParetoPlot reacts to the selected rows and every control; the Noisy types ignore
                     the front dropdown
    staircase        front_line: the attainment boundary for any mix of maximised and minimised objectives

Runs in-process and writes nothing.
"""

from __future__ import annotations

import ast
import functools
import inspect
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from harness.mo_runs import build, step_run
from noisyvis.common import front_distance, sol_tuple_ints
from noisyvis.dashboard import mo_frames
from noisyvis.dashboard.mo_frames import (
    PARETO_NOISY_PAIRS,
    STN_FRONT_KINDS,
    pareto_front_entries,
    pareto_front_members,
    pareto_pair_entries,
    pareto_runs,
    stn_front_state,
    stn_state_items,
    stn_view_entries,
)
from noisyvis.problems.instances import get_knapsack_problem_stats
from noisyvis.results.mo_view import MORunView
from noisyvis.tracking.logger import clear_active_logger, get_active_logger
from noisyvis.viz.plots import get_pareto_plot
from noisyvis.viz.plots.base import (
    FRONT_KEYS,
    UNSUPPORTED_METRIC_TITLE,
    create_empty_figure,
    front_line,
    genotype_front_distance,
    selected_front,
    unique_solutions,
)
from noisyvis.viz.plots.pareto import plot_basic, plot_igd_vs_dist, plot_noisy
from test_mo_stn_views import crafted

KINDS = tuple(STN_FRONT_KINDS)
NOISY_KINDS = ('noisy_population', 'noisy_archive')
METRIC = {kind: metric for kind, (_, _, metric) in STN_FRONT_KINDS.items()}
FRONT_PLOTS = ("Basic", "Subplots", "SubplotsMulti", "SubplotsHighlight", "paretoAnimation")
INDICATOR_PLOTS = ("IndVsDist", "IGDVsDist", "PPM", "MoveCorr", "Hist", "Scatter")
PARETO_PY = mo_frames.__file__.replace("mo_frames.py", "callbacks/pareto.py")


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


@functools.lru_cache(maxsize=None)
def record_of(name, prob):
    if name == "Crafted":
        clear_active_logger()
        record = crafted()[0].log.finalise()
        clear_active_logger()
        return record
    algo = build(name, prob=prob)
    step_run(algo)
    return algo.eval_log.finalise()


def view_of(name, prob):
    return mo_frames.row_view(pd.DataFrame([{"mo_record": record_of(name, prob)}]), 0)


RUNS = [("Crafted", None), ("NSGA2", "prior"), ("MoUMDA", "kp1"), ("SEMO", "cocz1")]


def one_run(name, prob, entries_of):
    """pareto_runs over a one-row frame: a single series with a single run."""
    df = pd.DataFrame([{"algo_name": name, "noise": 1.0, "mo_record": record_of(name, prob)}])
    return pareto_runs(df, [{"algo_name": name, "noise": 1.0, "_row": 0}], entries_of)


def kind_run(name, prob, kind, copies=1):
    groups, labels = one_run(name, prob, lambda view: pareto_front_entries(view, kind))
    return [[groups[0][0]] * copies], labels


def multiset(points):
    return sorted(tuple(float(v) for v in p) for p in points)


def weighted_desc(points, weights):
    w = np.asarray(points, dtype=float) * np.asarray(weights, dtype=float)
    return all(tuple(w[i]) >= tuple(w[i + 1]) for i in range(len(w) - 1))


def moves(entries):
    """(generations, dD, dO) between successive entries: distinct-genotype distance and stored-HV change."""
    gens = [e["gen_idx"] for e in entries[1:]]
    dD = [float(genotype_front_distance(p["front_solutions"], c["front_solutions"])) for p, c in zip(entries, entries[1:])]
    dO = [float(c["front_hypervolume"] - p["front_hypervolume"]) for p, c in zip(entries, entries[1:])]
    return gens, dD, dO


# ------------------------------------------------------------------------------ front types

def test_front_types_are_the_stn_front_kinds():
    assert set(FRONT_KEYS) == set(STN_FRONT_KINDS)
    assert {kind: label for kind, (*_, label) in FRONT_KEYS.items()} == {
        'noisy_population': 'noisy population', 'corrected_noisy': 'noisy population corrected',
        'clean_population': 'clean population', 'noisy_archive': 'historical archive noisy',
        'clean_archive': 'historical archive clean'}
    assert all(keys[:3] == ('front_solutions', 'front_fitnesses', 'front_hypervolume') for keys in FRONT_KEYS.values())
    assert [selected_front(k) for k in KINDS] == list(KINDS)
    assert selected_front('NoisyHV') is None and selected_front(None) is None
    assert set(PARETO_NOISY_PAIRS) == {'NoisyPopulationVsClean', 'NoisyArchiveVsClean', 'NoisyPopulationVsCorrected'}


# ------------------------------------------------------------------------------ entries

def _dominates(a, b, weights):
    wa, wb = np.asarray(a) * weights, np.asarray(b) * weights
    return bool(np.all(wa >= wb) and np.any(wa > wb))


@pytest.mark.parametrize("name,prob", RUNS)
@pytest.mark.parametrize("kind", KINDS)
def test_entries_are_their_generations_snapshots(name, prob, kind):
    """Each entry is the kind's front at the end of its own generation, read independently from MORunView:
    noisy kinds keep every observation (repeated genotypes included) under observed y; the clean kinds every
    genotype under f(x); the corrected front the noisy front's distinct genotypes no other dominates under
    f(x), once each. HV is the stored metric; members are in descending weighted-objective order."""
    view = view_of(name, prob)
    weights = np.asarray(view.weights, dtype=float)
    snapshot_at = {'noisy_population': view.current_noisy_front, 'corrected_noisy': view.current_noisy_front,
                   'clean_population': view.current_clean_front, 'noisy_archive': view.noisy_archive,
                   'clean_archive': view.clean_archive}[kind]
    entries = pareto_front_entries(view, kind)
    assert entries
    for e in entries:
        snapshot = snapshot_at(e["gen_idx"])
        if kind in NOISY_KINDS:
            expected_x = [tuple(int(v) for v in x) for x in snapshot.x]
            expected_y = list(snapshot.observed)
            assert len(e["front_solutions"]) == len(snapshot.obs_ids)          # every observation
        elif kind == 'corrected_noisy':
            true_of = {tuple(int(v) for v in x): tuple(t) for x, t in zip(snapshot.x, snapshot.true_obj)}
            keep = [g for g in true_of if not any(_dominates(true_of[o], true_of[g], weights) for o in true_of if o != g)]
            expected_x, expected_y = keep, [true_of[g] for g in keep]
        else:
            expected_x = [tuple(int(v) for v in x) for x in snapshot.x]
            expected_y = list(snapshot.true_obj)
        assert sorted(map(tuple, e["front_solutions"])) == sorted(expected_x), (kind, e["gen_idx"])
        assert multiset(e["front_fitnesses"]) == multiset(expected_y), (kind, e["gen_idx"])
        assert e["front_hypervolume"] == snapshot.hv[METRIC[kind]]
        assert weighted_desc(e["front_fitnesses"], weights)
        # each member's solution and objective values belong together
        pairs = {(tuple(map(int, s)), tuple(map(float, f))) for s, f in zip(e["front_solutions"], e["front_fitnesses"])}
        assert pairs == {(x, tuple(map(float, y))) for x, y in zip(expected_x, expected_y)}


def _reference_generations(view, kind):
    """The STN rule applied at every generation."""
    return [i["gen_idx"] for i in stn_state_items((g, stn_front_state(view, kind, g), None)
                                                  for g in range(view.n_generations))]


@pytest.mark.parametrize("name,prob", RUNS)
def test_entry_generations_follow_the_stn_rule(name, prob):
    view = view_of(name, prob)
    for kind in KINDS:
        entries = pareto_front_entries(view, kind)
        assert [e["gen_idx"] for e in entries] == _reference_generations(view, kind), kind
        assert [e["change_idx"] for e in entries] == list(range(len(entries)))
    for kind, view_id in (('noisy_population', 'population_noisy'), ('noisy_archive', 'archive_noisy')):
        assert [e["gen_idx"] for e in pareto_front_entries(view, kind)] == \
            [i["gen_idx"] for i in stn_view_entries(view, view_id)]
    assert pareto_front_entries(view, 'clean_archive') is pareto_front_entries(view, 'clean_archive')   # cached


def test_crafted_noisy_population_keeps_repeated_observations():
    """The crafted run's g0 noisy front: A observed twice (a, a2), B, C: four members of three genotypes;
    its corrected front: A and C once each."""
    view = view_of("Crafted", None)
    [g0] = [e for e in pareto_front_entries(view, 'noisy_population') if e["gen_idx"] == 0]
    assert len(g0["front_solutions"]) == 4 and len(unique_solutions(g0["front_solutions"])) == 3
    [c0] = [e for e in pareto_front_entries(view, 'corrected_noisy') if e["gen_idx"] == 0]
    assert len(c0["front_solutions"]) == len(unique_solutions(c0["front_solutions"])) == 2


@pytest.mark.parametrize("name,prob", RUNS)
def test_igd_reference_is_the_final_clean_archive(name, prob):
    view = view_of(name, prob)
    final = view.clean_archive(view.n_generations - 1)
    for kind in KINDS:
        entries = pareto_front_entries(view, kind)
        assert multiset(entries[-1]["final_clean_archive_fitnesses"]) == multiset(final.true_obj)
        assert all("final_clean_archive_fitnesses" not in e for e in entries[:-1])


@pytest.mark.parametrize("name,prob", RUNS)
@pytest.mark.parametrize("plot_type", sorted(PARETO_NOISY_PAIRS))
def test_pair_entries_follow_the_paired_stn_view(name, prob, plot_type):
    view_id = {'NoisyPopulationVsClean': 'population_noisy_vs_clean', 'NoisyArchiveVsClean': 'archive_noisy_vs_clean',
               'NoisyPopulationVsCorrected': 'population_noisy_vs_corrected'}[plot_type]
    a, b = PARETO_NOISY_PAIRS[plot_type]
    view = view_of(name, prob)
    entries = pareto_pair_entries(view, a, b)
    assert [e["gen_idx"] for e in entries] == [i["gen_idx"] for i in stn_view_entries(view, view_id)]
    for e in entries:
        assert (e["a_solutions"], e["a_fitnesses"]) == pareto_front_members(view, a, e["gen_idx"])
        assert (e["b_solutions"], e["b_fitnesses"]) == pareto_front_members(view, b, e["gen_idx"])


def test_runs_are_grouped_as_process_stn_data_groups_them():
    rows, records = [], []
    for i, (name, prob, noise) in enumerate([("MoUMDA", "kp1", 1.0), ("NSGA2", "prior", 1.0), ("SEMO", "cocz1", 2.0)]):
        rows.append({"algo_name": "A" if i < 2 else "B", "noise": noise, "_row": i})
        records.append({"mo_record": record_of(name, prob)})
    df = pd.DataFrame(records)
    groups, labels = pareto_runs(df, rows, lambda view: pareto_front_entries(view, 'clean_archive'))
    assert labels == [["A", 1.0], ["B", 2.0]]
    assert [str(label) for label in labels] == ["['A', 1.0]", "['B', 2.0]"]   # plain values in titles, not np.float64
    assert groups[0][0] == pareto_front_entries(view_of("MoUMDA", "kp1"), 'clean_archive')
    assert groups[0][1] == pareto_front_entries(view_of("NSGA2", "prior"), 'clean_archive')
    assert groups[1][0] == pareto_front_entries(view_of("SEMO", "cocz1"), 'clean_archive')
    assert pareto_runs(df, [], lambda view: []) == ([], [])


# ------------------------------------------------------------------------------ the plots read the kind's data

@pytest.mark.parametrize("name,prob", RUNS)
@pytest.mark.parametrize("kind", KINDS)
def test_front_plots_draw_the_kinds_members(name, prob, kind):
    """Basic, the subplot plots and the animation draw each entry's members as they are (repeated
    observations of a noisy kind kept)."""
    frontdata, labels = kind_run(name, prob, kind, copies=3)       # Subplots draws run 2
    expected = [multiset(e["front_fitnesses"]) for e in frontdata[0][0]]
    for plot_type in FRONT_PLOTS:
        fig = get_pareto_plot(plot_type)(frontdata, labels, IndVsDist_IndType=kind, attainment=True)
        traces = [frame.data[0] for frame in fig.frames] if plot_type == "paretoAnimation" else \
            [t for t in fig.data if t.mode == "lines+markers"]
        drawn = [multiset(zip(t.x, t.y)) for t in traces]
        if plot_type == "SubplotsMulti":
            drawn = drawn[:len(expected)]
        assert drawn == expected, plot_type
        assert FRONT_KEYS[kind][3] in fig.layout.title.text, plot_type


def _igd(approx, ref):
    return float(np.mean([min(float(np.linalg.norm(np.asarray(r) - np.asarray(p))) for p in approx) for r in ref]))


@pytest.mark.parametrize("name,prob", RUNS)
@pytest.mark.parametrize("kind", KINDS)
def test_indicator_plots_read_the_kinds_hv_members_and_objectives(name, prob, kind):
    frontdata, labels = kind_run(name, prob, kind)
    entries = frontdata[0][0]
    gens, dD, dO = moves(entries)
    hvs = [e["front_hypervolume"] for e in entries]
    cum = list(np.cumsum([0.0] + dD))

    # HV vs distance: the stored HVs against the cumulative distinct-genotype distance of the kind's members
    [trace] = get_pareto_plot("IndVsDist")(frontdata, labels, distance_method="cumulative", IndVsDist_IndType=kind).data
    assert list(trace.y) == hvs and list(trace.x) == pytest.approx(cum)
    # IGD vs distance: the kind's objective values against the final clean archive, written out here
    [trace] = plot_igd_vs_dist(frontdata, labels, distance_method="cumulative", IndVsDist_IndType=kind).data
    ref = entries[-1]["final_clean_archive_fitnesses"]
    assert list(trace.y) == pytest.approx([_igd(e["front_fitnesses"], ref) for e in entries])
    assert list(trace.x) == pytest.approx(cum)
    assert all(np.isfinite(trace.y))
    if len(entries) < 2:
        return
    # Progress per movement and movement correlation: dD and dO of the kind
    for plot_type in ("PPM", "MoveCorr"):
        kwargs = {"window": 5} if plot_type == "MoveCorr" else {}
        fig = get_pareto_plot(plot_type)(frontdata, labels, IndVsDist_IndType=kind, **kwargs)
        by_name = {t.name.split(" ")[0]: t for t in fig.data}
        assert list(by_name["dD"].x) == gens and list(by_name["dD"].y) == pytest.approx(dD), plot_type
        assert list(by_name["dO"].y) == pytest.approx(dO), plot_type
    # Histogram: the non-zero moves pooled; Scatter: every move
    hist = get_pareto_plot("Hist")(frontdata, labels, IndVsDist_IndType=kind)
    kept = [(d, o) for d, o in zip(dD, dO) if d != 0.0 or o != 0.0]
    assert list(hist.data[0].x) == pytest.approx([d for d, _ in kept])
    assert list(hist.data[1].x) == pytest.approx([o for _, o in kept])
    [scatter] = get_pareto_plot("Scatter")(frontdata, labels, IndVsDist_IndType=kind).data
    assert list(scatter.x) == pytest.approx(dD) and list(scatter.y) == pytest.approx(dO)


def test_the_five_kinds_give_five_different_series():
    """On the crafted run every kind's HV series (and so every HV, IGD and movement plot) differs: a plot
    reading another kind's data would not pass the tests above."""
    view = view_of("Crafted", None)
    series = {kind: tuple((e["gen_idx"], e["front_hypervolume"], len(e["front_solutions"]))
                          for e in pareto_front_entries(view, kind)) for kind in KINDS}
    assert len(set(series.values())) == len(KINDS)


def test_unknown_front_type_is_refused_not_substituted():
    frontdata, labels = kind_run("MoUMDA", "kp1", 'noisy_population')
    for plot_type in FRONT_PLOTS + INDICATOR_PLOTS:
        fig = get_pareto_plot(plot_type)(frontdata, labels, IndVsDist_IndType="NoisyHV")
        assert fig.layout.title.text == UNSUPPORTED_METRIC_TITLE and not fig.data, plot_type


# ------------------------------------------------------------------------------ the Noisy (A vs B) plots

@pytest.mark.parametrize("name,prob", RUNS)
@pytest.mark.parametrize("plot_type", sorted(PARETO_NOISY_PAIRS))
def test_noisy_plots_draw_both_fronts_and_link_shared_genotypes(name, prob, plot_type):
    a, b = PARETO_NOISY_PAIRS[plot_type]
    groups, labels = one_run(name, prob, lambda view: pareto_pair_entries(view, a, b))
    entries = groups[0][0]
    fig = plot_noisy(groups, labels, labels=(FRONT_KEYS[a][3], FRONT_KEYS[b][3]))
    markers = {t.name: t for t in fig.data if t.mode == "markers"}
    assert multiset(zip(markers[FRONT_KEYS[a][3]].x, markers[FRONT_KEYS[a][3]].y)) == \
        multiset(p for e in entries for p in e["a_fitnesses"])
    assert multiset(zip(markers[FRONT_KEYS[b][3]].x, markers[FRONT_KEYS[b][3]].y)) == \
        multiset(p for e in entries for p in e["b_fitnesses"])
    # a connector per A member whose genotype is on B, from its observed point to that genotype's point
    expected = []
    for e in entries:
        on_b = {sol_tuple_ints(s): f for s, f in zip(e["b_solutions"], e["b_fitnesses"])}
        expected += [(tuple(map(float, f)), tuple(map(float, on_b[sol_tuple_ints(s)])))
                     for s, f in zip(e["a_solutions"], e["a_fitnesses"]) if sol_tuple_ints(s) in on_b]
    connectors = [((t.x[0], t.y[0]), (t.x[1], t.y[1])) for t in fig.data if t.mode == "lines"]
    assert sorted(connectors) == sorted(expected)
    if plot_type == 'NoisyPopulationVsCorrected':
        # every corrected genotype is on the noisy front, so each is linked from all its observations
        for e in entries:
            noisy = {sol_tuple_ints(s) for s in e["a_solutions"]}
            assert {sol_tuple_ints(s) for s in e["b_solutions"]} <= noisy
    assert fig.layout.title.text.startswith(f"{FRONT_KEYS[a][3]} vs {FRONT_KEYS[b][3]}")
    assert all(len(t.x) == 2 and t.line.shape is None for t in fig.data if t.mode == "lines")


# ------------------------------------------------------------------------------ the callback

def _callback(namespace):
    """updateParetoPlot, GRIDLINE_PLOTS and knapsack_gridlines exec'd from callbacks/pareto.py into namespace
    (importing the module would load the dashboard data)."""
    tree = ast.parse(open(PARETO_PY).read())
    [func] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "updateParetoPlot"]
    helpers = [n for n in tree.body
               if (isinstance(n, ast.FunctionDef) and n.name == "knapsack_gridlines")
               or (isinstance(n, ast.Assign) and [t.id for t in n.targets] == ["GRIDLINE_PLOTS"])]
    assert len(helpers) == 2
    decorator = func.decorator_list[0]
    func.decorator_list = []
    exec(compile(ast.Module(body=helpers + [func], type_ignores=[]), PARETO_PY, "exec"), namespace)
    return namespace["updateParetoPlot"], decorator


# The fixture's problems: a knapsack instance, a knapsack instance whose row is not named 'knapsack', and a
# PID that is no knapsack instance
KP10, KP20, NOT_KP = "f1_l-d_kp_10_269", "f2_l-d_kp_20_878", "onemax_10"
PROBLEMS = [(KP10, "knapsack"), (KP20, "other"), (NOT_KP, "knapsack"), (KP10, "knapsack")]


@pytest.fixture(scope="module")
def pareto_callback():
    rows, records = [], []
    for i, ((name, prob), (pid, prob_name)) in enumerate(zip(RUNS, PROBLEMS)):  # one series of four runs
        rows.append({"algo_name": "S", "noise": 1.0, "_row": i})               # (Subplots draws run 2)
        records.append({"mo_record": record_of(name, prob), "PID": pid, "problem_name": prob_name})
    namespace = {"go": go, "df": pd.DataFrame(records), "PARETO_NOISY_PAIRS": PARETO_NOISY_PAIRS,
                 "pareto_front_entries": pareto_front_entries, "pareto_pair_entries": pareto_pair_entries,
                 "pareto_runs": pareto_runs, "get_pareto_plot": get_pareto_plot, "FRONT_KEYS": FRONT_KEYS,
                 "UNSUPPORTED_METRIC_TITLE": UNSUPPORTED_METRIC_TITLE, "create_empty_figure": create_empty_figure,
                 "selected_front": selected_front, "FRONT_PLOTS": FRONT_PLOTS,
                 "get_knapsack_problem_stats": get_knapsack_problem_stats}
    update, decorator = _callback(namespace)
    return update, decorator, rows


def test_callback_reacts_to_the_rows_and_every_control(pareto_callback):
    _, decorator, _ = pareto_callback
    inputs = [(c.args[0].value, c.args[1].value) for c in decorator.args[1].elts]
    assert inputs == [('STN_data', 'data'), ('paretoFrontPlotType', 'value'), ('IndVsDist_IndType', 'value'),
                      ('IndVsDist_DistType', 'value'), ('paretoPlotNumRuns', 'value'),
                      ('paretoPlotWindowSize', 'value'), ('paretoAttainmentSurface', 'value'),
                      ('paretoInvalidGridlines', 'value'), ('PID', 'data')]
    assert len(decorator.args) == 2 and not decorator.keywords                  # no State


def test_callback_renders_every_plot_type_with_every_front_type(pareto_callback):
    update, _, rows = pareto_callback
    for plot_type in FRONT_PLOTS + INDICATOR_PLOTS:
        for kind in KINDS:
            fig = update(rows, plot_type, kind, "cumulative", 2, 5, ["on"])
            assert isinstance(fig, go.Figure) and fig.data, (plot_type, kind)
            assert FRONT_KEYS[kind][3] in fig.layout.title.text or plot_type in ("IndVsDist", "IGDVsDist"), plot_type
        assert update(rows, plot_type, "NoisyHV", "cumulative", 2, 5, ["on"]).layout.title.text == UNSUPPORTED_METRIC_TITLE
    for plot_type, (a, b) in PARETO_NOISY_PAIRS.items():
        figs = [update(rows, plot_type, kind, "cumulative", 2, 5, ["on"]).to_plotly_json() for kind in KINDS + (None,)]
        assert all(f == figs[0] for f in figs), plot_type                        # the front dropdown is ignored
        assert figs[0]["layout"]["title"]["text"].startswith(f"{FRONT_KEYS[a][3]} vs {FRONT_KEYS[b][3]}")
    assert update([], "Basic", "noisy_population", "cumulative", 1, 5, []).layout.title.text == "No multi-objective data"


def test_callback_draws_the_selected_kinds_data(pareto_callback):
    """Through the callback, the plotted HVs, distances and points are those of the selected kind's entries
    for the first run (the Crafted run, whose five kinds all differ)."""
    update, _, rows = pareto_callback
    view = view_of("Crafted", None)
    for kind in KINDS:
        entries = pareto_front_entries(view, kind)
        [trace] = update(rows, "IndVsDist", kind, "cumulative", 1, 5, []).data
        assert list(trace.y) == [e["front_hypervolume"] for e in entries], kind
        assert list(trace.x) == pytest.approx(list(np.cumsum([0.0] + moves(entries)[1])))
        basic = update(rows, "Basic", kind, "cumulative", 1, 5, [])
        assert [multiset(zip(t.x, t.y)) for t in basic.data] == [multiset(e["front_fitnesses"]) for e in entries]


# ------------------------------------------------------------------------------ distance and staircase

def test_distance_counts_distinct_genotypes_only():
    """A genotype observed three times weighs once in the distance (front_distance alone would count it
    three times on the 'from' side)."""
    a = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]
    b = [[0, 0, 0, 0]]
    assert unique_solutions(a) == [[0, 0, 0, 0], [1, 1, 1, 1]]
    assert front_distance(a, b) == pytest.approx(0.125)
    assert genotype_front_distance(a, b) == pytest.approx(0.25)
    assert genotype_front_distance(b, b) == 0.0


def _corner_ok(corner, a, b, signs):
    """corner is weakly dominated by both neighbours (no better than either in any objective)."""
    return all(s * c <= s * v for point in (a, b) for c, v, s in zip(corner, point, signs))


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)])
def test_staircase_is_the_attainment_boundary(weights):
    signs = tuple(1.0 if w > 0 else -1.0 for w in weights)
    best_first = [(9.0, 1.0), (7.0, 2.0), (4.0, 6.0), (2.0, 8.0), (1.0, 9.0)]   # non-dominated, maximising
    pts = [(signs[0] * u1, signs[1] * u2) for u1, u2 in best_first]
    shuffled = pts[:]
    random.Random(0).shuffle(shuffled)
    xs, ys, shape = front_line(shuffled, weights, attainment=True)
    assert shape == "hv"
    assert list(zip(xs, ys)) == pts                     # best to worst on objective 1
    for i in range(len(xs) - 1):                        # 'hv' corner: (x of next, y of this)
        assert _corner_ok((xs[i + 1], ys[i]), (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), signs)


def test_staircase_ties_duplicates_and_single_points():
    # duplicates: zero-length steps, nothing dropped
    xs, ys, shape = front_line([(1.0, 3.0), (2.0, 2.0), (2.0, 2.0)], (1.0, 1.0), attainment=True)
    assert list(zip(xs, ys)) == [(2.0, 2.0), (2.0, 2.0), (1.0, 3.0)] and shape == "hv"
    # a tie on objective 1: worst to best on objective 2, so the boundary rises along x = 3 without a dip
    xs, ys, _ = front_line([(1.0, 5.0), (3.0, 2.0), (3.0, 1.0)], (1.0, 1.0), attainment=True)
    assert list(zip(xs, ys)) == [(3.0, 1.0), (3.0, 2.0), (1.0, 5.0)]
    assert front_line([(5.0, 7.0)], (1.0, -1.0), attainment=True) == ([5.0], [7.0], "hv")
    assert front_line([], (1.0, -1.0), attainment=True) == ([], [], "hv")


def test_staircase_off_keeps_the_straight_line():
    pts = [(1.0, 3.0), (3.0, 1.0), (2.0, 2.0)]
    assert front_line(pts, (1.0, 1.0), attainment=False) == ([1.0, 3.0, 2.0], [3.0, 1.0, 2.0], None)
    assert front_line(pts, None, attainment=True)[2] is None                 # directions unknown
    assert front_line(pts, (1.0, 1.0, 1.0), attainment=True)[2] is None      # not a two-objective front


@pytest.mark.parametrize("kind", KINDS)
def test_front_plots_draw_staircases_only_when_asked(kind):
    frontdata, labels = kind_run("MoUMDA", "kp1", kind, copies=3)
    for plot_type in FRONT_PLOTS:
        plot = get_pareto_plot(plot_type)
        on = plot(frontdata, labels, IndVsDist_IndType=kind, attainment=True)
        off = plot(frontdata, labels, IndVsDist_IndType=kind, attainment=False)
        lines = [t for t in on.data if t.mode == "lines+markers"]
        assert lines and all(t.line.shape == "hv" for t in lines), plot_type
        # knapsack: maximise objective 1, so a staircase runs from high to low x
        assert all(list(t.x) == sorted(t.x, reverse=True) for t in lines), plot_type
        assert all(t.line.shape is None for t in off.data), plot_type
        assert off.to_plotly_json() == plot(frontdata, labels, IndVsDist_IndType=kind).to_plotly_json()


def test_noisy_plot_has_no_attainment_option():
    assert "attainment" not in inspect.signature(plot_noisy).parameters


# ------------------------------------------------------------------------------ invalid-solution gridlines

def gridlines_of(fig):
    """(vertical x positions, horizontal y positions) of the figure's grey straight-line shapes."""
    shapes = [s for s in fig.layout.shapes if s.type == "line" and s.line.color == "grey"]
    assert len(shapes) == len(fig.layout.shapes)
    return ([s.x0 for s in shapes if s.x0 == s.x1 and s.xref == "x"],
            [s.y0 for s in shapes if s.y0 == s.y1 and s.yref == "y"])


def test_basic_plot_draws_gridlines_only_when_given():
    frontdata, labels = kind_run("MoUMDA", "kp1", "noisy_population")
    plain = plot_basic(frontdata, labels)
    assert not plain.layout.shapes
    assert plot_basic(frontdata, labels, gridlines=None).to_plotly_json() == plain.to_plotly_json()
    fig = plot_basic(frontdata, labels, gridlines=(412.0, 0.0))
    assert gridlines_of(fig) == ([412.0], [0.0])
    assert [t.to_plotly_json() for t in fig.data] == [t.to_plotly_json() for t in plain.data]
    assert gridlines_of(plot_basic(frontdata, labels, gridlines=(None, 0.0))) == ([], [0.0])
    assert gridlines_of(plot_basic(frontdata, labels, gridlines=(412.0, None))) == ([412.0], [])


def test_knapsack_gridlines_are_the_value_sum_and_zero_weight(pareto_callback):
    gridlines = pareto_callback[0].__globals__["knapsack_gridlines"]
    assert gridlines(KP10) == (get_knapsack_problem_stats(KP10)["sum_values"], 0.0) == (412.0, 0.0)
    assert gridlines(KP20) is None                                   # its rows are not named 'knapsack'
    assert gridlines(NOT_KP) is None                                 # named 'knapsack', but no instance
    assert gridlines("unknown_pid") is None and gridlines([]) is None and gridlines(None) is None


def test_callback_draws_gridlines_on_basic_for_a_knapsack_when_ticked(pareto_callback):
    update, _, rows = pareto_callback
    kind = "noisy_population"
    ticked = update(rows, "Basic", kind, "cumulative", 1, 5, ["on"], ["on"], KP10)
    assert gridlines_of(ticked) == ([412.0], [0.0])
    without = update(rows, "Basic", kind, "cumulative", 1, 5, ["on"])
    assert not without.layout.shapes
    assert [t.to_plotly_json() for t in ticked.data] == [t.to_plotly_json() for t in without.data]
    for gridlines_value, pid in (([], KP10), (None, KP10), (["on"], KP20), (["on"], NOT_KP), (["on"], [])):
        fig = update(rows, "Basic", kind, "cumulative", 1, 5, ["on"], gridlines_value, pid)
        assert fig.to_plotly_json() == without.to_plotly_json(), (gridlines_value, pid)
    for plot_type in FRONT_PLOTS[1:] + INDICATOR_PLOTS + tuple(PARETO_NOISY_PAIRS):  # Basic only, for now
        on = update(rows, plot_type, kind, "cumulative", 2, 5, ["on"], ["on"], KP10)
        off = update(rows, plot_type, kind, "cumulative", 2, 5, ["on"], [], KP10)
        assert on.to_json() == off.to_json(), plot_type
