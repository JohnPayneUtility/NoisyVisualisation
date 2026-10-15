"""The Pareto tab's front sequences, its historical-archive and attainment-surface options, and the metric
selection (Noisy HV / Clean HV) shared by every front and hypervolume plot.

    sequences        sequence_entries(view, source, front): one entry per change of the selected front,
                     source 'current' (population fronts) or 'historical' (passive archives), each entry
                     holding both fronts exactly as they stood at the end of its generation
    front_runs       the MO_data_PPP shape for any sequence, grouped as process_STN_data groups rows
    metric           Noisy HV is the noisy front (or archive) under y, Clean HV the clean front (or
                     archive) under f(x); HV and front distance come from the same sequence; distances are
                     between distinct genotypes; any other selection is refused, never substituted
    staircase        front_line: the attainment boundary for any mix of maximised and minimised objectives
    reuse            change points read from the record, HV series built once per view, opened records
                     and built sequences cached by identity, all without changing a value

Runs in-process and writes nothing.
"""

from __future__ import annotations

import functools
import inspect
import random

import numpy as np
import pandas as pd
import pytest

from harness.mo_runs import build, step_run
from noisyvis.common import front_distance
from noisyvis.dashboard.mo_frames import change_generations, front_entries, front_runs, row_view, sequence_entries
from noisyvis.results.mo_view import HV_METRICS, MORunView
from noisyvis.tracking.logger import clear_active_logger, get_active_logger
from noisyvis.viz.plots import get_pareto_plot
from noisyvis.viz.plots.base import UNSUPPORTED_METRIC_TITLE, front_line, genotype_front_distance, unique_solutions
from noisyvis.viz.plots.pareto import plot_igd_vs_dist, plot_ind_vs_dist, plot_noisy

RUNS = [("NSGA2", "prior"), ("MoUMDA", "kp1"), ("SEMO", "cocz1")]
SEQUENCES = [(source, front) for source in ("current", "historical") for front in ("noisy", "clean")]
FRONT_PLOTS = ("Basic", "Subplots", "SubplotsMulti", "SubplotsHighlight", "paretoAnimation")
HV_PLOTS = ("IndVsDist", "MoveCorr", "Hist", "Scatter")
METRIC = {"noisy": "NoisyHV", "clean": "CleanHV"}
HV_KEY = {"noisy": "algo_front_noisy_hypervolume", "clean": "clean_front_hypervolume"}
SOLUTION_KEY = {"noisy": "algo_front_solutions", "clean": "clean_front_solutions"}
FITNESS_KEY = {"noisy": "algo_front_noisy_fitnesses", "clean": "clean_front_fitnesses"}


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


@functools.lru_cache(maxsize=None)
def record_of(name, prob):
    algo = build(name, prob=prob)
    step_run(algo)
    return algo.eval_log.finalise()


def view_of(name, prob):
    return MORunView(record_of(name, prob))


def one_run(name, prob, source, front):
    """front_runs over a one-row frame: a single series with a single run."""
    df = pd.DataFrame([{"algo_name": name, "noise": 1.0, "mo_record": record_of(name, prob)}])
    return front_runs(df, [{"algo_name": name, "noise": 1.0, "_row": 0}], source, front)


def three_runs(name, prob, source, front):
    """The same run three times: Subplots always draws run 2."""
    [[entries]] = one_run(name, prob, source, front)
    return [[entries, entries, entries]]


def points(values):
    return sorted(tuple(float(v) for v in p) for p in values)


def genotypes(x):
    return sorted(tuple(int(v) for v in s) for s in x)


def snapshots(view, source, gen):
    if source == "historical":
        return view.noisy_archive(gen), view.clean_archive(gen)
    return view.current_noisy_front(gen), view.current_clean_front(gen)


# ------------------------------------------------------------------------------ sequences

@pytest.mark.parametrize("source,front", SEQUENCES)
@pytest.mark.parametrize("name,prob", RUNS)
def test_each_entry_holds_its_generations_fronts(name, prob, source, front):
    """Entries sit exactly at the selected front's change generations, and each holds the noisy and clean
    front (or archive) of the end of that generation, with that generation's recorded hypervolumes."""
    view = view_of(name, prob)
    entries = sequence_entries(view, source, front)
    if (source, front) == ("current", "noisy"):
        expected = change_generations(view)
    elif source == "current":
        expected = list(view.current_clean_front_history().starts)
    else:
        expected = list(getattr(view, f"{front}_archive_history")("generation").starts)
    assert [e["gen_idx"] for e in entries] == expected
    assert [e["change_idx"] for e in entries] == list(range(len(entries)))

    metrics = (("noisy_archive__noisy", "noisy_archive__clean", "clean_archive__clean") if source == "historical"
               else ("current_noisy_front__noisy", "current_noisy_front__clean", "current_clean_front__clean"))
    for e in entries:
        g = e["gen_idx"]
        noisy, clean = snapshots(view, source, g)
        assert genotypes(e["algo_front_solutions"]) == genotypes(noisy.x)
        assert points(e["algo_front_noisy_fitnesses"]) == points(noisy.observed)
        assert points(e["algo_front_clean_fitnesses"]) == points(noisy.true_obj)
        assert genotypes(e["clean_front_solutions"]) == genotypes(clean.x)
        assert points(e["clean_front_fitnesses"]) == points(clean.true_obj)
        assert (e["algo_front_noisy_hypervolume"], e["algo_front_clean_hypervolume"], e["clean_front_hypervolume"]) \
            == tuple(view.hv(m).at_generation(g) for m in metrics)
        assert e["opt_weights"] == list(view.weights)

    # the IGD reference: the clean front (or archive) of the final generation, on the last entry only
    final = snapshots(view, source, view.n_generations - 1)[1]
    assert points(entries[-1]["final_clean_front_fitnesses"]) == points(final.true_obj)
    assert not any("final_clean_front_fitnesses" in e for e in entries[:-1])


def test_current_noisy_sequence_is_front_entries():
    view = view_of("MoUMDA", "kp1")
    assert front_entries(view) == sequence_entries(view, "current", "noisy")


@pytest.mark.parametrize("name,prob", RUNS)
def test_change_generations_match_a_per_generation_scan(name, prob):
    """Reading the recorded change points gives exactly the generations a scan of every generation finds."""
    view = view_of(name, prob)
    scanned, last = [], None
    for g in range(view.n_generations):
        signature = frozenset(tuple(int(v) for v in x) for x in view.current_noisy_front(g).x)
        if signature != last:
            scanned.append(g)
            last = signature
    assert change_generations(view) == scanned


def test_hv_series_are_built_once_per_view():
    view, fresh = view_of("NSGA2", "prior"), view_of("NSGA2", "prior")
    for metric in HV_METRICS:
        series = view.hv(metric)
        assert view.hv(metric) is series
        assert np.array_equal(series.by_generation, fresh.hv(metric).by_generation, equal_nan=True)
        assert not series.by_generation.flags.writeable
    with pytest.raises(KeyError):
        view.hv("no_such_metric")


def test_records_and_sequences_are_reused_by_identity():
    """The same record yields the same view and the same built sequence; another record in the same row id
    (a different frame) yields its own."""
    first = pd.DataFrame([{"mo_record": record_of("MoUMDA", "kp1")}])
    second = pd.DataFrame([{"mo_record": record_of("SEMO", "cocz1")}])
    view = row_view(first, 0)
    assert row_view(first, 0) is view
    assert row_view(second, 0) is not view and row_view(second, 0).n_evals == view_of("SEMO", "cocz1").n_evals
    entries = sequence_entries(view, "historical", "noisy")
    assert sequence_entries(view, "historical", "noisy") is entries
    assert sequence_entries(view, "historical", "clean") is not entries
    assert entries == sequence_entries(view_of("MoUMDA", "kp1"), "historical", "noisy")   # a fresh build


@pytest.mark.parametrize("name,prob", RUNS)
def test_historical_entries_are_per_generation_snapshots(name, prob):
    """A historical entry holds only what was found up to and including its generation, never the final
    archive; with a fixed reference point both archives' hypervolumes are non-decreasing."""
    view = view_of(name, prob)
    for front in ("noisy", "clean"):
        entries = sequence_entries(view, "historical", front)
        for e in entries:
            noisy, clean = snapshots(view, "historical", e["gen_idx"])
            bound = view.evals_through(e["gen_idx"])
            assert (noisy.eval_ids < bound).all() and (clean.first_generated_eval < bound).all()
        hvs = [e[HV_KEY[front]] for e in entries]
        assert np.all(np.diff(hvs) >= 0), f"{front} archive HV decreased: {hvs}"
    noisy_entries = sequence_entries(view_of("MoUMDA", "kp1"), "historical", "noisy")
    assert genotypes(noisy_entries[0]["algo_front_solutions"]) != genotypes(noisy_entries[-1]["algo_front_solutions"])


def test_noisy_archive_pairs_each_observation_with_its_own_clean_value():
    """Under prior noise (x~ may differ from x), each archived observation's y is paired with f(x) of the
    genotype that same observation was generated as, never with another observation's."""
    view = view_of("NSGA2", "prior")
    assert view.prior_noise_observed
    for e in sequence_entries(view, "historical", "noisy"):
        archive = view.noisy_archive(e["gen_idx"])
        expected = []
        for eval_id in archive.eval_ids:
            event = view.event(int(eval_id))
            expected.append((tuple(float(v) for v in event.observed), tuple(float(v) for v in event.true_obj)))
        paired = list(zip(map(tuple, e["algo_front_noisy_fitnesses"]), map(tuple, e["algo_front_clean_fitnesses"])))
        assert sorted(paired) == sorted(expected)


def test_front_runs_groups_like_process_stn_data():
    """Series in groupby order of (algo_name, noise), runs in row order; rows without a record skipped."""
    df = pd.DataFrame([
        {"algo_name": "B", "noise": 0.5, "mo_record": record_of("SEMO", "cocz1")},
        {"algo_name": "A", "noise": 1.0, "mo_record": record_of("MoUMDA", "kp1")},
        {"algo_name": "A", "noise": 1.0, "mo_record": record_of("NSGA2", "prior")},
        {"algo_name": "A", "noise": 1.0, "mo_record": None},
    ])
    rows = [{"algo_name": "B", "noise": 0.5, "_row": 0}, {"algo_name": "A", "noise": 1.0, "_row": 1},
            {"algo_name": "A", "noise": 1.0, "_row": 2}, {"algo_name": "A", "noise": 1.0, "_row": 3}]
    groups = front_runs(df, rows, "historical", "clean")
    assert [len(runs) for runs in groups] == [2, 1]
    assert groups[0][0] == sequence_entries(view_of("MoUMDA", "kp1"), "historical", "clean")
    assert groups[0][1] == sequence_entries(view_of("NSGA2", "prior"), "historical", "clean")
    assert groups[1][0] == sequence_entries(view_of("SEMO", "cocz1"), "historical", "clean")
    assert front_runs(df, [], "current", "noisy") == []


# ------------------------------------------------------------------------------ metric and distance

def test_distance_counts_distinct_genotypes_only():
    """A genotype observed three times weighs once in the distance (front_distance alone would count it
    three times on the 'from' side)."""
    a = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]
    b = [[0, 0, 0, 0]]
    assert unique_solutions(a) == [[0, 0, 0, 0], [1, 1, 1, 1]]
    assert front_distance(a, b) == pytest.approx(0.125)
    assert genotype_front_distance(a, b) == pytest.approx(0.25)
    assert genotype_front_distance(b, b) == 0.0


def test_hv_keeps_every_observation_while_distance_dedups():
    """HV vs Distance plots the recorded HVs (over every distinct observation) against distinct-genotype
    distances."""
    entries = [
        {"algo_front_solutions": [[0, 0], [0, 0]], "algo_front_noisy_hypervolume": 1.5, "gen_idx": 0},
        {"algo_front_solutions": [[0, 0], [0, 0], [1, 1]], "algo_front_noisy_hypervolume": 2.5, "gen_idx": 3},
    ]
    [trace] = plot_ind_vs_dist([[entries]], ["s"], IndVsDist_IndType="NoisyHV").data
    assert list(trace.y) == [1.5, 2.5]
    assert list(trace.x) == pytest.approx([0.0, 0.25])


@pytest.mark.parametrize("source,front", SEQUENCES)
def test_hv_vs_distance_reads_one_front_sequence(source, front):
    """Both axes come from the selected sequence: y its HVs, x the cumulative distinct-genotype distance
    between its successive solution sets."""
    frontdata = one_run("NSGA2", "prior", source, front)
    entries = frontdata[0][0]
    [trace] = plot_ind_vs_dist(frontdata, ["s"], distance_method="cumulative", IndVsDist_IndType=METRIC[front]).data
    assert list(trace.y) == [e[HV_KEY[front]] for e in entries]
    sols = [e[SOLUTION_KEY[front]] for e in entries]
    expected = np.cumsum([0.0] + [genotype_front_distance(p, c) for p, c in zip(sols, sols[1:])])
    assert list(trace.x) == pytest.approx(list(expected))
    assert list(trace.marker.color) == [e["gen_idx"] for e in entries]


def test_igd_reference_is_the_final_generation():
    """The last entry's final_clean_front_fitnesses is the reference, not that entry's clean front."""
    entries = [
        {"algo_front_solutions": [[0]], "algo_front_noisy_fitnesses": [[0.0, 0.0]],
         "clean_front_fitnesses": [[0.0, 0.0]], "gen_idx": 0},
        {"algo_front_solutions": [[1]], "algo_front_noisy_fitnesses": [[3.0, 4.0]],
         "clean_front_fitnesses": [[0.0, 0.0]], "final_clean_front_fitnesses": [[3.0, 4.0]], "gen_idx": 2},
    ]
    [trace] = plot_igd_vs_dist([[entries]], ["s"]).data
    assert list(trace.y) == [5.0, 0.0]


@pytest.mark.parametrize("front", ["noisy", "clean"])
def test_front_plots_follow_the_metric(front):
    """Basic, the subplot plots and the animation draw the selected front: the noisy front under y or the
    clean front under f(x)."""
    frontdata = three_runs("MoUMDA", "kp1", "current", front)
    entries = frontdata[0][0]
    expected = [points(e[FITNESS_KEY[front]]) for e in entries]
    for plot_type in FRONT_PLOTS:
        fig = get_pareto_plot(plot_type)(frontdata, ["s"], IndVsDist_IndType=METRIC[front], attainment=True)
        traces = [t for t in fig.data if t.mode == "lines+markers"]
        if plot_type == "paretoAnimation":
            traces = [frame.data[0] for frame in fig.frames]
        assert [points(zip(t.x, t.y)) for t in traces] == expected, plot_type
        assert ("Clean PF" in fig.layout.title.text) == (front == "clean"), plot_type


def test_unsupported_metric_is_refused_not_substituted():
    frontdata = one_run("MoUMDA", "kp1", "current", "noisy")
    for plot_type in FRONT_PLOTS + HV_PLOTS:
        fig = get_pareto_plot(plot_type)(frontdata, ["s"], IndVsDist_IndType="IGD")
        assert fig.layout.title.text == UNSUPPORTED_METRIC_TITLE and not fig.data, plot_type


# ------------------------------------------------------------------------------ staircase

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


@pytest.mark.parametrize("source,front", SEQUENCES)
def test_front_plots_draw_staircases_only_when_asked(source, front):
    frontdata = three_runs("MoUMDA", "kp1", source, front)
    for plot_type in FRONT_PLOTS:
        plot = get_pareto_plot(plot_type)
        on = plot(frontdata, ["s"], IndVsDist_IndType=METRIC[front], attainment=True)
        off = plot(frontdata, ["s"], IndVsDist_IndType=METRIC[front], attainment=False)
        lines = [t for t in on.data if t.mode == "lines+markers"]
        assert lines and all(t.line.shape == "hv" for t in lines), plot_type
        # knapsack: maximise objective 1, so a staircase runs from high to low x
        assert all(list(t.x) == sorted(t.x, reverse=True) for t in lines), plot_type
        assert all(t.line.shape is None for t in off.data), plot_type
        assert off.to_plotly_json() == plot(frontdata, ["s"], IndVsDist_IndType=METRIC[front]).to_plotly_json()


def test_noisy_plot_keeps_its_direct_connectors():
    """The Noisy plot has no front lines: its clean-to-noisy connectors stay two-point straight lines."""
    assert "attainment" not in inspect.signature(plot_noisy).parameters
    fig = plot_noisy(one_run("NSGA2", "prior", "historical", "noisy"), ["s"])
    connectors = [t for t in fig.data if t.mode == "lines"]
    assert connectors and all(len(t.x) == 2 and t.line.shape is None for t in connectors)
