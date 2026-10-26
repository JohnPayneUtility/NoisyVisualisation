"""The Pareto tab's "Pareto Rank Layers" plot type: a run's whole population at one generation, by rank.

    steps        mo_frames.population_steps: the STN rule applied to population states, so a step starts
                 exactly where the multiset of member eval_ids changes (a change in a lower rank is a step
                 even while rank 0 is unchanged; a reorder alone is not)
    layers       mo_frames.rank_layer_data: the population at the *exact* generation, every member under
                 its rank among the whole population (PopulationSnapshot.noisy_rank / .true_rank), its
                 recorded objective values and its identities; filtering only selects ranks
    controls     mo_frames.rank_layer_controls / rank_layer_trigger: the run, step, generation and rank
                 range, and how they follow each other
    figure       viz.plots.pareto.plot_rank_layers: rank colours, draw order, legend, hover, axes
    callbacks    the Pareto callbacks exec'd from callbacks/pareto.py, and the real Dash callback chain
                 (harness/rank_layer_chain.py) over the HTTP protocol in an isolated temp root

Every member comparison is a multiset (collections.Counter), so repeated eval_ids (clones) and repeated
observations count as often as they occur. In-process except the chain test; writes nothing.
"""

from __future__ import annotations

import ast
import copy
import functools
import json
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.colors import sample_colorscale

from harness.mo_runs import ALGORITHM_NAMES, A, B, C, Lab, build, step_run
from harness.run_isolated import HARNESS_DIR, child_env, make_temp_root
from noisyvis.dashboard import mo_frames
from noisyvis.dashboard.mo_frames import (
    NO_POPULATION_MESSAGE,
    NO_RUNS_MESSAGE,
    PARETO_NOISY_PAIRS,
    RANK_LAYERS_PLOT,
    pareto_front_entries,
    pareto_pair_entries,
    pareto_runs,
    population_steps,
    rank_layer_controls,
    rank_layer_data,
    rank_layer_runs,
    rank_layer_selection,
    rank_layer_trigger,
)
from noisyvis.problems.instances import get_knapsack_problem_stats
from noisyvis.results.mo_view import MORunView
from noisyvis.tracking.logger import clear_active_logger, get_active_logger
from noisyvis.viz.plots import get_pareto_plot
from noisyvis.viz.plots.base import FRONT_KEYS, UNSUPPORTED_METRIC_TITLE, create_empty_figure, front_line, selected_front
from noisyvis.viz.plots.pareto import plot_rank_layers
from noisyvis.viz.plots.pareto.rank_layers import (
    HOVER_ID_LIMIT,
    LEGEND_MAX_RANKS,
    MAX_MARKER_SIZE,
    RANK0_COLOR,
    RANK_COLORSCALE,
    rank_color,
)

PROBLEMS = ("kp1", "cocz1", "prior")
RUNS = [(name, prob, False) for name in ALGORITHM_NAMES for prob in PROBLEMS] + \
       [("SEMO", "cocz1", True), ("NSGA2", "kp1", True)]
BASES = ("noisy", "true")
D, E = [0, 0, 0, 1], [1, 1, 0, 0]
PARETO_PY = mo_frames.__file__.replace("mo_frames.py", "callbacks/pareto.py")
FRONT_PLOTS = ("Basic", "Subplots", "SubplotsMulti", "SubplotsHighlight", "paretoAnimation")


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


@functools.lru_cache(maxsize=None)
def record_of(name, prob, long=False):
    """A finalised record; `long` runs SEMO 300 and the others 40 generations without early stopping, so
    SEMO keeps its population over many generations."""
    clear_active_logger()
    params = {"gen_limit": 300 if name == "SEMO" else 40, "stop_without_improvement_in_gens": None} if long else {}
    algo = build(name, prob=prob, **params)
    step_run(algo)
    record = algo.eval_log.finalise()
    clear_active_logger()
    return record


@functools.lru_cache(maxsize=None)
def version_1(name, prob, long=False):
    record = copy.deepcopy(record_of(name, prob, long))
    record.pop("current_population")
    record["version"] = 1
    return record


def view_of(name, prob, long=False):
    return mo_frames.row_view(pd.DataFrame([{"mo_record": record_of(name, prob, long)}]), 0)


def values_of(snapshot, basis):
    return snapshot.observed if basis == "noisy" else snapshot.true_obj


def snapshot_members(snapshot, basis, rows=None):
    """The members (eval_id, genotype, observation, rank, x, y) of a snapshot's rows, as a multiset."""
    values, ranks = values_of(snapshot, basis), snapshot.rank(basis)
    rows = range(len(snapshot)) if rows is None else rows
    return Counter((int(snapshot.eval_ids[i]), int(snapshot.geno_ids[i]), int(snapshot.obs_ids[i]), int(ranks[i]),
                    float(values[i, 0]), float(values[i, 1])) for i in rows)


def shown_members(layers):
    """The members the layers show, expanded from their points, as a multiset."""
    return Counter((e, g, o, p["rank"], p["x"], p["y"]) for p in layers["points"]
                   for e, g, o in zip(p["eval_ids"], p["geno_ids"], p["obs_ids"]))


def plotted(layers):
    """What the plot draws: each point's rank, values, multiplicity and member multiset (no order)."""
    return Counter((p["rank"], p["x"], p["y"], p["count"], tuple(sorted(p["eval_ids"]))) for p in layers["points"])


def lab_view(*populations):
    """MORunView of a Lab run observing the given populations (lists of point names) in turn."""
    lab = Lab()
    points = {"a": (A, (100, 50), (100.0, 50.0)), "b": (B, (90, 60), (90.0, 60.0)),
              "c": (C, (110, 55), (110.0, 55.0)), "d": (D, (50, 90), (50.0, 90.0)),
              "e": (E, (80, 70), (80.0, 70.0))}
    ind = {k: lab.evaluate(*v) for k, v in points.items()}
    for population in populations:
        lab.observe([ind[k] for k in population])
    clear_active_logger()
    return MORunView(lab.log.finalise()), {k: lab.log.resolve(v) for k, v in ind.items()}


# ------------------------------------------------------------------------------ steps

@pytest.mark.parametrize("name,prob,long", RUNS)
def test_steps_are_the_population_multiset_changes(name, prob, long):
    """A step starts exactly at the generations whose member multiset differs from the previous
    generation's (generation 0 always); every generation is in its own step."""
    view = view_of(name, prob, long)
    counts = [Counter(view.population(g).eval_ids.tolist()) for g in range(view.n_generations)]
    expected = [g for g in range(view.n_generations) if g == 0 or counts[g] != counts[g - 1]]
    steps = population_steps(view)
    assert steps.starts.tolist() == expected
    assert steps.ends.tolist() == [s - 1 for s in expected[1:]] + [view.n_generations - 1]
    assert len(steps) <= len(view.population_history())
    for g in range(view.n_generations):
        k = steps.step_of(g)
        assert steps.starts[k] <= g <= steps.ends[k]
        assert counts[g] == counts[steps.starts[k]]
    assert population_steps(view) is steps                                   # built once per view


def test_a_lower_rank_change_is_a_step_while_rank_0_is_unchanged():
    """[a, b, d] then [a, b, e]: rank 0 ({a}) and its HV never change, so the front-based steps (the STN
    rule over the noisy front) see one state; the population steps see the rank-2 member change."""
    view, eid = lab_view("abd", "abe")
    assert [e["gen_idx"] for e in pareto_front_entries(view, "noisy_population")] == [0]
    assert population_steps(view).starts.tolist() == [0, 1]
    before, after = rank_layer_data(view, 0), rank_layer_data(view, 1)
    assert [(p["rank"], p["eval_ids"]) for p in before["points"]] == [(0, [eid["a"]]), (1, [eid["b"]]), (2, [eid["d"]])]
    assert [(p["rank"], p["eval_ids"]) for p in after["points"]] == [(0, [eid["a"]]), (1, [eid["b"]]), (2, [eid["e"]])]


@pytest.mark.parametrize("name,prob", [("NSGA2", "kp1"), ("NSGA2", "cocz1"), ("MoUMDA_KMeans", "kp0")])
def test_real_runs_have_population_steps_the_front_steps_skip(name, prob):
    """Real 40-generation runs where the population changes in lower ranks while the noisy front does not
    (in others, e.g. NSGA-II under prior noise, the front happens to change at every population change)."""
    view = view_of(name, prob, True)
    population = set(population_steps(view).starts.tolist())
    front = {e["gen_idx"] for e in pareto_front_entries(view, "noisy_population")}
    assert population - front, "every population change also changed the noisy front"


def test_a_reorder_alone_is_no_step_but_keeps_its_recorded_order():
    """[a, b, a, c] then [c, a, b, a]: the recorder keeps both orders (two history rows), the steps merge
    them, and the plotted points, ranks and multiplicities are identical; the eval-id order is each
    generation's own."""
    view, eid = lab_view("abac", "caba")
    assert view.population_history().starts.tolist() == [0, 1]
    assert population_steps(view).starts.tolist() == [0]
    orders = [view.population(g).eval_ids.tolist() for g in (0, 1)]
    assert orders == [[eid["a"], eid["b"], eid["a"], eid["c"]], [eid["c"], eid["a"], eid["b"], eid["a"]]]
    for basis in BASES:
        g0, g1 = rank_layer_data(view, 0, basis), rank_layer_data(view, 1, basis)
        assert plotted(g0) == plotted(g1), basis                              # the same drawing
        assert shown_members(g0) == shown_members(g1), basis                  # the same member multiset
        assert (g0["generation"], g1["generation"]) == (0, 1)                 # each shows its own generation
        assert g0["step"] == g1["step"] == 0 and g0["step_range"] == g1["step_range"] == [0, 1]
        # the recorded order differs, and each generation's points follow its own
        assert [p["eval_ids"] for p in g0["points"]] != [p["eval_ids"] for p in g1["points"]], basis
        for layers, order in zip((g0, g1), orders):
            for p in layers["points"]:
                assert p["eval_ids"] == [e for e in order if e in p["eval_ids"]], basis
    # noisy basis: a and c share rank 0, in the order each population first holds them
    g0, g1 = rank_layer_data(view, 0), rank_layer_data(view, 1)
    assert [p["eval_ids"] for p in g0["points"] if p["rank"] == 0] == [[eid["a"], eid["a"]], [eid["c"]]]
    assert [p["eval_ids"] for p in g1["points"] if p["rank"] == 0] == [[eid["c"]], [eid["a"], eid["a"]]]
    f0, f1 = plot_rank_layers(rank_layer_data(view, 0)), plot_rank_layers(rank_layer_data(view, 1))
    drawn = [sorted((t.name, x, y, s) for t in f.data for x, y, s in zip(t.x, t.y, t.marker.size)) for f in (f0, f1)]
    assert drawn[0] == drawn[1]
    assert "Gen 0<br>" in f0.layout.title.text and "Gen 1<br>" in f1.layout.title.text


# ------------------------------------------------------------------------------ layers

@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name,prob,long", RUNS)
def test_layers_are_the_population_at_the_exact_generation(name, prob, long, basis):
    """At every generation, read in increasing and then decreasing order (so later reads hit the snapshot
    cache from another generation of the same step): the generation and evaluation count are that
    generation's, the members are exactly its population (a multiset, under PopulationSnapshot's ranks and
    the basis's recorded values), each point keeps its members in recorded order, and equal vectors share
    one point and one rank."""
    view = view_of(name, prob, long)
    steps = population_steps(view)
    for g in list(range(view.n_generations)) + list(range(view.n_generations - 1, -1, -1)):
        snapshot = view.population(g)
        layers = rank_layer_data(view, g, basis)
        assert layers["generation"] == g and layers["n_evals"] == view.evals_through(g)
        assert layers["step"] == steps.step_of(g) and layers["n_steps"] == len(steps)
        assert layers["step_range"][0] <= g <= layers["step_range"][1]
        assert shown_members(layers) == snapshot_members(snapshot, basis), (g, basis)
        assert layers["n_members"] == layers["n_shown"] == len(snapshot) == sum(p["count"] for p in layers["points"])
        ranks = snapshot.rank(basis)
        assert layers["n_ranks"] == int(ranks.max()) + 1 and layers["max_rank_shown"] == layers["n_ranks"] - 1
        assert layers["rank_counts"] == np.bincount(ranks).tolist()
        values, other = values_of(snapshot, basis), values_of(snapshot, "true" if basis == "noisy" else "noisy")
        other_ranks = snapshot.rank("true" if basis == "noisy" else "noisy")
        assert len({(p["x"], p["y"]) for p in layers["points"]}) == len(layers["points"])
        for p in layers["points"]:
            rows = [i for i in range(len(snapshot)) if (float(values[i, 0]), float(values[i, 1])) == (p["x"], p["y"])]
            assert p["eval_ids"] == [int(snapshot.eval_ids[i]) for i in rows]              # recorded order
            assert {int(ranks[i]) for i in rows} == {p["rank"]} and p["count"] == len(rows)
            assert Counter({tuple(v): n for v, n in p["other_values"]}) == \
                Counter(tuple(float(v) for v in other[i]) for i in rows)
            assert Counter({r: n for r, n in p["other_ranks"]}) == Counter(int(other_ranks[i]) for i in rows)
        assert [p["rank"] for p in layers["points"]] == sorted(p["rank"] for p in layers["points"])
        extent = [[float(values[:, i].min()), float(values[:, i].max())] for i in range(2)]
        assert layers["axis_extent"] == extent


def test_noisy_and_true_layers_plot_the_recorded_values_exactly():
    """Noisy layers draw y, true layers f(x), bit for bit (prior noise: y and f(x) differ)."""
    view = view_of("NSGA2", "prior")
    for g in range(view.n_generations):
        snapshot = view.population(g)
        for basis, values in (("noisy", snapshot.observed), ("true", snapshot.true_obj)):
            drawn = Counter((p["x"], p["y"]) for p in rank_layer_data(view, g, basis)["points"] for _ in p["eval_ids"])
            assert drawn == Counter((float(v[0]), float(v[1])) for v in values), (g, basis)
    assert any(not np.array_equal(view.population(g).observed, view.population(g).true_obj)
               for g in range(view.n_generations))


def test_clones_twins_and_repeated_observations_are_all_kept():
    """The same individual twice (one eval_id twice), its twin (same observation, own eval_id), another y of
    the same genotype and a prior-noise x~: noisy layers keep five members at two points, true layers five at
    one; the multisets keep the repeated eval_id twice."""
    lab = Lab()
    a = lab.evaluate(A, (100, 50), (80.0, 50.0))
    twin = lab.evaluate(A, (100, 50), (80.0, 50.0))
    other_y = lab.evaluate(A, (100, 50), (70.0, 40.0))
    prior = lab.evaluate(A, (100, 50), (80.0, 50.0), evaluated=[1, 1, 0, 0])
    lab.observe([a, a, twin, other_y, prior])
    view = MORunView(lab.log.finalise())
    clear_active_logger()
    ids = [lab.log.resolve(i) for i in (a, a, twin, other_y, prior)]

    noisy = rank_layer_data(view, 0, "noisy")
    assert [(p["x"], p["y"], p["count"]) for p in noisy["points"]] == [(80.0, 50.0, 4), (70.0, 40.0, 1)]
    assert Counter(noisy["points"][0]["eval_ids"]) == Counter([ids[0], ids[0], ids[2], ids[4]])
    assert len(set(noisy["points"][0]["obs_ids"])) == 2 and len(set(noisy["points"][0]["geno_ids"])) == 1
    assert {p["rank"] for p in noisy["points"]} == {0}                        # incomparable under (max, min)
    true = rank_layer_data(view, 0, "true")
    [point] = true["points"]
    assert (point["x"], point["y"], point["count"]) == (100.0, 50.0, 5)
    assert Counter(point["eval_ids"]) == Counter(ids) and point["eval_ids"] == ids
    assert sorted(map(tuple, (v for v, _ in point["other_values"]))) == [(70.0, 40.0), (80.0, 50.0)]
    hover = plot_rank_layers(noisy).data[-1].text[0]
    assert "×4 members · 3 evaluations · 1 genotype · 2 observations" in hover


def _ranks_at_least(view, n, basis="noisy"):
    return [g for g in range(view.n_generations) if view.population(g).rank(basis).max() + 1 >= n]


@pytest.mark.parametrize("name,prob,long", [("MoUMDA", "kp1", False), ("NSGA2", "kp1", True), ("MoUMDA_KMeans", "prior", False)])
@pytest.mark.parametrize("basis", BASES)
def test_filtering_selects_ranks_without_reranking(name, prob, long, basis):
    """Showing ranks 0..k shows exactly in_ranks(basis, k), under the whole population's ranks; the shown
    points are the full layers' points of those ranks, unchanged, and every rank keeps its colour."""
    view = view_of(name, prob, long)
    deep = 0
    for g in range(view.n_generations):
        snapshot, full = view.population(g), rank_layer_data(view, g, basis)
        full_fig = plot_rank_layers(full)
        colours = {t.name.split(" (")[0]: t.marker.color for t in full_fig.data if t.name}
        deep = max(deep, full["n_ranks"])
        for k in range(full["n_ranks"]):
            layers = rank_layer_data(view, g, basis, max_rank=k)
            rows = snapshot.in_ranks(basis, k)
            assert shown_members(layers) == snapshot_members(snapshot, basis, rows), (g, k)
            assert layers["points"] == [p for p in full["points"] if p["rank"] <= k]
            assert layers["n_ranks"] == full["n_ranks"] and layers["max_rank_shown"] == k
            assert layers["n_shown"] == len(rows) and layers["n_members"] == len(snapshot)
            fig = plot_rank_layers(layers)
            assert {t.name.split(" (")[0]: t.marker.color for t in fig.data if t.name} == \
                {r: c for r, c in colours.items() if int(r.split()[1]) <= k}
            assert fig.layout.xaxis.range == full_fig.layout.xaxis.range      # the view does not jump
        assert rank_layer_data(view, g, basis, max_rank=10 ** 6)["points"] == full["points"]   # clamped
        assert rank_layer_data(view, g, basis, max_rank=-3)["max_rank_shown"] == 0
    if basis == "noisy":
        assert deep >= 3, "no generation with three noisy ranks: the filter was barely exercised"


def test_semo_populations_have_one_noisy_rank_and_disable_the_rank_filter():
    view = view_of("SEMO", "cocz1", True)
    df = pd.DataFrame([{"mo_record": record_of("SEMO", "cocz1", True)}])
    rows = [{"algo_name": "SEMO", "noise": 1.0, "_row": 0}]
    for g in range(view.n_generations):
        assert rank_layer_data(view, g, "noisy")["n_ranks"] == 1
    controls = rank_layer_controls(df, rows, 0, "noisy")
    assert controls["rank_disabled"] and controls["rank_max"] == controls["rank_value"] == 0
    assert controls["gen"] == view.n_generations - 1 and controls["step"] == len(population_steps(view)) - 1


def test_version_1_records_explain_and_existing_plots_still_work():
    record = version_1("NSGA2", "kp1")
    df = pd.DataFrame([{"mo_record": record}])
    view = mo_frames.row_view(df, 0)
    message = NO_POPULATION_MESSAGE.format(version=1)
    assert rank_layer_data(view, 0) == {"unavailable": message}
    rows = [{"algo_name": "NSGA2", "noise": 1.0, "_row": 0}]
    [option] = rank_layer_runs(df, rows)
    assert option["label"].endswith("(no population recorded)") and option["value"] == 0
    controls = rank_layer_controls(df, rows, None, "noisy")
    assert controls["nav_disabled"] and controls["rank_disabled"] and controls["status"] == message
    assert controls["run"] == 0 and controls["gen"] is None
    layers, label = rank_layer_selection(df, rows, 0, "noisy", 5, 1)
    fig = plot_rank_layers(layers, label)
    assert fig.layout.title.text == message and not fig.data
    groups, labels = pareto_runs(df, rows, lambda v: pareto_front_entries(v, "noisy_population"))
    for plot_type in FRONT_PLOTS:
        assert get_pareto_plot(plot_type)([groups[0] * 3], labels, IndVsDist_IndType="noisy_population").data


def test_unsupported_basis_pool_and_objective_counts_are_refused():
    view = view_of("NSGA2", "kp1")
    with pytest.raises(ValueError, match="rank basis"):
        rank_layer_data(view, 0, "clean")
    with pytest.raises(ValueError, match="rank pool"):
        rank_layer_data(view, 0, pool="evaluations")
    with pytest.raises(ValueError, match="rank pool"):
        rank_layer_controls(pd.DataFrame([{"mo_record": record_of("NSGA2", "kp1")}]), [], pool="evaluations")
    three = copy.copy(view)
    three.n_obj = 3
    assert rank_layer_data(three, 0) == {"unavailable": "Rank layers draw two objectives; this run has 3"}


# ------------------------------------------------------------------------------ controls

@functools.lru_cache(maxsize=None)
def control_frame():
    """Four rows of three series (listed by name: MoUMDA, NSGA2, SEMO): a v1 NSGA-II run first in its series."""
    records = [record_of("NSGA2", "kp1", True), record_of("SEMO", "cocz1", True), record_of("MoUMDA", "kp1"),
               version_1("NSGA2", "kp1", True)]
    df = pd.DataFrame([{"mo_record": r} for r in records])
    rows = [{"algo_name": "NSGA2", "noise": 1.0, "_row": 3}, {"algo_name": "NSGA2", "noise": 1.0, "_row": 0},
            {"algo_name": "SEMO", "noise": 2.0, "_row": 1}, {"algo_name": "MoUMDA", "noise": 1.0, "_row": 2}]
    return df, rows


def test_runs_are_listed_as_the_pareto_tab_groups_them():
    df, rows = control_frame()
    options = rank_layer_runs(df, rows)
    assert options == [{"label": "MoUMDA, noise=1.0, run 0", "value": 2},
                       {"label": "NSGA2, noise=1.0, run 0 (no population recorded)", "value": 3},
                       {"label": "NSGA2, noise=1.0, run 1", "value": 0},
                       {"label": "SEMO, noise=2.0, run 0", "value": 1}]
    groups, _ = pareto_runs(df, rows, lambda view: [view])
    assert [v for runs in groups for [v] in runs] == [mo_frames.row_view(df, o["value"]) for o in options]
    assert rank_layer_runs(df, []) == [] and rank_layer_runs(df, None) == []


def test_controls_navigate_steps_and_exact_generations():
    df, rows = control_frame()
    view = mo_frames.row_view(df, 1)                                          # SEMO: long steps
    steps, last = population_steps(view), view.n_generations - 1
    c = rank_layer_controls(df, rows, 1, "noisy")                             # the initial call
    assert (c["run"], c["gen"], c["step"], c["step_max"], c["gen_max"]) == (1, last, len(steps) - 1, len(steps) - 1, last)
    assert not c["nav_disabled"] and c["status"].startswith(f"Generation {last} of {last} ·")
    assert f"step {len(steps)} of {len(steps)}" in c["status"]
    assert c["step_marks"][0] == "G0" and c["step_marks"][len(steps) - 1] == f"G{steps.starts[-1]}"
    assert len(c["step_marks"]) <= 8
    k = next(i for i in range(len(steps)) if steps.ends[i] - steps.starts[i] >= 2)
    s, e = int(steps.starts[k]), int(steps.ends[k])
    # the step slider goes to the step's first generation
    c = rank_layer_controls(df, rows, 1, "noisy", step=k, gen=0, trigger="step")
    assert (c["step"], c["gen"]) == (k, s)
    assert f"population unchanged over generations {s}–{e}" in c["status"]
    # an exact generation inside the step stays exact and selects that step
    c = rank_layer_controls(df, rows, 1, "noisy", step=0, gen=s + 1, trigger="generation")
    assert (c["step"], c["gen"]) == (k, s + 1) and c["status"].startswith(f"Generation {s + 1} of {last}")
    for typed, expected in ((-5, 0), (10 ** 6, last), ("12", 12), (3.0, 3)):
        c = rank_layer_controls(df, rows, 1, "noisy", step=0, gen=typed, trigger="generation")
        assert (c["gen"], c["step"]) == (expected, steps.step_of(expected)), typed
    for invalid in (None, "", "abc"):                                         # the slider's step stays
        c = rank_layer_controls(df, rows, 1, "noisy", step=k, gen=invalid, trigger="generation")
        assert (c["gen"], c["step"]) == (s, k), invalid
    assert rank_layer_controls(df, rows, 1, "noisy", step=k, gen=s + 1, trigger="first")["gen"] == 0
    assert rank_layer_controls(df, rows, 1, "noisy", step=0, gen=0, trigger="last")["gen"] == last
    # a basis change keeps the exact generation
    c = rank_layer_controls(df, rows, 1, "true", step=k, gen=s + 1, trigger="basis")
    assert (c["gen"], c["step"]) == (s + 1, k) and "(true objectives)" in c["status"]
    assert rank_layer_controls(df, rows, 1, "clean", gen=s + 1, trigger="basis")["status"].endswith("(noisy objectives)")


def test_switching_or_losing_the_run_resets_to_its_final_population():
    df, rows = control_frame()
    nsga = mo_frames.row_view(df, 0)
    c = rank_layer_controls(df, rows, 0, "noisy", step=2, gen=7, trigger="run")
    assert (c["run"], c["gen"], c["step"]) == (0, nsga.n_generations - 1, len(population_steps(nsga)) - 1)
    # a run that is no longer selected falls back to the first, even when another input triggered
    for trigger in ("generation", "step", "basis", None):
        c = rank_layer_controls(df, rows[2:], 0, "noisy", step=2, gen=7, trigger=trigger)
        moumda = mo_frames.row_view(df, 2)
        assert (c["run"], c["gen"]) == (2, moumda.n_generations - 1), trigger
    c = rank_layer_controls(df, [], 0, "noisy", step=2, gen=7, trigger="run")
    assert c["run"] is None and c["run_options"] == [] and c["nav_disabled"] and c["status"] == NO_RUNS_MESSAGE
    # a stale generation from a longer run is clamped, never out of range
    c = rank_layer_controls(df, rows, 2, "noisy", step=30, gen=10 ** 6, trigger="generation")
    assert c["gen"] == mo_frames.row_view(df, 2).n_generations - 1


def expected_rank_value(prev_value, prev_max, n_ranks):
    """The documented rule: "all" stays all; an explicit highest rank is clamped to the new range."""
    if prev_value is None or prev_max is None or prev_value >= prev_max:
        return n_ranks - 1
    return min(prev_value, n_ranks - 1)


def test_rank_range_follows_the_population_and_keeps_all_or_clamps():
    df, rows = control_frame()
    view = mo_frames.row_view(df, 2)                                          # MoUMDA: several noisy ranks
    counts = [int(view.population(g).noisy_rank.max()) + 1 for g in range(view.n_generations)]
    deep = int(np.argmax(counts))
    assert counts[deep] >= 4
    for g in range(view.n_generations):
        for prev_value, prev_max in ((None, None), (0, 0), (counts[deep] - 1, counts[deep] - 1), (1, 5), (2, 9), (7, 9)):
            c = rank_layer_controls(df, rows, 2, "noisy", gen=g, trigger="generation",
                                    prev_max_value=prev_value, prev_max=prev_max)
            assert c["rank_max"] == counts[g] - 1 and c["rank_disabled"] == (counts[g] < 2)
            assert c["rank_value"] == expected_rank_value(prev_value, prev_max, counts[g]), (g, prev_value, prev_max)
            assert set(c["rank_marks"]) <= set(range(counts[g])) and 0 in c["rank_marks"]
            # the shown ranks are the whole population's ranks up to the value, never re-ranked
            layers = rank_layer_data(view, g, "noisy", c["rank_value"])
            assert {p["rank"] for p in layers["points"]} == set(range(c["rank_value"] + 1))


def test_trigger_priority_when_several_inputs_change_together():
    assert rank_layer_trigger(None) is None and rank_layer_trigger({}) is None
    assert rank_layer_trigger({"paretoRankRun.value": "paretoRankRun",
                               "paretoRankGeneration.value": "paretoRankGeneration"}) == "run"
    assert rank_layer_trigger(["STN_data.data", "paretoRankStep.value"]) == "run"
    assert rank_layer_trigger(["paretoRankGeneration.value", "paretoRankLast.n_clicks"]) == "last"
    assert rank_layer_trigger(["paretoRankFirst.n_clicks", "paretoRankGeneration.value"]) == "first"
    assert rank_layer_trigger(["paretoRankStep.value", "paretoRankGeneration.value"]) == "generation"
    assert rank_layer_trigger(["paretoRankBasis.value", "paretoRankStep.value"]) == "step"
    assert rank_layer_trigger(["paretoRankBasis.value"]) == "basis"


# ------------------------------------------------------------------------------ the figure

def viridis(t):
    return sample_colorscale(RANK_COLORSCALE, [t])[0]


def test_rank_colours_for_one_two_and_many_ranks():
    """Rank 0 has its own colour, outside the scale. One rank: only that. Two ranks: rank 1 alone on the
    scale, whose bounds coincide, takes the midpoint (no division by zero). Many: ranks 1..R-1 span the
    scale end to end, each distinct."""
    assert rank_color(0, 1) == rank_color(0, 2) == rank_color(0, 50) == RANK0_COLOR
    assert rank_color(1, 2) == viridis(0.5)
    assert rank_color(1, 1) == viridis(0.5)                                   # degenerate input: still defined
    many = [rank_color(r, 20) for r in range(1, 20)]
    assert many[0] == viridis(0.0) and many[-1] == viridis(1.0) and len(set(many)) == 19
    assert RANK0_COLOR not in set(many) | {viridis(0.5)}
    assert rank_color(2, 3) == viridis(1.0) and rank_color(1, 3) == viridis(0.0)


def synthetic_layers(n_ranks, per_rank=1, counts=None, weights=(1.0, -1.0), basis="noisy"):
    """rank_layer_data-shaped layers: per_rank points per rank, rank r's points at x = 10 - r + j/10."""
    points = []
    for r in range(n_ranks):
        for j in range(per_rank):
            count = (counts or {}).get((r, j), 1)
            points.append({"rank": r, "x": 10.0 - r + j / 10, "y": 1.0 + r + j / 10, "count": count,
                           "eval_ids": list(range(count)), "geno_ids": [7] * count, "obs_ids": list(range(count)),
                           "other_values": [[[1.0, 2.0], count]], "other_ranks": [[0, count]]})
    xs, ys = [p["x"] for p in points], [p["y"] for p in points]
    return {"basis": basis, "pool": "population", "generation": 4, "n_generations": 9, "n_evals": 40, "step": 2,
            "n_steps": 5, "step_range": [3, 5], "weights": list(weights), "n_members": sum(p["count"] for p in points),
            "n_shown": sum(p["count"] for p in points), "n_ranks": n_ranks, "max_rank_shown": n_ranks - 1,
            "rank_counts": [per_rank] * n_ranks, "axis_extent": [[min(xs), max(xs)], [min(ys), max(ys)]],
            "points": points}


@pytest.mark.parametrize("n_ranks", [1, 2, 5, LEGEND_MAX_RANKS, LEGEND_MAX_RANKS + 2])
def test_rank_0_is_drawn_last_and_listed_first(n_ranks):
    fig = plot_rank_layers(synthetic_layers(n_ranks, per_rank=2), "S, run 0")
    markers = [t for t in fig.data if t.name]
    assert [t.name.split(" (")[0] for t in markers] == [f"Rank {r}" for r in range(n_ranks - 1, -1, -1)]
    assert [t.legendrank for t in markers] == list(range(n_ranks - 1, -1, -1))
    assert fig.data[-1 if n_ranks <= LEGEND_MAX_RANKS else -2].name.startswith("Rank 0 ")   # on top
    assert [t.marker.color for t in markers] == [rank_color(r, n_ranks) for r in range(n_ranks - 1, -1, -1)]
    shown = [int(t.name.split()[1]) for t in markers if t.showlegend]
    colourbars = [t for t in fig.data if not t.name and t.marker.showscale]
    if n_ranks <= LEGEND_MAX_RANKS:
        assert sorted(shown) == list(range(n_ranks)) and not colourbars
    else:
        assert sorted(shown) == list(range(LEGEND_MAX_RANKS))
        [bar] = colourbars
        assert (bar.marker.cmin, bar.marker.cmax, bar.marker.colorbar.title.text) == (1, n_ranks - 1, "Rank")
    assert fig.layout.legend.title.text == "Rank (noisy)"


def test_marker_sizes_show_rank_0_and_multiplicity():
    layers = synthetic_layers(3, per_rank=2, counts={(0, 1): 4, (1, 1): 4, (2, 1): 400})
    fig = plot_rank_layers(layers)
    sizes = {t.name.split(" (")[0]: list(t.marker.size) for t in fig.data if t.name}
    assert sizes["Rank 0"][0] > sizes["Rank 1"][0] == sizes["Rank 2"][0]
    assert sizes["Rank 0"][1] > sizes["Rank 0"][0] and sizes["Rank 1"][1] > sizes["Rank 1"][0]
    assert sizes["Rank 2"][1] == MAX_MARKER_SIZE
    # the plotted coordinates are the recorded ones, never moved apart
    for trace in (t for t in fig.data if t.name):
        rank = int(trace.name.split()[1])
        assert list(zip(trace.x, trace.y)) == [(p["x"], p["y"]) for p in layers["points"] if p["rank"] == rank]


def test_hover_names_rank_values_multiplicity_and_truncates_long_id_lists():
    layers = synthetic_layers(2, counts={(0, 0): HOVER_ID_LIMIT + 5})
    [rank0] = [t for t in plot_rank_layers(layers).data if t.name and t.name.startswith("Rank 0 ")]
    hover = rank0.text[0]
    assert rank0.hovertemplate == "%{text}<extra></extra>"
    assert "<b>Rank 0</b> (noisy objectives)" in hover and "noisy y: (10, 1)" in hover
    assert "true f(x): (1, 2) ×15" in hover and "true ranks: 0 ×15" in hover
    assert "×15 members · 15 evaluations · 1 genotype · 15 observations" in hover
    assert f"eval ids: {', '.join(map(str, range(HOVER_ID_LIMIT)))} … and 5 more" in hover
    assert rank0.name == "Rank 0 (15 members)"


def test_axes_name_the_space_and_each_objective_direction():
    fig = plot_rank_layers(synthetic_layers(2, weights=(1.0, -1.0)))
    assert fig.layout.xaxis.title.text == "Objective 1 (noisy y, maximised)"
    assert fig.layout.yaxis.title.text == "Objective 2 (noisy y, minimised)"
    fig = plot_rank_layers(synthetic_layers(2, weights=(-1.0, 1.0), basis="true"), "S, run 3")
    assert fig.layout.xaxis.title.text == "Objective 1 (true f(x), minimised)"
    assert fig.layout.yaxis.title.text == "Objective 2 (true f(x), maximised)"
    assert fig.layout.title.text.startswith("Pareto rank layers (true objectives) - S, run 3, Gen 4<br>")
    assert "Step 3 of 5 · population unchanged over gens 3–5" in fig.layout.title.text
    assert fig.layout.xaxis.range[0] < 9.0 and fig.layout.xaxis.range[1] > 10.0


def test_staircases_only_when_asked_and_beneath_the_markers():
    layers = synthetic_layers(3, per_rank=3)
    assert all(t.mode == "markers" for t in plot_rank_layers(layers).data)
    fig = plot_rank_layers(layers, staircase=True)
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == 3 and fig.data[:3] == tuple(lines)                   # drawn first, beneath
    for rank, line in zip((2, 1, 0), lines):
        pts = [(p["x"], p["y"]) for p in layers["points"] if p["rank"] == rank]
        xs, ys, shape = front_line(pts, layers["weights"], attainment=True)
        assert (list(line.x), list(line.y), line.line.shape) == (xs, ys, shape)
        assert line.line.color == rank_color(rank, 3) and line.hoverinfo == "skip" and not line.showlegend
    single = plot_rank_layers(synthetic_layers(2, per_rank=1), staircase=True)
    assert not [t for t in single.data if t.mode == "lines"]                  # one point is no staircase


def test_unavailable_and_empty_layers_are_titled_empty_figures():
    fig = plot_rank_layers({"unavailable": "Population not recorded"})
    assert fig.layout.title.text == "Population not recorded" and not fig.data
    empty = dict(synthetic_layers(1), points=[], n_members=0, n_shown=0, n_ranks=0, axis_extent=None)
    assert plot_rank_layers(empty).layout.title.text == "The population at generation 4 is empty"


def test_real_runs_plot_for_every_algorithm():
    for name in ALGORITHM_NAMES:
        for prob in PROBLEMS:
            view = view_of(name, prob)
            for basis in BASES:
                layers = rank_layer_data(view, None, basis)
                fig = plot_rank_layers(layers, name)
                assert layers["generation"] == view.n_generations - 1
                assert sum(len(t.x) for t in fig.data if t.name) == len(layers["points"]), (name, prob, basis)


# ------------------------------------------------------------------------------ the callbacks (exec'd)

def _pareto_functions(names, namespace):
    """Functions and assignments of callbacks/pareto.py exec'd into namespace (importing the module would
    load the dashboard data), and each function's decorator."""
    tree = ast.parse(open(PARETO_PY).read())
    body, decorators = [], {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            if node.decorator_list:
                decorators[node.name] = node.decorator_list[0]
            node.decorator_list = []
            body.append(node)
        elif isinstance(node, ast.Assign) and [t.id for t in node.targets][0] in names:
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), PARETO_PY, "exec"), namespace)
    return decorators


class _Ctx:
    triggered_prop_ids = {}


NO_UPDATE = object()


@pytest.fixture(scope="module")
def callbacks():
    df, rows = control_frame()
    ctx = _Ctx()
    namespace = {"go": go, "df": df, "PARETO_NOISY_PAIRS": PARETO_NOISY_PAIRS, "RANK_LAYERS_PLOT": RANK_LAYERS_PLOT,
                 "pareto_front_entries": pareto_front_entries, "pareto_pair_entries": pareto_pair_entries,
                 "pareto_runs": pareto_runs, "get_pareto_plot": get_pareto_plot, "FRONT_KEYS": FRONT_KEYS,
                 "UNSUPPORTED_METRIC_TITLE": UNSUPPORTED_METRIC_TITLE, "create_empty_figure": create_empty_figure,
                 "selected_front": selected_front, "FRONT_PLOTS": FRONT_PLOTS, "GRIDLINE_PLOTS": ("Basic",),
                 "get_knapsack_problem_stats": get_knapsack_problem_stats, "rank_layer_selection": rank_layer_selection,
                 "rank_layer_controls": rank_layer_controls, "rank_layer_trigger": rank_layer_trigger,
                 "ctx": ctx, "no_update": NO_UPDATE}
    decorators = _pareto_functions({"updateParetoPlot", "toggle_pareto_plot_options", "update_rank_layer_controls",
                                    "RANK_CONTROL_OUTPUTS", "knapsack_gridlines"}, namespace)
    return namespace, decorators, ctx, df, rows


def _ids(decorator_arg):
    elements = decorator_arg.elts if isinstance(decorator_arg, ast.List) else [decorator_arg]
    return [(e.args[0].value, e.args[1].value) for e in elements]


def test_callback_wiring_is_pinned(callbacks):
    namespace, decorators, *_ = callbacks
    plot = decorators["updateParetoPlot"]
    assert _ids(plot.args[1])[-5:] == [("paretoRankRun", "value"), ("paretoRankBasis", "value"),
                                       ("paretoRankGeneration", "value"), ("paretoRankMax", "value"),
                                       ("paretoRankStaircase", "value")]
    toggle = decorators["toggle_pareto_plot_options"]
    assert _ids(toggle.args[0]) == [("paretoFrontOptions", "style"), ("paretoRankLayerOptions", "style")]
    assert _ids(toggle.args[1]) == [("paretoFrontPlotType", "value")]
    controls = decorators["update_rank_layer_controls"]
    assert _ids(controls.args[0]) == [(c, p) for c, p, _ in namespace["RANK_CONTROL_OUTPUTS"]]
    assert _ids(controls.args[1]) == [("STN_data", "data"), ("paretoRankRun", "value"), ("paretoRankBasis", "value"),
                                      ("paretoRankStep", "value"), ("paretoRankGeneration", "value"),
                                      ("paretoRankFirst", "n_clicks"), ("paretoRankLast", "n_clicks")]
    assert _ids(controls.args[2]) == [("paretoRankMax", "value"), ("paretoRankMax", "max")]
    # every changing input is a trigger rank_layer_trigger knows
    assert {c for c, _ in _ids(controls.args[1])} == {c for c, _ in mo_frames.RANK_TRIGGERS}


def test_toggle_shows_rank_controls_only_for_rank_layers(callbacks):
    namespace, *_ = callbacks
    toggle = namespace["toggle_pareto_plot_options"]
    assert toggle("RankLayers") == ({"display": "none"}, {})
    for plot_type in ("Basic", "Subplots", "paretoAnimation", "NoisyPopulationVsClean", "IndVsDist", "Scatter", None):
        assert toggle(plot_type) == ({}, {"display": "none"}), plot_type


def test_plot_callback_draws_the_selected_run_and_leaves_other_types_unchanged(callbacks):
    namespace, _, _, df, rows = callbacks
    update = namespace["updateParetoPlot"]
    for run, basis, gen, top, stairs in ((1, "noisy", 17, None, []), (0, "true", 3, 0, ["on"]), (2, "noisy", None, 1, []),
                                         (3, "noisy", 2, 0, []), (99, None, 4, None, [])):
        fig = update(rows, "RankLayers", "noisy_population", "cumulative", 1, 5, ["on"], [], None,
                     run, basis, gen, top, stairs)
        layers, label = rank_layer_selection(df, rows, run, basis, gen, top)
        expected = plot_rank_layers(layers, label, staircase=bool(stairs), uirevision=f"{run}-{layers.get('basis')}")
        assert fig.to_json() == expected.to_json(), (run, basis, gen)
    assert update([], "RankLayers", None, None, 1, 5, [], [], None).layout.title.text == NO_RUNS_MESSAGE
    # the existing plot types ignore the rank-layer inputs
    for plot_type in FRONT_PLOTS + ("NoisyPopulationVsClean", "IndVsDist", "MoveCorr", "Scatter"):
        before = update(rows, plot_type, "noisy_population", "cumulative", 2, 5, ["on"]).to_json()
        after = update(rows, plot_type, "noisy_population", "cumulative", 2, 5, ["on"], [], None,
                       1, "true", 3, 0, ["on"]).to_json()
        assert before == after, plot_type


def test_controls_callback_sends_only_changed_navigation_values(callbacks):
    namespace, _, ctx, df, rows = callbacks
    update = namespace["update_rank_layer_controls"]
    keys = [(c, p) for c, p, _ in namespace["RANK_CONTROL_OUTPUTS"]]
    view = mo_frames.row_view(df, 1)
    steps = population_steps(view)
    k = next(i for i in range(len(steps)) if steps.ends[i] - steps.starts[i] >= 2)
    s = int(steps.starts[k])

    ctx.triggered_prop_ids = {"paretoRankGeneration.value": "paretoRankGeneration"}
    out = dict(zip(keys, update(rows, 1, "noisy", 0, s + 1, 0, 0, 0, 0)))
    assert out[("paretoRankGeneration", "value")] is NO_UPDATE               # the typed value stays
    assert out[("paretoRankStep", "value")] == k and out[("paretoRankRun", "value")] is NO_UPDATE
    ctx.triggered_prop_ids = {"paretoRankStep.value": "paretoRankStep"}
    out = dict(zip(keys, update(rows, 1, "noisy", k, s + 1, 0, 0, 0, 0)))
    assert out[("paretoRankStep", "value")] is NO_UPDATE and out[("paretoRankGeneration", "value")] == s
    ctx.triggered_prop_ids = {"STN_data.data": "STN_data"}
    out = dict(zip(keys, update(rows[2:], 0, "noisy", k, s + 1, 0, 0, 0, 0)))
    assert out[("paretoRankRun", "value")] == 2 and out[("paretoRankRun", "options")] == rank_layer_runs(df, rows[2:])
    ctx.triggered_prop_ids = {}


# ------------------------------------------------------------------------------ the real Dash callback chain

@pytest.fixture(scope="module")
def chain():
    """harness/rank_layer_chain.py in a fresh temp root with the write fence: the real app, over HTTP."""
    root = make_temp_root()
    env = child_env(root)
    env["PYTHONPATH"] = f"{HARNESS_DIR.parent.parent / 'src'}:{HARNESS_DIR.parent}"
    try:
        done = subprocess.run([sys.executable, str(HARNESS_DIR / "rank_layer_chain.py")], cwd=str(root), env=env,
                              capture_output=True, text=True, timeout=1800)
        assert done.returncode == 0, done.stderr[-4000:]
        report = json.loads(done.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    report["by_name"] = {e["name"]: e for e in report["events"]}
    return report


def _step_of(starts, gen):
    return max(i for i, s in enumerate(starts) if s <= gen)


def _series(report, row):
    return next(series for series, r in report["rows"].items() if r == row)


def test_chain_every_request_succeeds_and_figure_matches_the_controls(chain):
    """Throughout the chain: every callback answers 200, the figure shows the box's exact generation and
    step, ranks 0..the slider's value of the whole population (rank 0 drawn last), and the rank range
    follows the documented rule from the previous controls."""
    previous = None
    for event in chain["events"]:
        name, c, fig = event["name"], event["controls"], event["figure"]
        assert event.get("controls_status", 200) == 200 and event["plot_status"] == 200, (name, event)
        assert event.get("stn_status", 200) == 200, name
        run, gen = c["paretoRankRun.value"], c["paretoRankGeneration.value"]
        if name == "basic_plot":
            continue
        truth = chain["truth"][_series(chain, run)] if run is not None else None
        if truth is None or truth["version"] == 1:
            assert gen is None and c["paretoRankGeneration.disabled"] and c["paretoRankMax.disabled"], name
            assert fig["ranks"] == [] and fig["gen"] is None, name
            previous = c
            continue
        basis = c["paretoRankBasis.value"]
        n_ranks = truth[f"{basis}_ranks"][gen]
        assert fig["gen"] == gen, (name, fig["title"])
        assert c["paretoRankStep.value"] == _step_of(truth["starts"], gen), name
        assert c["paretoRankStep.max"] == len(truth["starts"]) - 1 and c["paretoRankGeneration.max"] == truth["n_generations"] - 1
        assert c["paretoRankMax.max"] == n_ranks - 1 and c["paretoRankMax.disabled"] == (n_ranks < 2), name
        assert fig["ranks"] == list(range(c["paretoRankMax.value"] + 1)), (name, fig)
        assert fig["marker_order"] == sorted(fig["marker_order"], reverse=True), name
        assert f"({basis} objectives)" in fig["title"], name
        status = c["paretoRankStepLabel.children"]
        assert status.startswith(f"Generation {gen} of {truth['n_generations'] - 1} ·"), (name, status)
        assert f"step {c['paretoRankStep.value'] + 1} of {len(truth['starts'])}" in status, name
        if event.get("controls_status") and previous is not None and previous["paretoRankRun.value"] is not None:
            assert c["paretoRankMax.value"] == expected_rank_value(previous["paretoRankMax.value"],
                                                                   previous["paretoRankMax.max"], n_ranks), name
        previous = c


def test_chain_navigation_follows_each_user_action(chain):
    e, rows, truth = chain["by_name"], chain["rows"], chain["truth"]
    first, semo, nsga = truth["MoUMDA"], truth["SEMO"], truth["NSGA2"]
    last = first["n_generations"] - 1
    gen = lambda name: e[name]["controls"]["paretoRankGeneration.value"]
    step = lambda name: e[name]["controls"]["paretoRankStep.value"]
    run = lambda name: e[name]["controls"]["paretoRankRun.value"]

    assert chain["toggle"]["default"] == chain["toggle"]["basic"] == \
        {"paretoFrontOptions.style": {}, "paretoRankLayerOptions.style": {"display": "none"}}
    assert chain["toggle"]["rank_layers"] == {"paretoFrontOptions.style": {"display": "none"},
                                              "paretoRankLayerOptions.style": {}}
    assert e["initial"]["figure"]["title"] == NO_RUNS_MESSAGE and run("initial") is None
    # selecting rows: the first run (by series name), at its final population, every rank shown
    assert run("select") == rows["MoUMDA"] and (gen("select"), step("select")) == (last, len(first["starts"]) - 1)
    assert [o["value"] for o in e["select"]["run_options"]] == [rows["MoUMDA"], rows["NSGA2"], rows["NSGA2-v1"], rows["SEMO"]]
    assert e["select"]["controls"]["paretoRankMax.value"] == first["noisy_ranks"][last] - 1
    # the step slider moves the generation box to the step's first generation (and is not echoed back)
    assert gen("step") == first["starts"][3] and "paretoRankStep.value" not in e["step"]["returned"]
    # an exact generation moves the slider (and is not echoed back); out of range is clamped
    assert (gen("generation"), step("generation")) == (7, _step_of(first["starts"], 7))
    assert "paretoRankGeneration.value" not in e["generation"]["returned"]
    assert gen("generation_out_of_range") == last and gen("generation_negative") == 0
    assert gen("final") == last and (gen("first"), step("first")) == (0, 0)
    # an explicit highest rank reaches the plot only; a basis change keeps the generation
    deep = chain["deep_generation"]
    assert deep is not None and e["rank_max_1"]["figure"]["ranks"] == [0, 1] and "returned" not in e["rank_max_1"]
    assert gen("basis_true") == gen("basis_noisy") == deep
    assert e["rank_max_all"]["figure"]["ranks"] == list(range(first["noisy_ranks"][deep]))
    # switching run: its final population; SEMO has a single noisy rank, so the filter is disabled
    assert run("run_semo") == rows["SEMO"] and gen("run_semo") == semo["n_generations"] - 1
    assert e["run_semo"]["controls"]["paretoRankMax.disabled"]
    # an exact generation inside a long step is shown as typed, not as the step's start
    mid = chain["semo_mid_generation"]
    assert mid is not None and gen("semo_mid_generation") == mid != semo["starts"][step("semo_mid_generation")]
    assert e["semo_mid_generation"]["figure"]["gen"] == mid
    # several inputs in one request: run beats generation, generation beats step, step beats basis
    assert run("run_and_generation") == rows["NSGA2"] and gen("run_and_generation") == nsga["n_generations"] - 1
    assert (gen("step_and_generation"), step("step_and_generation")) == (9, _step_of(nsga["starts"], 9))
    assert gen("basis_and_step") == nsga["starts"][2]
    # the selected run leaves the selection: the first remaining run, at its final population
    assert run("deselect_nsga2") == rows["MoUMDA"] and gen("deselect_nsga2") == last
    # a version-1 run explains itself; the controls are disabled
    for name in ("run_v1", "v1_only"):
        assert run(name) == rows["NSGA2-v1"], name
        assert e[name]["figure"]["title"] == NO_POPULATION_MESSAGE.format(version=1), name
        assert e[name]["controls"]["paretoRankStepLabel.children"] == NO_POPULATION_MESSAGE.format(version=1)
    assert e["v1_only"]["run_options"] == [{"label": "NSGA2-v1, noise=1.0, run 0 (no population recorded)",
                                            "value": rows["NSGA2-v1"]}]
    assert run("empty_selection") is None and e["empty_selection"]["figure"]["title"] == NO_RUNS_MESSAGE


def test_chain_existing_plot_types_still_render(chain):
    assert chain["by_name"]["basic_plot"]["figure"]["title"].startswith("noisy population PF - ['NSGA2', 1.0], Run 0")
    counts = chain["basic_trace_counts"]
    assert {status for status, _ in counts.values()} == {200}
    for plot_type in ("Basic", "paretoAnimation", "NoisyPopulationVsClean", "IndVsDist", "Scatter"):
        assert counts[plot_type][1] > 0, plot_type
