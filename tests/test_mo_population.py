"""Recorded MO populations and their Pareto ranks (mo_record version 2).

    recording           observe_generation keeps every population member as the eval_id of the evaluation
                        whose y it carries, in the population's order with duplicates, as a change-point
                        history (a reorder is a change; an unchanged population adds nothing)
    ranks               common.pareto.nondominated_ranks: DEAP's sortLogNondominated (default) and
                        sortNondominated give the same 0-based ranks, rank 0 is exactly nondominated_mask,
                        duplicates rank alike, and no creator class, global state or RNG is touched
    view                MORunView.population / population_history reproduce the runtime population;
                        PopulationSnapshot ranks are computed on demand over all members
    rank 0 and HV       population ranks reproduce the persisted fronts and hypervolumes of every
                        generation: the noisy front (observations, under y), the clean front (genotypes,
                        under f(x)) and the corrected noisy front (noisy rank-0 genotypes re-ranked under
                        f(x)); the corrected front is not the clean front, and the tests never equate them
    algorithms          every MO algorithm records the population it actually holds: NSGA-II survivors
                        and unevaluated clones, MoUMDA's refreshed populations, SEMO's archive

Long runs (SEMO's one evaluation per generation) are checked at every population change point and a sample
of the unchanged generations around them; shorter runs at every generation. Runs in-process, writes nothing.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import pytest

from deap import creator, tools

from harness.mo_runs import A, ALGORITHM_NAMES, B, C, KP_WEIGHTS, Lab, build, rng_state, step_run
from noisyvis.common.pareto import hypervolume_of, nondominated_mask, nondominated_ranks
from noisyvis.dashboard.mo_frames import _front_rows
from noisyvis.results.mo_view import MORecordError, MORunView, PopulationHistory, PopulationSnapshot
from noisyvis.tracking.logger import clear_active_logger, get_active_logger
from noisyvis.tracking.mo_logger import MOEvaluationLogger, ProvenanceError

PROBLEMS = ["kp0", "kp1", "kpviol1", "cocz1", "prior"]
D, E = [0, 0, 0, 1], [1, 1, 0, 0]
EXHAUSTIVE_UP_TO = 200  # runs with more generations are checked at change points plus a sample
HV_RTOL = 1e-12         # only for HVs of a different point multiset than the persisted one


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


def long_run(name, prob):
    """A run long enough for SEMO to leave its population unchanged over many generations."""
    algo = build(name, prob=prob, gen_limit=2000 if name == "SEMO" else 40, stop_without_improvement_in_gens=None)
    step_run(algo)
    return algo, MORunView(algo.eval_log.finalise())


def generations_to_check(view, seed=0):
    """Every generation of a short run; otherwise generation 0, the last one, every population change point,
    the generation before each, and a seeded sample of 50 others."""
    n = view.n_generations
    if n <= EXHAUSTIVE_UP_TO:
        return list(range(n))
    starts = [int(s) for s in view.population_history().starts]
    sample = random.Random(seed).sample(range(n), 50)
    return sorted({0, n - 1, *starts, *(s - 1 for s in starts if s), *sample})


def peel_ranks(values, weights):
    """Test-only reference: repeatedly remove the non-dominated rows (nondominated_mask)."""
    w = np.asarray(values, dtype=float) * np.asarray(weights, dtype=float)
    ranks, left, k = np.full(len(w), -1), np.arange(len(w)), 0
    while len(left):
        keep = nondominated_mask(w[left])
        ranks[left[keep]] = k
        left, k = left[~keep], k + 1
    return ranks


# ------------------------------------------------------------------------------ the ranking adapter

def test_constructed_ranks_with_ties_and_duplicates():
    points = [(4, 1), (1, 4), (3, 3), (2, 2), (2, 2), (1, 1), (3, 3)]
    expected = [0, 0, 0, 1, 1, 2, 0]
    for nd in ("log", "standard"):
        assert nondominated_ranks(points, (1.0, 1.0), nd=nd).tolist() == expected
        # minimising the second objective: (4, 1) dominates all; then (3, 3), (2, 2), (1, 1) trade off;
        # (1, 4) is dominated by (1, 1)
        assert nondominated_ranks(points, (1.0, -1.0), nd=nd).tolist() == [0, 2, 1, 1, 1, 1, 1]
    assert nondominated_ranks(np.zeros((0, 2)), (1.0, 1.0)).tolist() == []
    assert nondominated_ranks([(5.0, 5.0)], (1.0, -1.0)).tolist() == [0]
    with pytest.raises(ValueError, match="nd must be"):
        nondominated_ranks(points, (1.0, 1.0), nd="other")
    with pytest.raises(ValueError, match=r"\(N, M\)"):
        nondominated_ranks([1.0, 2.0], (1.0, 1.0))


def test_both_deap_sorts_agree_with_each_other_and_with_the_mask():
    """Seeded random sets (2-4 objectives, mixed directions, continuous or tied values, explicit duplicate
    rows): sortLogNondominated == sortNondominated == peeling with nondominated_mask, and rank 0 is exactly
    the mask the logger uses for its fronts."""
    rng = np.random.default_rng(20261021)
    for trial in range(200):
        n, m = int(rng.integers(1, 70)), int(rng.integers(2, 5))
        values = rng.integers(0, 6, (n, m)).astype(float) if trial % 2 else rng.random((n, m))
        if trial % 3 == 0:
            values = np.vstack([values, values[: n // 2]])
        weights = tuple(float(w) for w in rng.choice([-1.0, 1.0], m))
        log, standard = nondominated_ranks(values, weights), nondominated_ranks(values, weights, nd="standard")
        assert np.array_equal(log, standard), trial
        assert np.array_equal(log, peel_ranks(values, weights)), trial
        assert np.array_equal(log == 0, nondominated_mask(values * np.asarray(weights))), trial


def test_ranking_registers_no_creator_class_and_draws_no_rng():
    build("NSGA2", prob="kp1")  # the algorithms' own creator classes exist
    before_creator, individual = dict(vars(creator)), creator.Individual
    random.seed(4)
    np.random.seed(4)
    state = rng_state()
    nondominated_ranks(np.random.default_rng(1).random((50, 3)), (1.0, -1.0, 1.0))
    assert rng_state() == state
    assert dict(vars(creator)) == before_creator and creator.Individual is individual


# ------------------------------------------------------------------------------ recording (exact cases)

def test_population_keeps_order_duplicates_and_distinct_observations():
    """The same object twice, an exact twin (same obs_id, own eval_id), another y of the same genotype and
    a prior-noise x~: five members, five rows, in order. An identical list adds no row; a reorder does."""
    lab = Lab()
    log = lab.log
    a = lab.evaluate(A, (100, 50), (80.0, 50.0))
    twin = lab.evaluate(A, (100, 50), (80.0, 50.0))
    other_y = lab.evaluate(A, (100, 50), (70.0, 40.0))
    prior = lab.evaluate(A, (100, 50), (80.0, 50.0), evaluated=[1, 1, 0, 0])
    population = [a, a, twin, other_y, prior]
    lab.observe(population)
    eids = tuple(log.resolve(ind) for ind in population)
    assert log.population_at(0) == eids == (0, 0, 1, 2, 3)
    assert lab.obs(a) == lab.obs(twin), "twins share an observation but stay two members"

    lab.observe(list(population))                      # unchanged: no new row
    lab.observe([copy.deepcopy(a), a, twin, other_y, prior])  # a clone keeps its parent's eval_id: unchanged
    lab.observe(population[::-1])                      # reordered: a change
    lab.observe([other_y])                             # smaller: a change
    assert log.population_gens == [0, 3, 4]
    assert log.population_at(2) == eids and log.population_at(3) == eids[::-1] and log.population_at(4) == (2,)

    view = MORunView(log.finalise())
    assert [tuple(view.population(g).eval_ids) for g in range(5)] == [log.population_at(g) for g in range(5)]
    snapshot = view.population(0)
    assert snapshot.geno_ids.tolist() == [lab.geno(A)] * 5 and snapshot.x.tolist() == [A] * 5
    assert snapshot.observed.tolist() == [[80.0, 50.0], [80.0, 50.0], [80.0, 50.0], [70.0, 40.0], [80.0, 50.0]]
    assert snapshot.true_obj.tolist() == [[100.0, 50.0]] * 5
    assert snapshot.obs_ids.tolist() == [lab.obs(a), lab.obs(a), lab.obs(a), lab.obs(other_y), lab.obs(prior)]


def test_a_failed_observation_records_no_population():
    lab = Lab()
    a = lab.evaluate(A, (100, 50), (80.0, 70.0))
    a[3] = 1  # genotype changed without re-evaluation
    with pytest.raises(ProvenanceError):
        lab.observe([a])
    assert lab.log.population_gens == [] and lab.log.population_members == []


# ------------------------------------------------------------------------------ the view

def test_snapshot_ranks_ranges_and_history():
    lab = Lab()
    pts = {"a": (A, (100, 50), (100.0, 50.0)), "b": (B, (90, 60), (90.0, 60.0)),
           "c": (C, (110, 55), (80.0, 70.0)), "d": (D, (50, 90), (50.0, 90.0))}
    ind = {k: lab.evaluate(*v) for k, v in pts.items()}
    lab.observe([ind["a"], ind["b"], ind["a"], ind["c"], ind["d"]])
    lab.observe([ind["a"], ind["b"], ind["a"], ind["c"], ind["d"]])
    lab.observe([ind["d"], ind["a"]])
    view = MORunView(lab.log.finalise())

    snapshot = view.population(1)
    assert isinstance(snapshot, PopulationSnapshot) and len(snapshot) == 5 and snapshot.generation == 1
    # noisy: a dominates b, b dominates c, c dominates d; true: a and c are incomparable, b < a, d last
    assert snapshot.noisy_rank.tolist() == [0, 1, 0, 2, 3]
    assert snapshot.true_rank.tolist() == [0, 1, 0, 0, 2]
    assert snapshot.rank("noisy") is snapshot.noisy_rank, "ranks are computed once"
    assert snapshot.in_ranks("noisy", 1).tolist() == [0, 1, 2]
    assert snapshot.in_ranks("noisy", 3, first=2).tolist() == [3, 4]
    assert snapshot.in_ranks("true", 0).tolist() == [0, 2, 3]
    with pytest.raises(ValueError, match="rank kind"):
        snapshot.rank("clean")
    for array in (snapshot.eval_ids, snapshot.noisy_rank, snapshot.true_rank):
        with pytest.raises(ValueError):
            array[0] = 9

    history = view.population_history()
    assert isinstance(history, PopulationHistory) and history.starts.tolist() == [0, 2] and len(history) == 2
    assert [s.generation for s in history] == [0, 2] and history[-1].generation == 2
    assert history.index_at(1) == 0 and history.at(1).generation == 1
    assert history.at(1).eval_ids.tolist() == history[0].eval_ids.tolist() == [0, 1, 0, 2, 3]
    with pytest.raises(IndexError):
        history.at(3)
    with pytest.raises(IndexError):
        history[2]


def test_version_1_records_answer_fronts_but_not_populations():
    algo = build("NSGA2", prob="kp1")
    step_run(algo)
    record = algo.eval_log.finalise()
    old = copy.deepcopy(record)
    old.pop("current_population")
    old["version"] = 1
    view = MORunView(old)
    assert not view.has_population and MORunView(record).has_population
    view.current_noisy_front(0), view.clean_archive(view.n_generations - 1), view.summary_scalars()
    for query in (lambda: view.population(0), view.population_history):
        with pytest.raises(MORecordError, match="not recorded .*version 1.*regenerate"):
            query()


# ------------------------------------------------------------------------------ rank 0 and HV equivalence

def first_rows(ids, rows):
    """The first of `rows` for each distinct id among ids[rows], in increasing id order."""
    _, first = np.unique(ids[rows], return_index=True)
    return rows[first]


def assert_hv(view, reference, stored, exact):
    """A recomputed HV against a persisted one: NaN exactly when there is no reference point, never compared
    numerically then; equal bitwise for the same point multiset, else within HV_RTOL."""
    if view.ref_point is None:
        assert reference is None and math.isnan(stored), (reference, stored)
        return
    assert reference is not None and not math.isnan(stored), (reference, stored)
    if exact:
        assert reference == stored, (reference, stored)
    else:
        assert math.isclose(reference, stored, rel_tol=HV_RTOL, abs_tol=0.0), (reference, stored)


def check_generation(view, gen):
    """Population ranks at `gen` reproduce the persisted fronts and HVs; returns (corrected, clean) genotypes."""
    pop, w, ref = view.population(gen), view.weights, view.ref_point
    noisy, clean = view.current_noisy_front(gen), view.current_clean_front(gen)

    # noisy rank 0 = the persisted noisy front, one member per distinct observation; HV of their y
    noisy_rows = first_rows(pop.obs_ids, pop.in_ranks("noisy", 0))
    assert pop.obs_ids[noisy_rows].tolist() == noisy.obs_ids.tolist(), gen
    assert_hv(view, hypervolume_of(pop.observed[noisy_rows], w, ref), noisy.hv["current_noisy_front__noisy"], True)

    # true rank 0 = the persisted clean front, one member per distinct genotype; HV of their f(x)
    clean_rows = first_rows(pop.geno_ids, pop.in_ranks("true", 0))
    assert pop.geno_ids[clean_rows].tolist() == clean.geno_ids.tolist(), gen
    assert_hv(view, hypervolume_of(pop.true_obj[clean_rows], w, ref), clean.hv["current_clean_front__clean"], True)

    # corrected: the distinct genotypes of noisy rank 0, re-ranked under f(x) = the dashboard's corrected front
    candidates = first_rows(pop.geno_ids, pop.in_ranks("noisy", 0))
    corrected_rows = candidates[nondominated_ranks(pop.true_obj[candidates], w) == 0]
    snapshot, rows = _front_rows(view, "corrected_noisy", gen)
    assert pop.geno_ids[corrected_rows].tolist() == np.unique(snapshot.geno_ids[rows]).tolist(), gen
    stored = noisy.hv["current_noisy_front__clean"]
    # (a) the persisted definition: f(x) of every noisy-front observation, dominated and repeated points kept
    assert_hv(view, hypervolume_of(pop.true_obj[noisy_rows], w, ref), stored, True)
    # (b) an independent reference over the corrected front's own genotypes (a different point multiset)
    assert_hv(view, hypervolume_of(pop.true_obj[corrected_rows], w, ref), stored, False)
    if ref is not None:  # every corrected point is weakly dominated by a clean-front point of the same population
        assert stored <= clean.hv["current_clean_front__clean"] * (1 + HV_RTOL), gen
    return pop.geno_ids[corrected_rows].tolist(), pop.geno_ids[clean_rows].tolist()


@pytest.mark.parametrize("prob", PROBLEMS)
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_population_ranks_reproduce_the_persisted_fronts_and_hypervolumes(name, prob):
    for algo, view in ((None, MORunView(_finalised(name, prob))), long_run(name, prob)):
        gens = generations_to_check(view)
        if view.n_generations > EXHAUSTIVE_UP_TO:
            assert len(gens) < view.n_generations / 4, "long runs are sampled, not exhaustive"
        for gen in gens:
            check_generation(view, gen)


def _finalised(name, prob):
    algo = build(name, prob=prob)
    step_run(algo)
    return algo.eval_log.finalise()


def test_without_a_reference_point_every_hypervolume_is_nan():
    lab = Lab()
    lab.log = MOEvaluationLogger(KP_WEIGHTS, None)
    a, b = lab.evaluate(A, (100, 50), (100.0, 50.0)), lab.evaluate(B, (90, 40), (80.0, 60.0))
    lab.observe([a, b])
    view = MORunView(lab.log.finalise())
    assert view.ref_point is None
    assert math.isnan(view.current_noisy_front(0).hv["current_noisy_front__noisy"])
    assert math.isnan(view.current_clean_front(0).hv["current_clean_front__clean"])
    check_generation(view, 0)


def corrected_and_clean(evaluations):
    """One observed generation of crafted (genotype, f(x), y) members: (corrected genotypes, clean genotypes,
    stored corrected HV, stored clean HV)."""
    lab = Lab()
    lab.observe([lab.evaluate(*e) for e in evaluations])
    view = MORunView(lab.log.finalise())
    corrected, clean = check_generation(view, 0)
    hv = view.current_noisy_front(0).hv["current_noisy_front__clean"], view.current_clean_front(0).hv[
        "current_clean_front__clean"]
    return [view.genotype(g).tolist() for g in corrected], [view.genotype(g).tolist() for g in clean], hv, lab


def test_the_corrected_front_can_miss_clean_front_genotypes():
    """A and B are incomparable under f(x) (A worth more, B lighter), but B's noisy y is dominated by A's:
    the corrected front is {A}, a strict subset of the clean front {A, B}, with a smaller HV."""
    corrected, clean, (hv_corrected, hv_clean), _ = corrected_and_clean([
        (A, (100, 50), (100.0, 50.0)), (B, (90, 40), (80.0, 60.0))])
    assert corrected == [A] and sorted(clean) == sorted([A, B])
    assert hv_corrected < hv_clean


def test_the_corrected_front_can_hold_genotypes_off_the_clean_front():
    """E dominates D under f(x), but E's noisy y is dominated by D's: the corrected front is {D}, which is
    not on the clean front {E} at all."""
    corrected, clean, (hv_corrected, hv_clean), _ = corrected_and_clean([
        (E, (100, 40), (70.0, 80.0)), (D, (90, 50), (95.0, 45.0))])
    assert corrected == [D] and clean == [E] and D not in clean
    assert hv_corrected < hv_clean


def test_real_noisy_runs_have_corrected_fronts_that_differ_from_clean_fronts():
    differing = 0
    for name in ALGORITHM_NAMES:
        _, view = long_run(name, "kp1")
        for gen in generations_to_check(view):
            corrected, clean = check_generation(view, gen)
            differing += corrected != clean
    assert differing > 0


# ------------------------------------------------------------------------------ the algorithms

@pytest.mark.parametrize("prob", ["kp1", "prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_the_recorded_population_is_the_population_the_algorithm_holds(name, prob):
    """After every generation: the recorded eval_ids are the algorithm's own population, in its order (never
    MoUMDA_ParetoArchive's internal archive), and the live DEAP ranks of those individuals equal the view's."""
    algo = build(name, prob=prob)
    live = []

    def observe(algo):
        log = algo.eval_log
        assert log.population_at(algo.gens) == tuple(log.resolve(ind) for ind in algo.population)
        fronts = tools.sortNondominated(algo.population, len(algo.population))
        rank = {id(ind): k for k, front in enumerate(fronts) for ind in front}
        live.append([rank[id(ind)] for ind in algo.population])

    step_run(algo, observe)
    view = MORunView(algo.eval_log.finalise())
    assert [view.population(g).noisy_rank.tolist() for g in range(view.n_generations)] == live


def test_nsga2_records_survivors_clones_and_lower_ranks():
    """Generation 0 is the evaluated initial population, reordered by NSGA-II's initial sort; later
    populations keep survivors evaluated generations earlier and draw only on the previous population and
    the generation's own evaluations; without variation every offspring is a clone, so eval_ids repeat; and
    survivors' ranks within the population are the ranks NSGA-II gave them in the merged parent+offspring pool."""
    algo, view = long_run("NSGA2", "kp1")
    n = len(algo.population)
    assert sorted(view.population(0).eval_ids.tolist()) == list(range(n))
    survivors = noisy_lower = False
    for g in range(1, view.n_generations):
        members = view.population(g).eval_ids
        allowed = set(view.population(g - 1).eval_ids.tolist()) | set(view.evals_in(g))
        assert set(members.tolist()) <= allowed, g
        survivors |= bool((members < view.gen_last_eval[g - 1]).any())
        noisy_lower |= bool((view.population(g).noisy_rank > 0).any())
    assert survivors and noisy_lower

    clones = build("NSGA2", prob="kp1", cxpb=0.0, mutpb=0.0)
    merged_ranks = []
    select = clones.toolbox.select

    def recording_select(individuals, k):
        fronts = tools.sortNondominated(individuals, len(individuals))
        rank = {id(ind): r for r, front in enumerate(fronts) for ind in front}
        chosen = select(individuals, k)
        merged_ranks.append([rank[id(ind)] for ind in chosen])
        return chosen

    clones.toolbox.register("select", recording_select)
    step_run(clones)
    view = MORunView(clones.eval_log.finalise())
    assert view.n_evals == len(clones.population), "no variation: nothing is re-evaluated"
    assert any(len(set(view.population(g).eval_ids.tolist())) < len(clones.population)
               for g in range(view.n_generations)), "unevaluated clones share their parent's eval_id"
    assert [view.population(g).noisy_rank.tolist() for g in range(1, view.n_generations)] == merged_ranks


@pytest.mark.parametrize("name", [n for n in ALGORITHM_NAMES if n.startswith("MoUMDA")])
def test_moumda_populations_are_each_generations_evaluations(name):
    algo, view = long_run(name, "kp1")
    assert view.population(0).eval_ids.tolist() == list(range(int(view.gen_last_eval[0])))
    repeated_genotypes = False
    for g in range(1, view.n_generations):
        pop = view.population(g)
        assert pop.eval_ids.tolist() == list(view.evals_in(g)), g
        repeated_genotypes |= len(np.unique(pop.geno_ids)) < len(pop)
        if name in ("MoUMDA_noDuplicates", "MoUMDA_prevent_duplicates"):
            assert len(np.unique(pop.geno_ids)) == len(pop), g
    if name not in ("MoUMDA_noDuplicates", "MoUMDA_prevent_duplicates"):
        assert repeated_genotypes, "the same genotype evaluated again is a distinct member"
    if name == "MoUMDA_ParetoArchive":
        archive = {algo.eval_log.resolve(ind) for ind in algo.archive}
        assert not archive <= set(view.population(view.n_generations - 1).eval_ids.tolist()), \
            "the internal archive is not the population"


def replay_semo(view):
    """Test-only cross-check: SEMO's archive rule replayed from the events (one evaluation per generation)."""
    w = np.asarray(view.weights)
    events = view.events()
    y, geno = events.observed * w, events.orig_geno

    def dominates(a, b):
        return bool((y[a] >= y[b]).all() and (y[a] > y[b]).any())

    population, states = [0], [(0,)]
    for e in range(1, view.n_evals):
        if not any(dominates(p, e) for p in population) and not any(geno[p] == geno[e] for p in population):
            population = [p for p in population if not dominates(e, p)] + [e]
        states.append(tuple(population))
    return states


@pytest.mark.parametrize("prob", ["kp1", "cocz1"])
def test_semo_records_its_archive_as_a_sparse_history(prob):
    _, view = long_run("SEMO", prob)
    history = view.population_history()
    replayed = replay_semo(view)
    assert len(history) == 1 + sum(a != b for a, b in zip(replayed, replayed[1:])) < view.n_generations / 10
    true_ranked_lower = False
    for gen in generations_to_check(view):
        pop = view.population(gen)
        assert tuple(pop.eval_ids.tolist()) == replayed[gen], gen
        assert (pop.noisy_rank == 0).all(), "SEMO's population is its noisy non-dominated archive"
        true_ranked_lower |= bool((pop.true_rank > 0).any())
    if prob == "kp1":
        assert true_ranked_lower, "noisy archive members can be truly dominated"
