"""Current-front recording (MO recording plan v6, Work Group 3).

Pins the population-based current-front histories of the MO evaluation log, for every MO algorithm:

    gen_last_eval[g]      exclusive end offset: genuine evaluations completed by the end of generation g
                          (g0 included; never the legacy recorder's clean evaluations)
    current noisy front   ND(current population) under the observed y; members are canonical obs_ids
    current clean front   ND(unique original genotypes of the current population) under f(x); members
                          are genotype ids; computed independently, never the noisy front re-scored
    change points         a row only when membership changes; a membership that disappears and returns
                          is a new row; the state at g is the last row starting at or before g
    hypervolumes          current_noisy_front__noisy, current_noisy_front__clean (the same members
                          re-scored with f(x)), current_clean_front__clean, with the legacy convention

The legacy recorder is the same-run oracle: its every-generation output and its production
change-ordinal output (a new entry only when the noisy front's genotype set changes) are reproduced
from the new histories. Recording stays observational (see also test_mo_eval_logging.py's on/off test,
which runs with front recording active).

Runs in-process and writes nothing.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from deap import creator

from harness.mo_runs import (
    A,
    ALGORITHM_NAMES,
    B,
    C,
    COCZ_WEIGHTS,
    F_A,
    F_B,
    F_C,
    KP_WEIGHTS,
    Lab,
    build,
    direct_hv,
    rng_state,
    step_run,
)
from noisyvis.algorithms.multi_objective import base
from noisyvis.common import pareto
from noisyvis.tracking.logger import clear_active_logger, get_active_logger
from noisyvis.tracking.mo_logger import HV_METRICS, EvaluationLogError, ProvenanceError

POSTERIOR = ["kp0", "kp1", "kpviol1", "cocz1"]


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


def dominates(a, b):
    """Pareto dominance on maximised (weighted) values, as deap.base.Fitness.dominates: no worse in every
    objective and strictly better in at least one; equal points never dominate each other."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def brute_force_nd(points: dict) -> set:
    """Keys whose maximised point no other key's point dominates."""
    return {k for k, p in points.items() if not any(dominates(q, p) for j, q in points.items() if j != k)}


def weighted(values, weights):
    return tuple(v * w for v, w in zip(values, weights))


# ------------------------------------------------------------------------------ constructed cases


def test_clean_front_is_independent_of_the_noisy_front():
    """f(A) dominates f(B) but y(B) dominates y(A): the noisy front is {B's observation}, the clean front
    {A}. Re-scoring the noisy front with f(x) would give {B}, so an implementation that did that fails.
    current_noisy_front__clean is then HV{f(B)}, current_clean_front__clean HV{f(A)}."""
    lab = Lab()
    a = lab.evaluate(A, F_A, (80.0, 70.0))
    b = lab.evaluate(B, F_B, (95.0, 55.0))
    lab.observe([a, b])
    log = lab.log
    assert log.noisy_front_at(0) == (lab.obs(b),)
    assert log.clean_front_at(0) == (lab.geno(A),)
    noisy_hv, rescored_hv, clean_hv = (log.hv_trajectory(m)[0] for m in HV_METRICS)
    assert noisy_hv == lab.hv([(95.0, 55.0)])
    assert rescored_hv == lab.hv([F_B])
    assert clean_hv == lab.hv([F_A])
    assert rescored_hv != clean_hv


def test_change_points_and_generation_lookup():
    """g0 [A, B]; g1 the same population (no rows); g2 B re-evaluated with a new y (noisy row only);
    g3 C added, clean-ND but noisy-dominated (clean row only); g4 C removed (the g0 clean membership
    returns and is a new row). Lookup is the last row starting at or before g."""
    lab = Lab()
    log = lab.log
    a, b = lab.evaluate(A, F_A, (80.0, 70.0)), lab.evaluate(B, F_B, (95.0, 55.0))
    lab.observe([a, b])                                        # g0
    lab.observe([a, b])                                        # g1
    b2 = lab.evaluate(B, F_B, (96.0, 54.0))
    lab.observe([a, b2])                                       # g2
    c = lab.evaluate(C, F_C, (70.0, 90.0))
    lab.observe([a, b2, c])                                    # g3
    lab.observe([a, b2])                                       # g4

    gA, gC = lab.geno(A), lab.geno(C)
    assert log.noisy_front_gens == [0, 2]
    assert log.noisy_front_members == [(lab.obs(b),), (lab.obs(b2),)]
    assert log.clean_front_gens == [0, 3, 4]
    assert log.clean_front_members == [(gA,), tuple(sorted((gA, gC))), (gA,)]

    assert [log.noisy_front_at(g) for g in range(5)] == [(lab.obs(b),)] * 2 + [(lab.obs(b2),)] * 3
    assert log.clean_front_at(2) == (gA,) and log.clean_front_at(3) == tuple(sorted((gA, gC)))
    assert log.clean_front_at(4) == (gA,)
    for g in (-1, 5):
        with pytest.raises(IndexError):
            log.noisy_front_at(g)
        with pytest.raises(IndexError):
            log.clean_front_at(g)

    # HVs change independently: only the noisy ones at g2, only the clean one at g3 and g4
    assert log.hv_trajectory("current_noisy_front__noisy") == \
        [lab.hv([(95.0, 55.0)])] * 2 + [lab.hv([(96.0, 54.0)])] * 3
    assert log.hv_trajectory("current_noisy_front__clean") == [lab.hv([F_B])] * 5
    assert log.hv_trajectory("current_clean_front__clean") == \
        [lab.hv([F_A])] * 3 + [lab.hv([F_A, F_C])] + [lab.hv([F_A])]
    with pytest.raises(KeyError):
        log.hv_trajectory("current_front")


def test_generation_boundaries_are_exclusive_end_offsets():
    """gen_last_eval[g] is the number of evaluations completed by the end of generation g, an exclusive
    end offset and not the last eval_id: generation g's events are range(gen_last_eval[g-1], gen_last_eval[g])
    (from 0 for g0), and gen_last_eval[g] - 1 is its last eval_id when it has any."""
    lab = Lab()
    log = lab.log
    a, b = lab.evaluate(A, F_A, (80.0, 70.0)), lab.evaluate(B, F_B, (95.0, 55.0))
    lab.observe([a, b])                                        # g0: eval_ids 0, 1
    lab.observe([a, b])                                        # g1: none
    b2, c = lab.evaluate(B, F_B, (96.0, 54.0)), lab.evaluate(C, F_C, (70.0, 90.0))
    lab.observe([a, b2, c])                                    # g2: eval_ids 2, 3
    assert log.gen_last_eval == [2, 2, 4]
    starts = [0] + log.gen_last_eval[:-1]
    assert [list(range(s, e)) for s, e in zip(starts, log.gen_last_eval)] == [[0, 1], [], [2, 3]]
    assert log.resolve(b) == log.gen_last_eval[0] - 1 and log.resolve(c) == log.gen_last_eval[2] - 1


def test_duplicates_clones_and_observation_identity():
    """Clones and exact (x, x~, y) twins are one noisy member; the same genotype with a different y, or
    with a different x~ (same x and y), are distinct noisy members; the clean front counts genotypes."""
    lab = Lab()
    log = lab.log
    a = lab.evaluate(A, F_A, (80.0, 50.0))
    twin = lab.evaluate(A, F_A, (80.0, 50.0))                  # exact twin: same obs_id
    other_y = lab.evaluate(A, F_A, (70.0, 40.0))               # incomparable with (80, 50)
    prior = lab.evaluate(A, F_A, (80.0, 50.0), evaluated=[1, 1, 0, 0])  # same x and y, different x~
    lab.observe([a, a, twin, other_y, prior])                 # the same object twice, too
    assert lab.obs(twin) == lab.obs(a)
    assert log.noisy_front_at(0) == tuple(sorted({lab.obs(a), lab.obs(other_y), lab.obs(prior)}))
    assert log.clean_front_at(0) == (lab.geno(A),)

    lab.observe([copy.deepcopy(a), a])                         # a clone and its parent: one member
    assert log.noisy_front_at(1) == (lab.obs(a),)
    assert log.noisy_front_gens == [0, 1]


def test_observing_out_of_order_or_mid_evaluation_fails_loudly():
    lab = Lab()
    a = lab.evaluate(A, F_A, (80.0, 70.0))
    with pytest.raises(EvaluationLogError, match="out of order"):
        lab.log.observe_generation(1, [a])
    lab.observe([a])
    with pytest.raises(EvaluationLogError, match="out of order"):
        lab.log.observe_generation(0, [a])

    def observing(individual):
        lab.log.observe_generation(1, [a])
        return (1.0, 1.0)

    with pytest.raises(EvaluationLogError, match="in progress"):
        lab.log.evaluate(observing, creator.Individual(B), {})
    assert lab.log.n_generations == 1


def test_stale_members_and_nondeterministic_true_objectives_are_refused():
    lab = Lab()
    a = lab.evaluate(A, F_A, (80.0, 70.0))
    a[3] = 1  # genotype changed without re-evaluation
    with pytest.raises(ProvenanceError):
        lab.observe([a])
    assert lab.log.n_generations == 0

    b = lab.evaluate(B, F_B, (95.0, 55.0))
    b_again = lab.evaluate(B, (91, 60), (95.0, 55.0))  # f(B) reported differently
    with pytest.raises(EvaluationLogError, match="not deterministic"):
        lab.observe([b, b_again])
    assert lab.log.n_generations == 0 and lab.log.noisy_front_members == [], "nothing half-recorded"


def test_a_new_y_for_the_same_genotypes_is_a_noisy_change_but_not_a_legacy_change():
    """g0 {obs1(x)} -> g1 {obs2(x)}: the current noisy front changes (new row), but its genotype set does
    not, so the legacy recorder (record_every_gen=False) would not have recorded a new entry."""
    lab = Lab()
    x1 = lab.evaluate(A, F_A, (80.0, 70.0))
    lab.observe([x1])
    x2 = lab.evaluate(A, F_A, (81.0, 69.0))
    lab.observe([x2])
    assert lab.log.noisy_front_gens == [0, 1]
    assert legacy_change_generations(lab.log) == [0]


# ------------------------------------------------------------------------------ real runs

def fronts_observer(algo, observed):
    """Checks every generation of a logged run against brute force computed from the population."""
    log = algo.eval_log
    weights, ref = algo.opt_weights, algo.ref_point
    tf, tf_kwargs = algo.true_fitness_function
    fitness = algo.fitness_function[0]
    g = algo.gens

    # generation boundary: exclusive end offset, equal to the counted evaluations
    assert log.n_generations == g + 1
    assert log.gen_last_eval[g] == algo.evals == fitness.calls == len(log)
    start = log.gen_last_eval[g - 1] if g else 0
    assert [log.event(i).x for i in range(start, log.gen_last_eval[g])] == \
        [x for x, _ in fitness.log[observed["calls"]:]]
    observed["calls"] = fitness.calls

    # current noisy front: brute-force ND of the population's distinct observations
    by_obs = {}
    for ind in algo.population:
        by_obs.setdefault(log.obs_id[log.resolve(ind)], ind.fitness.wvalues)
    noisy = log.noisy_front_at(g)
    assert set(noisy) == brute_force_nd(by_obs) and list(noisy) == sorted(noisy)
    assert frozenset(tuple(int(v) for v in log.genotypes[log.obs_orig_geno(o)]) for o in noisy) \
        == algo._front_signature, "the noisy front's genotype set is the no_improvement signature"

    # current clean front: brute-force ND of the population's distinct genotypes under tf
    true = {tuple(ind): tf(list(ind), **tf_kwargs) for ind in algo.population}
    clean_genotypes = brute_force_nd({x: weighted(t, weights) for x, t in true.items()})
    clean = log.clean_front_at(g)
    assert {log.genotypes[i] for i in clean} == clean_genotypes and list(clean) == sorted(clean)

    # hypervolumes, written out independently
    hvs = {m: log.hv_trajectory(m)[g] for m in HV_METRICS}
    assert hvs["current_noisy_front__noisy"] == direct_hv([log.obs_observed(o) for o in noisy], weights, ref)
    assert hvs["current_noisy_front__clean"] == direct_hv(
        [true[log.genotypes[log.obs_orig_geno(o)]] for o in noisy], weights, ref)
    assert hvs["current_clean_front__clean"] == direct_hv([true[x] for x in clean_genotypes], weights, ref)


@pytest.mark.parametrize("prob", POSTERIOR + ["prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_current_fronts_follow_the_population(name, prob):
    """Every generation from 0, every algorithm (MoUMDA_ParetoArchive included: its population, never its
    internal archive): boundaries, both fronts and all three HVs equal brute force; the history rows
    start at g0, strictly increase in generation, and consecutive memberships differ."""
    algo = build(name, prob=prob)
    observed = {"calls": 0}
    step_run(algo, lambda algo: fronts_observer(algo, observed))
    log = algo.eval_log
    assert log.gen_last_eval[0] == (1 if name == "SEMO" else 12)
    for gens, members in ((log.noisy_front_gens, log.noisy_front_members),
                          (log.clean_front_gens, log.clean_front_members)):
        assert gens[0] == 0 and all(a < b for a, b in zip(gens, gens[1:]))
        assert all(a != b for a, b in zip(members, members[1:]))
    assert all(len(log.hv_trajectory(m)) == algo.gens + 1 for m in HV_METRICS)


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_legacy_recorder_evaluations_are_not_generation_events(name):
    algo = build(name, prob="kp1")
    tf, tf_kwargs = algo.true_fitness_function
    calls = []
    algo.true_fitness_function = (lambda ind, **kw: calls.append(1) or tf(ind, **kw), tf_kwargs)
    step_run(algo)
    assert calls, "the legacy recorder made clean evaluations"
    assert algo.eval_log.gen_last_eval[-1] == algo.evals == len(algo.eval_log)


def test_nsga2_without_variation_has_flat_boundaries():
    algo = build("NSGA2", prob="kp1", cxpb=0.0, mutpb=0.0)
    step_run(algo)
    assert algo.eval_log.gen_last_eval == [12] * (algo.gens + 1)


def snapshot(algo, population):
    state = [rng_state(), algo.gens, algo.evals, algo.stop_trigger, algo._front_unchanged_gens,
             [(tuple(ind), ind.fitness.wvalues, getattr(ind, "_eval_tag", None)) for ind in population]]
    for name in ("archive", "probability_vector", "_prepared_probability_vector", "cluster_labels",
                 "cluster_centers", "cluster_sizes", "cluster_probability_vectors"):
        if hasattr(algo, name):
            state.append(repr(getattr(algo, name)))
    return state


@pytest.mark.parametrize("prob", ["kp1", "prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_observing_a_generation_changes_nothing(name, prob):
    """Immediately before and after every observe_generation call: identical RNG states, population
    genotypes, fitness and tags, and algorithm state."""
    algo = build(name, prob=prob)
    log = algo.eval_log
    original = log.observe_generation
    calls = []

    def spy(gen, population):
        before = snapshot(algo, population)
        original(gen, population)
        assert snapshot(algo, population) == before
        calls.append(gen)

    log.observe_generation = spy
    step_run(algo)
    assert calls == list(range(algo.gens + 1))


# ------------------------------------------------------------------------------ legacy projection

def legacy_entry(algo, k):
    """Legacy recorder entry k, as the dashboard consumes it."""
    return {
        "noisy": sorted((tuple(ind), tuple(float(v) for v in fit))
                        for ind, fit in zip(algo.pareto_solutions[k], algo.pareto_fitnesses[k])),
        "rescored": sorted((tuple(ind), repr(t))
                           for ind, t in zip(algo.pareto_solutions[k], algo.pareto_true_fitnesses[k])),
        "clean": sorted((tuple(ind), tuple(float(v) for v in fit))
                        for ind, fit in zip(algo.true_pareto_solutions[k], algo.true_pareto_fitnesses[k])),
        "hv": (algo.noisy_pf_noisy_hypervolumes[k], algo.noisy_pf_true_hypervolumes[k],
               algo.true_pf_hypervolumes[k]),
    }


def projected_entry(log, g):
    """The same bundle, projected from the new current-front histories at generation g."""
    noisy, clean = log.noisy_front_at(g), log.clean_front_at(g)
    return {
        "noisy": sorted((log.genotypes[log.obs_orig_geno(o)], tuple(float(v) for v in log.obs_observed(o)))
                        for o in noisy),
        "rescored": sorted((log.genotypes[log.obs_orig_geno(o)], repr(log.true_objectives_of(log.obs_orig_geno(o))))
                           for o in noisy),
        "clean": sorted((log.genotypes[i], tuple(float(v) for v in log.true_objectives_of(i))) for i in clean),
        "hv": tuple(log.hv_trajectory(m)[g] for m in HV_METRICS),
    }


def legacy_change_generations(log):
    """Generations at which the legacy recorder (record_every_gen=False) appended an entry: those where
    the genotype set of the noisy front in force differs from the one of its last entry (generation 0
    always). A change of obs_ids alone (same genotypes, new y) is not one."""
    generations, last = [], None
    for g in range(log.n_generations):
        signature = frozenset(tuple(int(v) for v in log.genotypes[log.obs_orig_geno(o)])
                              for o in log.noisy_front_at(g))
        if signature != last:
            generations.append(g)
            last = signature
    return generations


def assert_entries_match(legacy, projected, prob, where):
    if prob != "prior":
        assert projected == legacy, where
        return
    # Prior noise: the legacy recorder collapses (x, y) twins, the new front keeps observations with a
    # different x~ apart (approved semantics), so the noisy fronts are compared as (x, y) sets and the
    # two noisy-front HVs up to the resulting last-ulp differences. The clean front is exact.
    assert set(projected["noisy"]) == set(legacy["noisy"]), where
    assert set(projected["rescored"]) == set(legacy["rescored"]), where
    assert projected["clean"] == legacy["clean"], where
    assert all(math.isclose(p, q, rel_tol=1e-12) for p, q in zip(projected["hv"][:2], legacy["hv"][:2])), where
    assert projected["hv"][2] == legacy["hv"][2], where


@pytest.mark.parametrize("prob", POSTERIOR + ["prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_every_generation_legacy_output_is_reproduced(name, prob):
    """Same run, legacy recorder with record_every_gen=True: its entry g equals the new histories at
    generation g, including all three HVs exactly (posterior noise)."""
    algo = build(name, prob=prob, record_every_gen=True)
    step_run(algo)
    log = algo.eval_log
    assert len(algo.pareto_solutions) == log.n_generations == algo.gens + 1
    for g in range(log.n_generations):
        assert_entries_match(legacy_entry(algo, g), projected_entry(log, g), prob, f"generation {g}")


@pytest.mark.parametrize("prob", POSTERIOR + ["prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_production_legacy_output_is_reproduced(name, prob):
    """Same run, production legacy recorder (record_every_gen=False): entry k is the state at the k-th
    generation where the noisy front's genotype set changed, and n_gens_pareto_best is the run lengths
    between those generations. The dashboard's current inputs are reproducible from the new data."""
    algo = build(name, prob=prob)
    step_run(algo)
    log = algo.eval_log
    changes = legacy_change_generations(log)
    assert len(algo.pareto_solutions) == len(changes)
    for k, g in enumerate(changes):
        assert_entries_match(legacy_entry(algo, k), projected_entry(log, g), prob, f"entry {k} (generation {g})")
    assert algo.n_gens_pareto_best == [b - a for a, b in zip(changes, changes[1:] + [log.n_generations])]


def test_obs_only_noisy_changes_occur_in_real_runs():
    """The noisy history is finer than the legacy entries: real noisy runs re-observe the same front
    genotypes with new y, which is a new noisy row but not a legacy entry."""
    finer = 0
    for name in ALGORITHM_NAMES:
        algo = build(name, prob="kp1")
        step_run(algo)
        finer += len(algo.eval_log.noisy_front_gens) > len(legacy_change_generations(algo.eval_log))
    assert finer > 0


# ------------------------------------------------------------------------------ shared Pareto helpers

@pytest.mark.parametrize("weights", [KP_WEIGHTS, COCZ_WEIGHTS, (-1.0, -1.0)])
def test_hypervolume_of_uses_the_legacy_convention(weights):
    rng = np.random.default_rng(4)
    values = [tuple(float(v) for v in rng.integers(0, 50, 2)) for _ in range(9)]
    ref = [(-10.0 if w > 0 else 60.0) for w in weights]
    # the legacy recorder's inline formula
    w = np.asarray(weights, dtype=float)
    sign = np.where(w > 0, -1.0, 1.0)
    legacy = float(base.hypervolume(np.asarray(values, dtype=float) * sign, np.asarray(ref, dtype=float) * sign))
    assert pareto.hypervolume_of(values, weights, ref) == legacy == direct_hv(values, weights, ref)
    assert pareto.hypervolume_of([], weights, ref) == 0.0
    assert pareto.hypervolume_of(values, weights, None) is None


def test_the_moved_helpers_are_the_ones_the_algorithms_use():
    assert base.nondominated_mask is pareto.nondominated_mask
    assert base.hypervolume is pareto.hypervolume
