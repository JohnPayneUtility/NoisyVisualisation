"""Passive historical archives (MO recording plan v6, Work Group 4).

Pins the two archives the MO evaluation log replays from its event stream, for every MO algorithm:

    noisy archive   ND(all canonical observations seen so far, under y); members are obs_ids
    clean archive   ND(all genotypes generated as x so far, under f(x)); members are genotype ids,
                    discovered at their first event as x (never when only seen as x~)
    intervals       evaluation precision, half-open: present after event e iff enter <= e < exit;
                    exit = NOT_EXITED (-1) while still a member; exit_by = the evicting member
    generations     the archive at generation g is the archive after the first gen_last_eval[g] events
    hypervolumes    noisy_archive__noisy, noisy_archive__clean (the same members under f(x)),
                    clean_archive__clean, at every membership change, with the legacy convention

Dominance is Pareto dominance on weighted values: no worse in every objective and strictly better in at
least one, so equal vectors never dominate each other and distinct equal-valued members all stay.
The archives come from the events only, never from the population or the current fronts, and never
influence the optimiser.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np
import pytest

from harness.mo_runs import (
    A,
    ALGORITHM_NAMES,
    B,
    C,
    KP_WEIGHTS,
    LAB_REF,
    Lab,
    build,
    direct_hv,
    rng_state,
    step_run,
)
from noisyvis.common.pareto import NOT_EXITED, replay_nondominated_archive
from noisyvis.tracking.logger import clear_active_logger, get_active_logger
from noisyvis.tracking.mo_logger import ARCHIVE_HV_METRICS, HV_METRICS, EvaluationLogError

PROBLEMS = ["kp0", "kp1", "kpviol1", "cocz1", "prior"]
D, E = [0, 0, 0, 1], [1, 1, 0, 0]


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


# ------------------------------------------------------------------------------ independent brute force

def nd_keys(points: dict) -> set:
    """Keys of the points no other point Pareto-dominates (maximised values; equal points never dominate)."""
    keys = list(points)
    if not keys:
        return set()
    p = np.array([points[k] for k in keys], dtype=float)
    dominated = ((p[None, :, :] >= p[:, None, :]).all(axis=2) & (p[None, :, :] > p[:, None, :]).any(axis=2)).any(axis=1)
    return {k for k, d in zip(keys, dominated) if not d}


def weighted(values, weights):
    return tuple(v * w for v, w in zip(values, weights))


def brute_noisy(log, n):
    """ND over the distinct (x, x~, y) observations among the first n events, as their obs_ids."""
    points, obs_of = {}, {}
    for e in range(n):
        key = (log.orig_geno[e], log.eval_geno[e], log.observed[e])
        points.setdefault(key, weighted(log.observed[e], log.weights))
        obs_of.setdefault(key, log.obs_id[e])
    return tuple(sorted(obs_of[k] for k in nd_keys(points)))


def brute_clean(log, n):
    """ND under f(x) over the genotypes generated as x among the first n events (never x~)."""
    points = {}
    for e in range(n):
        points.setdefault(log.orig_geno[e], weighted(log.true_objectives[e], log.weights))
    return tuple(sorted(nd_keys(points)))


def archive_points(log, metric, members):
    if metric == "noisy_archive__noisy":
        return [log.obs_observed(o) for o in members]
    if metric == "noisy_archive__clean":
        return [log.true_objectives_of(log.obs_orig_geno(o)) for o in members]
    return [log.true_objectives_of(g) for g in members]


def members_for(log, metric, n):
    return log.noisy_archive_after(n) if metric.startswith("noisy") else log.clean_archive_after(n)


def not_below(later, earlier):
    """later >= earlier up to float rounding (a point with zero exclusive volume can move an ulp)."""
    return later >= earlier - 1e-12 * abs(earlier)


# ------------------------------------------------------------------------------ the generic replay

def test_replay_keeps_exactly_the_nondominated_candidates_with_intervals():
    """Rejection, entry, multi-eviction, equal points and the half-open interval convention."""
    replay = replay_nondominated_archive([
        (0, 10, (50, -50)),   # enters
        (1, 11, (40, -60)),   # dominated by 10: never enters
        (2, 12, (40, -40)),   # incomparable with 10: enters
        (3, 13, (40, -40)),   # equal to 12: neither dominates, enters too
        (4, 14, (60, -30)),   # dominates 10, 12 and 13: all three exit at 4
    ])
    assert replay.members == (10, 12, 13, 14)
    assert replay.enter == (0, 2, 3, 4)
    assert replay.exit == (4, 4, 4, NOT_EXITED)
    assert replay.exit_by == (14, 14, 14, NOT_EXITED)
    assert replay.changes == ((0, 10, ()), (2, 12, ()), (3, 13, ()), (4, 14, (10, 12, 13)))
    assert [replay.members_after(n) for n in range(6)] == \
        [(), (10,), (10,), (10, 12), (10, 12, 13), (14,)]


# ------------------------------------------------------------------------------ constructed cases

def test_entry_rejection_and_eviction_in_both_archives():
    lab = Lab()
    for bits, value in ((A, (50, 50)), (B, (40, 60)), (C, (40, 40)), (D, (60, 30))):
        lab.evaluate(bits, value, tuple(float(v) for v in value))
    log = lab.log
    obs = [log.obs_id[e] for e in range(4)]
    geno = [lab.geno(bits) for bits in (A, B, C, D)]

    noisy = log.noisy_archive
    assert noisy.members == (obs[0], obs[2], obs[3])       # B was dominated by A: never entered
    assert (noisy.enter, noisy.exit, noisy.exit_by) == ((0, 2, 3), (3, 3, NOT_EXITED), (obs[3], obs[3], NOT_EXITED))
    clean = log.clean_archive
    assert clean.members == (geno[0], geno[2], geno[3])
    assert (clean.enter, clean.exit, clean.exit_by) == ((0, 2, 3), (3, 3, NOT_EXITED), (geno[3], geno[3], NOT_EXITED))

    assert [log.noisy_archive_at_eval(e) for e in range(4)] == \
        [(obs[0],), (obs[0],), (obs[0], obs[2]), (obs[3],)]
    assert log.clean_archive_at_eval(2) == (geno[0], geno[2])
    for bad in (-1, 4):
        with pytest.raises(IndexError):
            log.noisy_archive_at_eval(bad)
    with pytest.raises(IndexError):
        log.clean_archive_after(5)
    assert log.noisy_archive_after(0) == () and log.archive_hv_after("noisy_archive__noisy", 0) is None


def test_equal_vectors_are_distinct_members():
    """Different observations with equal y, and different genotypes with equal f(x), are all members:
    equal vectors never dominate each other. A later candidate dominating them evicts all at once."""
    lab = Lab()
    a = lab.evaluate(A, (50, 50), (50.0, 50.0))
    b = lab.evaluate(B, (50, 50), (50.0, 50.0))                 # different x, equal y and f(x)
    a_prior = lab.evaluate(A, (50, 50), (50.0, 50.0), evaluated=[1, 0, 1, 0])  # same x and y, other x~
    log = lab.log
    noisy_members = tuple(sorted({lab.obs(a), lab.obs(b), lab.obs(a_prior)}))
    assert len(noisy_members) == 3 and log.noisy_archive_after(3) == noisy_members
    assert log.clean_archive_after(3) == tuple(sorted((lab.geno(A), lab.geno(B))))

    d = lab.evaluate(D, (60, 40), (60.0, 40.0))
    assert log.noisy_archive_after(4) == (lab.obs(d),)
    assert log.clean_archive_after(4) == (lab.geno(D),)
    assert set(log.noisy_archive.exit[:3]) == {3} and set(log.noisy_archive.exit_by[:3]) == {lab.obs(d)}
    assert set(log.clean_archive.exit[:2]) == {3}


def test_exact_twins_add_no_member_and_no_change():
    lab = Lab()
    lab.evaluate(A, (50, 50), (50.0, 50.0))
    lab.evaluate(A, (50, 50), (50.0, 50.0))  # exact (x, x~, y) twin
    lab.evaluate(A, (50, 50), (50.0, 50.0))
    log = lab.log
    assert len(log.noisy_archive.changes) == 1 and len(log.clean_archive.changes) == 1
    assert log.noisy_archive_after(3) == (log.obs_id[0],) and log.clean_archive_after(3) == (lab.geno(A),)


def filler(lab, count):
    """Evaluations of a dominated genotype E: they occupy event numbers without changing either archive
    once a better member exists."""
    for _ in range(count):
        lab.evaluate(E, (1, 99), (1.0, 99.0))


def test_half_open_intervals_and_enter_exit_within_one_generation():
    """P is evaluated at event 0 and nine dominated fillers follow; generation 0 ends. A enters at event
    10 and B evicts it (and P) at event 11, both inside generation 1. A's interval is [10, 11): in the
    archive after event 10 only, although no generation boundary ever sees it."""
    lab = Lab()
    p = lab.evaluate(C, (50, 50), (50.0, 50.0))
    filler(lab, 9)
    lab.observe([p])                                              # generation 0: events 0..9
    a = lab.evaluate(A, (55, 55), (55.0, 55.0))                   # event 10: incomparable with P
    b = lab.evaluate(B, (60, 50), (60.0, 50.0))                   # event 11: dominates A and P
    lab.observe([b])                                              # generation 1: events 10..11
    log = lab.log

    noisy = log.noisy_archive
    row = noisy.members.index(lab.obs(a))
    assert (noisy.enter[row], noisy.exit[row], noisy.exit_by[row]) == (10, 11, lab.obs(b))
    assert lab.obs(a) in log.noisy_archive_at_eval(10)
    assert lab.obs(a) not in log.noisy_archive_at_eval(11)
    assert lab.obs(a) in log.noisy_archive_after(11) and lab.obs(a) not in log.noisy_archive_after(12)
    assert log.noisy_archive_at(0) == (lab.obs(p),)
    assert log.noisy_archive_at(1) == (lab.obs(b),)
    assert lab.geno(A) not in log.clean_archive_at(1)
    assert row < len(noisy.members) and noisy.exit[noisy.members.index(lab.obs(b))] == NOT_EXITED


def test_genotype_first_seen_as_x_tilde_is_discovered_only_as_x():
    """Eval 20: B appears only as x~ (x = C). Eval 80: B is generated as x. B is interned at 20 but
    enters the clean archive at 80, never earlier. f(x) of eval 20 is f(C), so the determinism check
    must group by the original genotype only; grouping by x~ would wrongly compare f(C) with f(B)."""
    lab = Lab()
    lab.evaluate(E, (1, 99), (1.0, 99.0))
    filler(lab, 19)                                                # events 0..19
    lab.evaluate(C, (10, 10), (10.0, 10.0), evaluated=B)           # event 20: B only as x~
    filler(lab, 59)                                                # events 21..79
    lab.evaluate(B, (90, 10), (90.0, 10.0))                        # event 80: B as x
    log = lab.log
    g_b = lab.geno(B)
    assert log.genotypes.index(tuple(B)) == g_b and log.eval_geno[20] == g_b and log.orig_geno[20] != g_b

    clean = log.clean_archive
    assert clean.enter[clean.members.index(g_b)] == 80
    assert g_b not in log.clean_archive_after(80) and g_b in log.clean_archive_after(81)
    assert all(g_b not in log.clean_archive_at_eval(e) for e in range(80))
    # the noisy archive does see the eval-20 observation (x = C, x~ = B) as its own candidate
    assert log.obs_id[20] in log.noisy_archive_at_eval(20)


def test_an_evaluated_candidate_never_in_the_population_is_archived():
    """A candidate evaluated and discarded without ever being in an observed population still belongs to
    both archives: they come from the events, not from the population or the current fronts."""
    lab = Lab()
    a = lab.evaluate(A, (50, 50), (50.0, 50.0))
    lab.observe([a])
    discarded = lab.evaluate(B, (40, 40), (40.0, 40.0))             # incomparable with A
    lab.observe([a])
    log = lab.log
    assert log.noisy_front_at(1) == (lab.obs(a),) and log.clean_front_at(1) == (lab.geno(A),)
    assert lab.obs(discarded) in log.noisy_archive_at(1)
    assert lab.geno(B) in log.clean_archive_at(1)


def test_nondeterministic_true_objectives_are_refused():
    lab = Lab()
    lab.evaluate(A, (50, 50), (50.0, 50.0))
    lab.evaluate(A, (51, 50), (50.0, 50.0))
    with pytest.raises(EvaluationLogError, match="not deterministic"):
        lab.log.noisy_archive


def test_archive_hypervolumes_and_the_non_monotone_cross_metric():
    """Each HV equals an independent calculation after each change. noisy_archive__clean judges the
    noisily-best observations by f(x), so it can fall: B's y dominates A's while f(B) is far worse."""
    lab = Lab()
    lab.evaluate(A, (50, 50), (50.0, 50.0))
    lab.evaluate(B, (10, 90), (60.0, 40.0))
    log = lab.log
    for n in (1, 2):
        for metric in ARCHIVE_HV_METRICS:
            assert log.archive_hv_after(metric, n) == \
                direct_hv(archive_points(log, metric, members_for(log, metric, n)), KP_WEIGHTS, LAB_REF)
    assert log.archive_hv_after("noisy_archive__noisy", 2) > log.archive_hv_after("noisy_archive__noisy", 1)
    assert log.archive_hv_after("noisy_archive__clean", 2) < log.archive_hv_after("noisy_archive__clean", 1)
    assert log.archive_hv_after("clean_archive__clean", 2) == log.archive_hv_after("clean_archive__clean", 1)
    with pytest.raises(KeyError):
        log.archive_hv_after("clean_front", 1)


def test_generation_views_follow_the_current_boundaries_not_the_cache():
    """The replay is cached by event count, but a generation can pass with no new evaluations: generation
    views always read the current gen_last_eval. More events invalidate the cached replay."""
    lab = Lab()
    a = lab.evaluate(A, (50, 50), (50.0, 50.0))
    lab.observe([a])
    log = lab.log
    assert log.archive_hv_trajectory("noisy_archive__noisy") == [lab.hv([(50.0, 50.0)])]
    lab.observe([a])                                               # generation 1: no new evaluations
    assert log.archive_hv_trajectory("noisy_archive__noisy") == [lab.hv([(50.0, 50.0)])] * 2
    assert log.noisy_archive_at(1) == log.noisy_archive_at(0)

    before = log.noisy_archive
    lab.evaluate(D, (60, 40), (60.0, 40.0))
    assert log.noisy_archive is not before
    assert log.noisy_archive_after(len(log)) == brute_noisy(log, len(log)) == (log.obs_id[1],)
    for g in (-1, 2):
        with pytest.raises(IndexError):
            log.noisy_archive_at(g)


# ------------------------------------------------------------------------------ real runs

def gen_of(log, eval_id):
    return bisect_right(log.gen_last_eval, eval_id)


def population_history(algo, seen):
    log = algo.eval_log
    for ind in algo.population:
        eval_id = log.resolve(ind)
        seen["obs"].add(log.obs_id[eval_id])
        seen["geno"].add(log.orig_geno[eval_id])


def run(name, prob):
    algo = build(name, prob=prob)
    seen = {"obs": set(), "geno": set()}
    step_run(algo, lambda algo: population_history(algo, seen))
    return algo, seen


@pytest.mark.parametrize("prob", PROBLEMS)
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_archives_equal_brute_force_after_every_evaluation(name, prob):
    """After every event and at every generation (g0 = the initialisation evaluations), both archives
    equal an independent brute-force ND; each member entered exactly at its discovery event."""
    algo, _ = run(name, prob)
    log = algo.eval_log
    for n in range(len(log) + 1):
        assert log.noisy_archive_after(n) == brute_noisy(log, n), f"noisy archive after {n} events"
        assert log.clean_archive_after(n) == brute_clean(log, n), f"clean archive after {n} events"
    for g in range(log.n_generations):
        assert log.noisy_archive_at(g) == brute_noisy(log, log.gen_last_eval[g])
        assert log.clean_archive_at(g) == brute_clean(log, log.gen_last_eval[g])
    noisy, clean = log.noisy_archive, log.clean_archive
    assert all(enter == log.obs_first_eval[o] for o, enter in zip(noisy.members, noisy.enter))
    assert all(enter == log._orig_eval_by_geno[g] for g, enter in zip(clean.members, clean.enter))


@pytest.mark.parametrize("prob", PROBLEMS)
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_archive_hypervolumes_in_real_runs(name, prob):
    """Every archive HV equals an independent calculation at every generation; noisy_archive__noisy and
    clean_archive__clean never decrease (after every event and every generation), and they bound the
    current fronts' own-space HVs from above. noisy_archive__clean is not required to be monotone."""
    algo, _ = run(name, prob)
    log = algo.eval_log
    weights, ref = log.weights, log.ref_point
    for metric in ARCHIVE_HV_METRICS:
        trajectory = log.archive_hv_trajectory(metric)
        assert len(trajectory) == algo.gens + 1
        for g, value in enumerate(trajectory):
            members = members_for(log, metric, log.gen_last_eval[g])
            assert value == direct_hv(archive_points(log, metric, members), weights, ref), (metric, g)
    for metric in ("noisy_archive__noisy", "clean_archive__clean"):
        per_event = [log.archive_hv_after(metric, n) for n in range(1, len(log) + 1)]
        assert all(not_below(b, a) for a, b in zip(per_event, per_event[1:])), metric
    current = {m: log.hv_trajectory(m) for m in HV_METRICS}
    archive = {m: log.archive_hv_trajectory(m) for m in ARCHIVE_HV_METRICS}
    assert all(not_below(a, c) for a, c in zip(archive["noisy_archive__noisy"], current["current_noisy_front__noisy"]))
    assert all(not_below(a, c) for a, c in zip(archive["clean_archive__clean"], current["current_clean_front__clean"]))


def test_archives_are_not_a_union_of_current_fronts():
    """Across real noisy runs, the archives hold members that were never in any observed population
    (evaluated and discarded), and members that entered and were evicted within one generation."""
    never_in_population = within_one_generation = 0
    for prob in ("kp1", "prior"):
        for name in ALGORITHM_NAMES:
            algo, seen = run(name, prob)
            log = algo.eval_log
            never_in_population += len(set(log.noisy_archive.members) - seen["obs"])
            never_in_population += len(set(log.clean_archive.members) - seen["geno"])
            for replay in (log.noisy_archive, log.clean_archive):
                within_one_generation += sum(
                    exit != NOT_EXITED and gen_of(log, enter) == gen_of(log, exit)
                    for enter, exit in zip(replay.enter, replay.exit))
    assert never_in_population > 0
    assert within_one_generation > 0


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_prior_noise_discovery_in_real_runs(name):
    """Genotypes seen as x~ before they are generated as x are clean candidates only from their first
    x event; the noisy archive keeps observations with different x~ apart."""
    algo, _ = run(name, "prior")
    log = algo.eval_log
    first_as_x_tilde = {}
    for e in range(len(log)):
        if log.eval_geno[e] != log.orig_geno[e]:
            first_as_x_tilde.setdefault(log.eval_geno[e], e)
    early = {g: e for g, e in first_as_x_tilde.items()
             if g not in log._orig_eval_by_geno or e < log._orig_eval_by_geno[g]}
    assert early, "some genotype is evaluated as x~ before it is ever generated"
    clean = log.clean_archive
    for g, enter in zip(clean.members, clean.enter):
        assert enter == log._orig_eval_by_geno[g]
        if g in early:
            assert enter > early[g]
    for g in early:
        for n in range(log._orig_eval_by_geno.get(g, len(log) - 1) + 1):
            assert g not in log.clean_archive_after(n)


# ------------------------------------------------------------------------------ non-interference

def snapshot(algo):
    state = [rng_state(), algo.gens, algo.evals, algo.stop_trigger,
             (algo._front_signature, algo._front_unchanged_gens),
             [(tuple(ind), ind.fitness.wvalues, getattr(ind, "_eval_tag", None)) for ind in algo.population]]
    for name in ("archive", "probability_vector", "_prepared_probability_vector", "cluster_labels",
                 "cluster_centers", "cluster_sizes", "cluster_probability_vectors"):
        if hasattr(algo, name):
            state.append(repr(getattr(algo, name)))
    return state


def logger_columns(log):
    return repr([log.orig_geno, log.eval_geno, log.true_objectives, log.observed, log.obs_id,
                 log.gen_last_eval, log.noisy_front_gens, log.noisy_front_members, log.noisy_front_hv_noisy,
                 log.noisy_front_hv_clean, log.clean_front_gens, log.clean_front_members, log.clean_front_hv])


def trace(name, prob, query):
    algo = build(name, prob=prob)
    states = []

    def observe(algo):
        if query:
            log, before = algo.eval_log, (snapshot(algo), logger_columns(algo.eval_log))
            log.noisy_archive_at(algo.gens), log.clean_archive_at(algo.gens)
            log.archive_hv_trajectory("clean_archive__clean")
            assert (snapshot(algo), logger_columns(log)) == before, "a replay changed state"
        states.append(snapshot(algo))

    step_run(algo, observe)
    return states, algo.fitness_function[0].log, logger_columns(algo.eval_log), algo.pareto_fitnesses


@pytest.mark.parametrize("prob", ["kp1", "prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_querying_the_archives_changes_nothing(name, prob):
    """A run whose archives are replayed after every generation is identical, generation by generation,
    to one that never replays them: x, x~, y, populations, PA archive, probability vectors, KMeans
    state, gens/evals/stop, stagnation, both RNGs, the WG3 histories and the legacy recorder."""
    queried, plain = trace(name, prob, True), trace(name, prob, False)
    assert queried[0] == plain[0]
    assert queried[1] == plain[1]
    assert queried[2] == plain[2]
    assert repr(queried[3]) == repr(plain[3])
