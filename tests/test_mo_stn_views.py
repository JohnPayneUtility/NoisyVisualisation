"""Multi-objective STN views, data side (mo_frames.stn_view_entries and its helpers).

Five views pair a primary front with an optional satellite front, from Algorithm POV (the noisy front is the
primary trajectory) or True POV (a paired view's roles swapped). Each front kind is read from MORunView
with the stored HV of the front it represents:

    noisy_population    current noisy front          current_noisy_front__noisy (all observations' y)
    clean_population    current clean front          current_clean_front__clean (independent of the noisy one)
    corrected_noisy     the noisy front's distinct   current_noisy_front__clean
                        genotypes still ND under f(x)
    noisy_archive       noisy archive through g      noisy_archive__noisy
    clean_archive       clean archive through g      clean_archive__clean

A front state is (distinct genotypes, HV). A run yields an item where its primary or satellite state
changes, and per-role state ids recur when a state does.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from harness.mo_runs import A, B, C, F_A, F_B, F_C, Lab, build, step_run
from noisyvis.dashboard.mo_frames import (
    DEFAULT_STN_POV,
    DEFAULT_STN_VIEW,
    STN_FRONT_KINDS,
    STN_VIEWS,
    STNFrontState,
    stn_candidate_generations,
    stn_front_state,
    stn_state_items,
    stn_view_entries,
    stn_view_fronts,
)
from noisyvis.results.mo_view import MORunView
from noisyvis.tracking.logger import clear_active_logger, get_active_logger

PAIRS = {  # view -> Algorithm POV (primary, satellite)
    'population_noisy': ('noisy_population', None),
    'population_noisy_vs_clean': ('noisy_population', 'clean_population'),
    'population_noisy_vs_corrected': ('noisy_population', 'corrected_noisy'),
    'archive_noisy': ('noisy_archive', None),
    'archive_noisy_vs_clean': ('noisy_archive', 'clean_archive'),
}
METRICS = {
    'noisy_population': 'current_noisy_front__noisy',
    'clean_population': 'current_clean_front__clean',
    'corrected_noisy': 'current_noisy_front__clean',
    'noisy_archive': 'noisy_archive__noisy',
    'clean_archive': 'clean_archive__clean',
}


@pytest.fixture(autouse=True)
def no_active_logger():
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


# ------------------------------------------------------------------------------ the crafted run
#
# Knapsack weights: maximise the first objective, minimise the second.
#   true    A (100, 50)  B (90, 60)  C (110, 55)  D (120, 52)  E (130, 54)
#           A dominates B; D dominates C; E dominates B and C; A, D, E are mutually incomparable
#   noisy   a (100, 50) a2 (101, 51) a3 = p (102, 51.5) b (105, 55) c (110, 60) d (95, 70) e (90, 80)
#           a, a2, a3/p, b, c are mutually incomparable; a dominates d and e
#
#   g0  [a, a2, b, c, d]      noisy {A,B,C} (4 obs)   corrected {A,C}   clean {A,D}
#   g1  the same population   nothing changes
#   g2  a2 -> a3              noisy: same genotypes, new HV
#   g3  a3 -> p               p is A with a3's y but another x~: a new observation, same genotypes and HV
#   g4  + e                   clean {A,D,E}; noisy unchanged (e is noisy-dominated)
#   g5  - e                   clean {A,D} again; the clean archive keeps E
#   g6  - c                   noisy {A,B}; corrected {A}; clean unchanged
#   g7  + c, - d              noisy back to its g2 state; clean {A,C}

D, E = [0, 0, 0, 1], [1, 1, 0, 0]
F_D, F_E = (120, 52), (130, 54)
Y_A, Y_A2, Y_A3, Y_B, Y_C, Y_D, Y_E = (100.0, 50.0), (101.0, 51.0), (102.0, 51.5), (105.0, 55.0), (110.0, 60.0), \
    (95.0, 70.0), (90.0, 80.0)


def crafted():
    lab = Lab()
    a, a2 = lab.evaluate(A, F_A, Y_A), lab.evaluate(A, F_A, Y_A2)
    b, c, d = lab.evaluate(B, F_B, Y_B), lab.evaluate(C, F_C, Y_C), lab.evaluate(D, F_D, Y_D)
    lab.observe([a, a2, b, c, d])                                  # g0
    lab.observe([a, a2, b, c, d])                                  # g1
    a3 = lab.evaluate(A, F_A, Y_A3)
    lab.observe([a, a3, b, c, d])                                  # g2
    p = lab.evaluate(A, F_A, Y_A3, evaluated=[0, 1, 1, 0])
    lab.observe([a, p, b, c, d])                                   # g3
    e = lab.evaluate(E, F_E, Y_E)
    lab.observe([a, p, b, c, d, e])                                # g4
    lab.observe([a, p, b, c, d])                                   # g5
    lab.observe([a, p, b, d])                                      # g6
    lab.observe([a, p, b, c])                                      # g7
    return lab, MORunView(lab.log.finalise())


@pytest.fixture(scope="module")
def run():
    clear_active_logger()
    return crafted()


def genos(lab, *bits):
    return tuple(sorted(lab.geno(x) for x in bits))


def items_of(view, view_id, pov=DEFAULT_STN_POV, fields=('gen_idx', 'state1', 'state2', 'primary_changed')):
    return [tuple(item[f] for f in fields) for item in stn_view_entries(view, view_id, pov)]


# ------------------------------------------------------------------------------ T1: views and POV

def test_views_map_to_their_fronts_from_both_points_of_view():
    assert set(STN_VIEWS) == set(PAIRS) and DEFAULT_STN_VIEW == 'population_noisy' and DEFAULT_STN_POV == 'algo'
    assert set(STN_FRONT_KINDS) == set(METRICS)
    assert {kind: STN_FRONT_KINDS[kind][2] for kind in STN_FRONT_KINDS} == METRICS
    for view_id, (noisy, true) in PAIRS.items():
        assert stn_view_fronts(view_id) == stn_view_fronts(view_id, 'algo') == (noisy, true)
        assert stn_view_fronts(view_id, 'true') == ((true, noisy) if true else (noisy, None)), view_id
    for bad in (('npnhv', 'algo'), ('population_noisy', 'clean'), ('archive_noisy_vs_clean', None)):
        with pytest.raises(ValueError):
            stn_view_fronts(*bad)


# ------------------------------------------------------------------------------ T2: front kinds

def test_front_kinds_hold_their_distinct_genotypes_and_stored_hv(run):
    lab, view = run
    state = {kind: stn_front_state(view, kind, 0) for kind in STN_FRONT_KINDS}

    assert state['noisy_population'].geno_ids == genos(lab, A, B, C)
    assert state['corrected_noisy'].geno_ids == genos(lab, A, C)           # B is dominated by A under f
    assert state['clean_population'].geno_ids == genos(lab, A, D)          # D: not on the noisy front
    assert state['noisy_archive'].geno_ids == genos(lab, A, B, C)
    assert state['clean_archive'].geno_ids == genos(lab, A, D)

    # repeated observations of A count as observations, not genotypes; the HV keeps every observation
    assert len(state['noisy_population'].solutions) == 3 and state['noisy_population'].n_observations == 4
    assert state['noisy_population'].hv == lab.hv([Y_A, Y_A2, Y_B, Y_C]) != lab.hv([Y_A, Y_B, Y_C])
    assert state['corrected_noisy'].n_observations == 2                     # its genotypes, A and C
    assert state['corrected_noisy'].hv == lab.hv([F_A, F_C])
    assert state['clean_population'].hv == lab.hv([F_A, F_D])
    assert state['clean_population'].n_observations == 2
    assert state['clean_archive'].hv == lab.hv([F_A, F_D])

    # solutions are the genotypes, distinct and in lexicographic order
    assert state['noisy_population'].solutions == sorted([A, B, C])
    assert state['clean_population'].solutions == sorted([A, D])


def _dominates(a, b, weights):
    wa, wb = np.asarray(a) * weights, np.asarray(b) * weights
    return bool(np.all(wa >= wb) and np.any(wa > wb))


def test_every_state_matches_morunview_directly(run):
    """At every generation, every kind: its genotypes are the snapshot's distinct genotypes (for the
    corrected front, those no other dominates under f, written out here independently), its HV the stored
    metric, its member count the snapshot's members (the corrected front's: its genotypes)."""
    _, view = run
    weights = np.asarray(view.weights, dtype=float)
    snapshots = {
        'noisy_population': view.current_noisy_front, 'clean_population': view.current_clean_front,
        'corrected_noisy': view.current_noisy_front, 'noisy_archive': view.noisy_archive,
        'clean_archive': view.clean_archive,
    }
    for gen in range(view.n_generations):
        for kind, snapshot_at in snapshots.items():
            snapshot = snapshot_at(gen)
            true_of = {int(g): tuple(t) for g, t in zip(snapshot.geno_ids, snapshot.true_obj)}
            members = sorted(true_of)
            if kind == 'corrected_noisy':
                members = [g for g in members
                           if not any(_dominates(true_of[o], true_of[g], weights) for o in true_of if o != g)]
            state = stn_front_state(view, kind, gen)
            assert state.geno_ids == tuple(members), (kind, gen)
            assert state.solutions == sorted([int(v) for v in view.genotype(g)] for g in members)
            assert state.hv == snapshot.hv[METRICS[kind]]
            if kind == 'corrected_noisy':           # the corrected front's members are its genotypes
                assert state.n_observations == len(state.geno_ids) == len(state.solutions)
            else:
                assert state.n_observations == len(snapshot.geno_ids)
    with pytest.raises(ValueError):
        stn_front_state(view, 'clean_front', 0)


def test_corrected_front_is_a_true_nd_subset_of_the_noisy_front_and_not_the_clean_front(run):
    lab, view = run
    for gen in range(view.n_generations):
        noisy = set(stn_front_state(view, 'noisy_population', gen).geno_ids)
        corrected = set(stn_front_state(view, 'corrected_noisy', gen).geno_ids)
        assert corrected and corrected <= noisy
    # g0: the clean front holds D, which the noisy front does not; the corrected front holds C, which the
    # clean front does not (D dominates it), so neither is the other
    corrected, clean = (set(stn_front_state(run[1], k, 0).geno_ids) for k in ('corrected_noisy', 'clean_population'))
    assert lab.geno(D) in clean - set(stn_front_state(view, 'noisy_population', 0).geno_ids)
    assert lab.geno(C) in corrected - clean
    # g6: the noisy front {A, B} corrects to {A}; its HV is the stored re-scored HV, equal to HV{f(A)}
    g6 = stn_front_state(view, 'corrected_noisy', 6)
    assert g6.geno_ids == genos(lab, A) and g6.hv == lab.hv([F_A]) == lab.hv([F_A, F_A, F_B])


# ------------------------------------------------------------------------------ T4: archives through g

def test_archive_states_are_the_archives_through_each_generation(run):
    lab, view = run
    assert [stn_front_state(view, 'clean_archive', g).geno_ids for g in (0, 3, 4, 5, 7)] == \
        [genos(lab, A, D)] * 2 + [genos(lab, A, D, E)] * 3
    assert stn_front_state(view, 'clean_archive', 0).geno_ids != \
        stn_front_state(view, 'clean_archive', view.n_generations - 1).geno_ids
    # g5: the population lost E but the clean archive keeps it
    assert stn_front_state(view, 'clean_population', 5).geno_ids == genos(lab, A, D)
    # the noisy archive gains observations of A (a3 at g2, p at g3) but never a genotype
    noisy = [stn_front_state(view, 'noisy_archive', g) for g in range(view.n_generations)]
    assert {s.geno_ids for s in noisy} == {genos(lab, A, B, C)}
    assert [s.n_observations for s in noisy] == [4, 4, 5, 6, 6, 6, 6, 6]
    assert noisy[2].hv == lab.hv([Y_A, Y_A2, Y_A3, Y_B, Y_C]) == noisy[3].hv != noisy[1].hv


# ------------------------------------------------------------------------------ T5: change rule and ids

def test_population_noisy_items_follow_genotype_or_hv_changes(run):
    lab, view = run
    # g2: HV-only change (new state); g3: a new observation with the same genotypes and HV (no item, though
    # the logger records a membership row there); g6: new genotypes; g7: the g2 state again (its id)
    assert 3 in view.current_noisy_front_history().starts
    assert items_of(view, 'population_noisy') == [(0, 0, None, True), (2, 1, None, True), (6, 2, None, True),
                                                  (7, 1, None, True)]
    items = stn_view_entries(view, 'population_noisy')
    assert [i['change_idx'] for i in items] == [0, 1, 2, 3]
    assert [i['metric1'] for i in items] == [lab.hv([Y_A, Y_A2, Y_B, Y_C]), lab.hv([Y_A, Y_A3, Y_B, Y_C]),
                                            lab.hv([Y_A, Y_A3, Y_B]), lab.hv([Y_A, Y_A3, Y_B, Y_C])]
    assert [len(i['front1']) for i in items] == [3, 3, 2, 3] and [i['n_obs1'] for i in items] == [4, 4, 3, 4]
    assert items[0]['front1'] == items[1]['front1'] and items[0]['metric1'] != items[1]['metric1']
    assert all(i['front2'] is None and i['metric2'] is None and i['n_obs2'] is None for i in items)


def test_paired_views_track_primary_and_satellite_states_independently(run):
    lab, view = run
    # Algorithm POV, noisy vs clean: g4 and g5 are satellite-only changes (Case B: the primary state is
    # kept and primary_changed is False), g5 returns to clean state 0, g6 is primary-only (Case A), g7 both
    # (Case C: the primary revisits state 1, the satellite is new)
    assert items_of(view, 'population_noisy_vs_clean') == [
        (0, 0, 0, True), (2, 1, 0, True), (4, 1, 1, False), (5, 1, 0, False), (6, 2, 0, True), (7, 1, 2, True)]
    items = stn_view_entries(view, 'population_noisy_vs_clean')
    assert [i['metric2'] for i in items] == [lab.hv([F_A, F_D]), lab.hv([F_A, F_D]), lab.hv([F_A, F_D, F_E]),
                                            lab.hv([F_A, F_D]), lab.hv([F_A, F_D]), lab.hv([F_A, F_C])]
    assert [i['front2'] for i in items][2] == sorted([A, D, E]) and items[2]['n_obs2'] == 3

    # True POV: the same generations and states with the roles swapped; primary_changed follows the clean front
    assert items_of(view, 'population_noisy_vs_clean', 'true') == [
        (0, 0, 0, True), (2, 0, 1, False), (4, 1, 1, True), (5, 0, 1, True), (6, 0, 2, False), (7, 2, 1, True)]
    algo, true = (stn_view_entries(view, 'population_noisy_vs_clean', pov) for pov in ('algo', 'true'))
    for a, t in zip(algo, true):
        assert (a['front1'], a['metric1'], a['n_obs1']) == (t['front2'], t['metric2'], t['n_obs2'])
        assert (a['front2'], a['metric2'], a['n_obs2']) == (t['front1'], t['metric1'], t['n_obs1'])

    # noisy vs corrected: the corrected front only moves with the noisy genotypes (g6 both change, g7 both
    # return), never on its own
    assert items_of(view, 'population_noisy_vs_corrected') == [
        (0, 0, 0, True), (2, 1, 0, True), (6, 2, 1, True), (7, 1, 0, True)]
    assert items_of(view, 'population_noisy_vs_corrected', 'true') == [
        (0, 0, 0, True), (2, 0, 1, False), (6, 1, 2, True), (7, 0, 1, True)]

    # archives: g2 the noisy archive's HV grows (a3); g3 p joins with a3's point (same genotypes and HV: no
    # item); g4 E joins the clean archive (satellite-only); nothing changes afterwards
    assert items_of(view, 'archive_noisy') == [(0, 0, None, True), (2, 1, None, True)]
    assert items_of(view, 'archive_noisy_vs_clean') == [(0, 0, 0, True), (2, 1, 0, True), (4, 1, 1, False)]
    assert items_of(view, 'archive_noisy_vs_clean', 'true') == [(0, 0, 0, True), (2, 0, 1, False), (4, 1, 1, True)]


def test_items_carry_the_evaluations_completed_by_their_generation(run):
    """evals is the evaluation count at the end of the item's generation: g0 evaluated a, a2, b, c, d; g2 adds
    a3, g3 p and g4 e (the crafted run's other generations evaluate nothing)."""
    lab, view = run
    assert view.evals_through(0) == 5
    for view_id in STN_VIEWS:
        for pov in ('algo', 'true'):
            items = stn_view_entries(view, view_id, pov)
            assert all(i['evals'] == view.evals_through(i['gen_idx']) for i in items)
    assert [(i['gen_idx'], i['evals']) for i in stn_view_entries(view, 'population_noisy_vs_clean')] == \
        [(0, 5), (2, 6), (4, 8), (5, 8), (6, 8), (7, 8)]


def test_single_front_views_ignore_the_point_of_view(run):
    _, view = run
    for view_id in ('population_noisy', 'archive_noisy'):
        assert stn_view_entries(view, view_id, 'true') is stn_view_entries(view, view_id, 'algo')
    assert stn_view_entries(view) is stn_view_entries(view, 'population_noisy', 'algo')


def _reference_items(view, view_id, pov):
    """The same rule applied at every generation, rather than at the candidate change points."""
    primary, satellite = stn_view_fronts(view_id, pov)
    items = stn_state_items(
        (gen, stn_front_state(view, primary, gen), None if satellite is None else stn_front_state(view, satellite, gen))
        for gen in range(view.n_generations))
    return [{**item, 'evals': view.evals_through(item['gen_idx'])} for item in items]


def _runs():
    crafted_view = crafted()[1]
    real = []
    for name, prob in (("NSGA2", "kp1"), ("NSGA2", "prior"), ("SEMO", "kp1"), ("MoUMDA_ParetoArchive", "kp1")):
        algo = build(name, prob=prob)
        step_run(algo)
        real.append(MORunView(algo.eval_log.finalise()))
    return [crafted_view] + real


def test_candidate_change_points_miss_no_state_change():
    """Every view and POV, on the crafted run and on real runs: the items from the recorded change points
    are exactly the items from a generation-by-generation scan (so every genotype-set or HV change of
    either front, including noisy HV changes without a genotype change, is at a candidate generation)."""
    for view in _runs():
        for view_id in STN_VIEWS:
            for pov in ('algo', 'true'):
                assert stn_view_entries(view, view_id, pov) == _reference_items(view, view_id, pov), (view_id, pov)
        for primary, satellite in STN_VIEWS.values():
            assert stn_candidate_generations(view, primary, satellite)[0] == 0


# ------------------------------------------------------------------------------ the pure rule

def state(genotypes, hv, n=None):
    genotypes = tuple(genotypes)
    return STNFrontState(genotypes, [[g] for g in genotypes], hv, len(genotypes) if n is None else n)


def rows(items):
    return [(i['gen_idx'], i['state1'], i['state2'], i['primary_changed'], i['change_idx']) for i in items]


def test_cases_a_to_d_and_revisits():
    P, Q, S, X, Y = state((1, 2), 1.0), state((1, 3), 1.0), state((1, 2), 2.0), state((7,), 5.0), state((8,), 5.0)
    items = stn_state_items([
        (0, P, X),
        (1, P, X),          # D: neither changes
        (2, Q, X),          # A: primary only (same HV, other genotypes)
        (3, Q, Y),          # B: satellite only
        (4, S, Y),          # A: primary only (same genotypes as P, other HV)
        (5, P, X),          # C: both change, both revisit their first state
        (6, state((1, 2), 1.0, n=9), X),   # the observation count is not part of the state
    ])
    assert rows(items) == [(0, 0, 0, True, 0), (2, 1, 0, True, 1), (3, 1, 1, False, 2), (4, 2, 1, True, 3),
                           (5, 0, 0, True, 4)]
    assert items[3]['front1'] == items[0]['front1'] and items[3]['metric1'] != items[0]['metric1']


def test_nan_hv_is_one_value_and_single_front_items_have_no_satellite():
    nan = float('nan')
    items = stn_state_items([(0, state((1,), nan), None), (1, state((1,), nan), None), (2, state((2,), nan), None),
                             (3, state((1,), nan), None)])
    assert rows(items) == [(0, 0, None, True, 0), (2, 1, None, True, 1), (3, 0, None, True, 2)]
    assert all(math.isnan(i['metric1']) and i['front2'] is None and i['metric2'] is None for i in items)
    assert stn_state_items([]) == []
