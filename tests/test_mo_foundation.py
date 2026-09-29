"""MO correctness foundation (MO recording plan v6, Work Group 1).

Pins the rules every later MO recording change must preserve:

    generation 0        the evaluated initial population of every MO algorithm
    current fronts      ND(current population)
    no_improvement      stability of the genotype set of noisy ND(current population), counted from g0
    MoUMDA convergence  the probability-vector / support logic, unchanged and independent
    zero-noise MO evaluators consume no RNG

Phase 0 first pinned the behaviour Work Group 1 changed: zero-noise evaluators drawing RNG, legacy
recording perturbing the search, MoUMDA_ParetoArchive recording and stagnating on its internal
archive, and only NSGA-II observing generation 0. Each pin was flipped into the invariant below by the
phase that deliberately changed it (Phase 1, 2b and 2c); their docstrings say what the old behaviour was.

Runs in-process and writes nothing, like test_mo_umda_kmeans.py.
"""

from __future__ import annotations

import functools
import random
from unittest import mock

import numpy as np
import pytest

from deap import creator, tools

from noisyvis.algorithms.multi_objective import (
    MoUMDA,
    MoUMDA_KMeans,
    MoUMDA_noDuplicates,
    MoUMDA_ParetoArchive,
    NSGA2,
    SEMO,
    front_sig,
)
from noisyvis.algorithms.multi_objective import base
from noisyvis.algorithms.operators import binary_attribute
from noisyvis.problems import (
    countingOnesCountingZeros,
    eval_noisy_kp_v1_mo,
    eval_noisy_kp_v1_mo_violation,
    load_problem_KP,
)

WEIGHTS = (1.0, -1.0)
KP_PID = "f1_l-d_kp_10_269"


@functools.lru_cache(maxsize=None)
def knapsack():
    n_items, capacity, _, _, _, items_dict, _ = load_problem_KP(KP_PID)
    return n_items, float(capacity), items_dict


def rng_state():
    return (random.getstate(), repr(np.random.get_state()))


class Counting:
    """Wraps a fitness function, counts its calls and keeps every (genotype, returned vector)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        self.log = []

    def __call__(self, individual, **kwargs):
        self.calls += 1
        result = self.fn(individual, **kwargs)
        self.log.append((tuple(individual), tuple(result)))
        return result


# ------------------------------------------------------------------------------ algorithms

ALGORITHMS = {
    "SEMO": (SEMO, {}),
    "NSGA2": (NSGA2, {"pop_size": 12, "cxpb": 0.9, "mutpb": 0.2, "mate_op": "deap.tools.cxTwoPoint",
                      "mutate_op": "deap.tools.mutFlipBit", "mutate_params": {"indpb": 0.1}}),
    "MoUMDA": (MoUMDA, {"pop_size": 12, "select_size": 6}),
    "MoUMDA_prevent_duplicates": (MoUMDA, {"pop_size": 12, "select_size": 6, "prevent_duplicates": True}),
    "MoUMDA_noDuplicates": (MoUMDA_noDuplicates, {"pop_size": 12, "select_size": 6}),
    "MoUMDA_ParetoArchive": (MoUMDA_ParetoArchive, {"pop_size": 12, "select_size": 6}),
    "MoUMDA_KMeans": (MoUMDA_KMeans, {"pop_size": 12, "select_size": 6}),
}
POPULATION_SIZE = {"SEMO": 1}  # every other case has pop_size 12


def build(name, noise=1.0, seed=1, **params):
    """Construct one MO algorithm on the 10-item knapsack exactly as the MO runner would."""
    cls, init_args = ALGORITHMS[name]
    n_items, capacity, items_dict = knapsack()
    fit_params = {"items_dict": items_dict, "capacity": capacity, "noise_intensity": noise}
    kwargs = dict(
        sol_length=n_items,
        opt_weights=WEIGHTS,
        attr_function=binary_attribute,
        fitness_function=(Counting(eval_noisy_kp_v1_mo), fit_params),
        ref_point=[0.0, 2 * sum(float(v[1]) for v in items_dict.values())],
        gen_limit=10,
    )
    kwargs.update(params)
    random.seed(seed)
    np.random.seed(seed)
    algo = cls(**init_args, **kwargs)
    algo.test_true_fitness = (eval_noisy_kp_v1_mo, dict(fit_params, noise_intensity=0))  # the test's f(x) oracle
    return algo


def genotype(ind):
    return tuple(int(x) for x in ind)


def dominates(a, b):
    """Weak Pareto dominance on maximised (weighted) values, as deap.base.Fitness.dominates."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def brute_force_nd(points):
    """Indices of the non-dominated entries of a list of maximised value tuples."""
    return [i for i, p in enumerate(points)
            if not any(dominates(q, p) for j, q in enumerate(points) if j != i)]


def noisy_front(individuals):
    """(genotype, observed values) of the noisy non-dominated members."""
    keep = brute_force_nd([ind.fitness.wvalues for ind in individuals])
    return {(genotype(individuals[i]), tuple(individuals[i].fitness.values)) for i in keep}


def noisy_signature(individuals):
    """The stagnation signature: the genotype set of the noisy non-dominated members."""
    return frozenset(g for g, _ in noisy_front(individuals))


def clean_front(individuals, algo):
    """Genotypes of the unique members that are non-dominated under the true objectives."""
    tf, tf_kwargs = algo.test_true_fitness
    unique = sorted({genotype(ind) for ind in individuals})
    values = [tuple(v * w for v, w in zip(tf(list(g), **tf_kwargs), WEIGHTS)) for g in unique]
    return {unique[i] for i in brute_force_nd(values)}


class ObservedSources:
    """Captures, at each observation, the individuals the algorithm observes (its stagnation source) and
    the stagnation counter after the update."""

    def __init__(self):
        self.sources = []
        self.objects = []
        self.counters = []

    def __enter__(self):
        original = base.OptimisationAlgorithm._update_front_stagnation
        recorder = self

        def spy(algo, population):
            recorder.objects.append(population)
            recorder.sources.append([creator.Individual(list(ind)) for ind in population])
            for copy, ind in zip(recorder.sources[-1], population):
                copy.fitness.values = ind.fitness.values
            original(algo, population)
            recorder.counters.append(algo._front_unchanged_gens)

        self._patch = mock.patch.object(base.OptimisationAlgorithm, "_update_front_stagnation", spy)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def logged_noisy_front(log, gen):
    """(genotype, observed values) of the evaluation log's current noisy front at a generation."""
    return {(tuple(int(v) for v in log.genotypes[log.obs_orig_geno(o)]), tuple(log.obs_observed(o)))
            for o in log.noisy_front_at(gen)}


def logged_clean_front(log, gen):
    return {tuple(int(v) for v in log.genotypes[g]) for g in log.clean_front_at(gen)}


def counters_of(signatures):
    """The no_improvement counter after each observation of a sequence of signatures."""
    counters = []
    for k, sig in enumerate(signatures):
        counters.append(counters[-1] + 1 if k and sig == signatures[k - 1] else 1)
    return counters


# ------------------------------------------------------------------------------ zero-noise evaluators

def evaluator_cases():
    n_items, capacity, items_dict = knapsack()
    kp = {"items_dict": items_dict, "capacity": capacity}
    feasible = [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    infeasible = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    cases = []
    for fn in (eval_noisy_kp_v1_mo, eval_noisy_kp_v1_mo_violation):
        for bits in (feasible, infeasible):
            for noisy_objective in (0, 1, 2, 3):
                for penalty in (0, 1):
                    cases.append((fn, bits, dict(kp, noisy_objective=noisy_objective, penalty=penalty)))
    for noisy_objective in (0, 1, 2, 3):
        cases.append((countingOnesCountingZeros, [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                      {"noisy_objective": noisy_objective}))
    return cases


EVALUATOR_CASES = evaluator_cases()
EVALUATOR_IDS = [f"{fn.__name__}-{'feasible' if bits[1] == 0 else 'infeasible'}-no{kw['noisy_objective']}"
                 f"-p{kw.get('penalty', 0)}" for fn, bits, kw in EVALUATOR_CASES]


def draws_for(noisy_objective):
    """How many Gaussian draws a noisy MO evaluation makes: objective 1, objective 2, or both."""
    return {0: 2, 1: 1, 2: 1}.get(noisy_objective, 0)


def reference_evaluation(fn, bits, noise_intensity, kwargs):
    """The objective vector the evaluator computed before the zero-noise fix: random.gauss(0, sigma)
    drawn for objective 1 then objective 2 (whichever noisy_objective selects), whatever sigma is."""
    noisy_objective = kwargs["noisy_objective"]
    if fn is countingOnesCountingZeros:
        sigma = noise_intensity
    else:
        items = kwargs["items_dict"]
        sigma = noise_intensity * (sum(w for _, w in items.values()) / len(items))
    noise1 = random.gauss(0, sigma) if noisy_objective in (0, 1) else 0
    noise2 = random.gauss(0, sigma) if noisy_objective in (0, 2) else 0
    if fn is countingOnesCountingZeros:
        return (sum(bits) + noise1, len(bits) - sum(bits) + noise2)
    items, capacity, penalty = kwargs["items_dict"], kwargs["capacity"], kwargs["penalty"]
    weight = sum(items[i][1] * bits[i] for i in range(len(bits))) + noise2
    value = sum(items[i][0] * bits[i] for i in range(len(bits))) + noise1
    second = weight if fn is eval_noisy_kp_v1_mo else max(0, float(sum(items[i][1] * bits[i] for i in range(len(bits))) - capacity))
    if weight > capacity and penalty == 1:
        return (capacity - weight, weight)
    return (value, second)


@pytest.mark.parametrize("fn,bits,kwargs", EVALUATOR_CASES, ids=EVALUATOR_IDS)
def test_zero_noise_evaluation_consumes_no_rng(fn, bits, kwargs):
    """At zero noise the MO evaluators draw nothing, from Python or NumPy, and return exactly what they
    returned before the fix (which drew random.gauss(0, 0) and added its 0.0): equal values of the same
    types, compared by repr, so an int where a float was returned (or -0.0 for 0.0) would fail."""
    random.seed(3)
    np.random.seed(3)
    before = rng_state()
    result = fn(list(bits), noise_intensity=0, **kwargs)
    assert rng_state() == before
    expected = reference_evaluation(fn, bits, 0, kwargs)
    assert result == expected
    assert [type(v) for v in result] == [type(v) for v in expected]
    assert repr(result) == repr(expected)


F, I = np.float64, np.int64  # knapsack sums over the loaded instance are NumPy integers
ZERO_NOISE_TYPES = [
    # (evaluator, noisy_objective, exact types at zero noise) for a feasible solution, penalty 0.
    # An objective selected as noisy is a float, as it was when random.gauss(0, 0) (0.0) was added to
    # it; an objective that is not noisy keeps its integer type. Noise never touches the violation,
    # max(0, float(weight - capacity)), which for this feasible solution is the int 0.
    (eval_noisy_kp_v1_mo, 0, (F, F)),
    (eval_noisy_kp_v1_mo, 1, (F, I)),
    (eval_noisy_kp_v1_mo, 2, (I, F)),
    (eval_noisy_kp_v1_mo, 3, (I, I)),
    (eval_noisy_kp_v1_mo_violation, 0, (F, int)),
    (eval_noisy_kp_v1_mo_violation, 1, (F, int)),
    (eval_noisy_kp_v1_mo_violation, 2, (I, int)),
    (eval_noisy_kp_v1_mo_violation, 3, (I, int)),
    (countingOnesCountingZeros, 0, (float, float)),  # Python int counts become Python floats
    (countingOnesCountingZeros, 1, (float, int)),
    (countingOnesCountingZeros, 2, (int, float)),
    (countingOnesCountingZeros, 3, (int, int)),
]


@pytest.mark.parametrize("fn,noisy_objective,types", ZERO_NOISE_TYPES,
                         ids=[f"{fn.__name__}-no{no}" for fn, no, _ in ZERO_NOISE_TYPES])
def test_zero_noise_objective_types(fn, noisy_objective, types):
    if fn is countingOnesCountingZeros:
        result = fn([1, 0, 1, 1, 0], noise_intensity=0, noisy_objective=noisy_objective)
    else:
        n_items, capacity, items_dict = knapsack()
        result = fn([1, 0, 1, 1, 0, 0, 0, 0, 0, 1], items_dict=items_dict, capacity=capacity,
                    noise_intensity=0, noisy_objective=noisy_objective, penalty=0)
    assert tuple(type(v) for v in result) == types, result


@pytest.mark.parametrize("fn,bits,kwargs", EVALUATOR_CASES, ids=EVALUATOR_IDS)
@pytest.mark.parametrize("noise", [0.5, 2.0])
def test_nonzero_noise_evaluation_is_unchanged(fn, bits, kwargs, noise):
    """With noise the evaluators draw exactly as before: the same number of Gaussian draws, in the
    same order, giving bit-identical objective values and the same final Python RNG state."""
    random.seed(5)
    expected = reference_evaluation(fn, bits, noise, kwargs)
    expected_state = random.getstate()
    random.seed(5)
    numpy_before = repr(np.random.get_state())
    result = fn(list(bits), noise_intensity=noise, **kwargs)
    assert result == expected
    assert repr(result) == repr(expected), "same values of the same types"
    assert random.getstate() == expected_state
    assert repr(np.random.get_state()) == numpy_before
    random.seed(5)
    for _ in range(draws_for(kwargs["noisy_objective"])):
        random.gauss(0, 1)
    assert random.getstate() == expected_state, "the draw count per noisy objective is unchanged"


# ------------------------------------------------------------------------------ lifecycle


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_initial_population_is_evaluated_at_construction(name):
    """Every MO algorithm has a fully evaluated initial population before its first update."""
    algo = build(name)
    size = POPULATION_SIZE.get(name, 12)
    assert algo.gens == 0
    assert len(algo.population) == size
    assert algo.evals == size == algo.fitness_function[0].calls
    assert all(ind.fitness.valid for ind in algo.population)


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_observing_the_initial_front_needs_no_evaluation_or_rng(name):
    algo = build(name)
    calls = algo.fitness_function[0].calls
    before = rng_state()
    front = noisy_front(algo.population)
    signature = front_sig([ind for ind in algo.population if (genotype(ind), tuple(ind.fitness.values)) in front])
    assert front and signature
    assert rng_state() == before
    assert algo.fitness_function[0].calls == calls


def population_snapshot(individuals):
    return [(genotype(ind), tuple(ind.fitness.values)) for ind in individuals]


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_constructors_observe_nothing(name):
    """No constructor records or counts anything: generation 0 is observed by run(), for every algorithm."""
    algo = build(name, log_evaluations=True)
    assert algo.eval_log.n_generations == 0
    assert algo._front_unchanged_gens is None and algo._front_signature is None


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_observing_generation_zero_draws_and_evaluates_nothing(name):
    """Observing generation 0 records the constructor's evaluated population with no evaluator call and
    no Python or NumPy RNG draw, and starts the stagnation counter at 1."""
    algo = build(name, log_evaluations=True)
    calls, evals = algo.fitness_function[0].calls, algo.evals
    before = rng_state()
    algo._observe_generation()
    assert rng_state() == before
    assert (algo.fitness_function[0].calls, algo.evals, algo.gens) == (calls, evals, 0)
    assert algo.eval_log.n_generations == 1 and algo.eval_log.gen_last_eval == [evals]
    assert algo._front_unchanged_gens == 1
    assert algo._front_signature == noisy_signature(algo.population)


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_generation_zero_is_the_evaluated_initial_population(name):
    """For every algorithm the first observed state is the population the constructor built and
    evaluated, and every generation 0..gens is observed and recorded."""
    with ObservedSources() as observed:
        algo = build(name, log_evaluations=True, gen_limit=4)
        initial, initial_object = population_snapshot(algo.population), algo.population
        algo.run()
    assert observed.objects[0] is initial_object
    assert population_snapshot(observed.sources[0]) == initial
    assert logged_noisy_front(algo.eval_log, 0) == noisy_front(observed.sources[0])
    assert algo.gens == 4
    assert len(observed.sources) == algo.eval_log.n_generations == 5


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_generation_one_follows_exactly_one_update(name):
    """Generation 1 is the population after one perform_generation(): its evaluations are exactly that
    update's, and it is the second observed state."""
    with ObservedSources() as observed:
        algo = build(name, log_evaluations=True, gen_limit=1)
        calls = algo.fitness_function[0].calls
        algo.run()
    assert algo.gens == 1 and algo.stop_trigger == "gen_limit"
    assert len(observed.objects) == 2 and observed.objects[1] is algo.population
    assert algo.fitness_function[0].calls - calls == algo.evals - calls  # one update's evaluations only
    assert population_snapshot(observed.sources[1]) == population_snapshot(algo.population)
    assert algo.eval_log.gen_last_eval == [calls, algo.evals]


# ------------------------------------------------------------------------------ front source and stagnation


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_recorded_fronts_and_stagnation_follow_the_current_population(name):
    """For every algorithm each observation reads the current population: the recorded noisy front is
    ND(population) under the observed fitness, the recorded clean front is the clean ND of the
    population's unique genotypes, and the no_improvement counter follows the noisy genotype-set
    signature of the population. (Until Phase 2b, MoUMDA_ParetoArchive used its internal archive.)"""
    with ObservedSources() as observed:
        algo = build(name, log_evaluations=True, gen_limit=6)
        algo.run()
    assert observed.objects[-1] is algo.population
    assert len(observed.sources) == algo.eval_log.n_generations == algo.gens + 1
    for gen, source in enumerate(observed.sources):
        assert logged_noisy_front(algo.eval_log, gen) == noisy_front(source)
        assert logged_clean_front(algo.eval_log, gen) == clean_front(source, algo)
    assert observed.counters == counters_of([noisy_signature(s) for s in observed.sources])


def _crafted(bits, values):
    ind = creator.Individual(list(bits))
    ind.fitness.values = values
    return ind


A_BITS, B_BITS, C_BITS = [1] + [0] * 9, [0, 1] + [0] * 8, [0, 0, 1] + [0] * 7


def pareto_archive_fixture():
    """A MoUMDA_ParetoArchive whose population front and internal archive are set by hand."""
    algo = build("MoUMDA_ParetoArchive", noise=0.0)
    algo.population = [_crafted(A_BITS, (10.0, 5.0)), _crafted(B_BITS, (5.0, 5.0))]
    algo.archive = [_crafted(C_BITS, (8.0, 4.0))]
    return algo


def test_pareto_archive_stagnation_follows_its_population_not_its_archive():
    """The generic no_improvement counter of MoUMDA_ParetoArchive follows its population's noisy
    front, exactly as for every other algorithm; its internal archive plays no part. (Until
    Phase 2b it followed the archive, with the opposite outcome in both cases.)

    Case A changes the population front and leaves the archive alone: the counter resets.
    Case B changes the archive and leaves the population front alone: the counter increments."""
    algo = pareto_archive_fixture()
    algo._observe_generation()
    assert algo._front_unchanged_gens == 1

    algo.population = [_crafted(B_BITS, (12.0, 5.0)), _crafted(A_BITS, (10.0, 5.0))]  # B now dominates A
    algo._observe_generation()
    assert algo._front_unchanged_gens == 1, "case A: population front changed, archive did not: reset"

    algo.archive = [_crafted(A_BITS, (9.0, 3.0))]
    algo._observe_generation()
    assert algo._front_unchanged_gens == 2, "case B: archive changed, population front did not: no reset"
    assert algo._front_signature == {tuple(B_BITS)}, "the signature is the population's front"


def test_pareto_archive_has_no_generic_observation_override():
    """MoUMDA_ParetoArchive observes, records and stagnates through the shared base methods."""
    for method in ("_observe_generation", "_update_front_stagnation", "stop_condition"):
        assert method not in vars(MoUMDA_ParetoArchive), method
    assert MoUMDA_ParetoArchive._observe_generation is base.OptimisationAlgorithm._observe_generation


def test_observing_leaves_the_pareto_archive_untouched():
    """Observation and recording read the population only: the internal archive object and every
    member (genotype and fitness) are unchanged, and no RNG is drawn."""
    algo = build("MoUMDA_ParetoArchive", noise=1.0, gen_limit=3)
    algo.run()
    archive = algo.archive
    snapshot = [(genotype(ind), ind.fitness.values) for ind in archive]
    before = rng_state()
    algo._observe_generation()
    assert algo.archive is archive
    assert [(genotype(ind), ind.fitness.values) for ind in algo.archive] == snapshot
    assert rng_state() == before


def test_pareto_archive_records_its_population_where_it_differs_from_the_archive():
    """In a real run the internal archive's noisy front and the population's noisy front differ, and
    the recorded front is always the population's."""
    fronts = []

    def observe(self):
        fronts.append((noisy_front(self.population), noisy_front(self.archive)))
        base.OptimisationAlgorithm._observe_generation(self)
        fronts[-1] += (logged_noisy_front(self.eval_log, self.gens),)

    with mock.patch.object(MoUMDA_ParetoArchive, "_observe_generation", observe, create=True):
        algo = build("MoUMDA_ParetoArchive", noise=1.0, gen_limit=10, log_evaluations=True)
        algo.run()
    assert fronts and all(recorded == population for population, _, recorded in fronts)
    assert any(population != archive for population, archive, _ in fronts)


def test_stop_condition_reads_the_algorithm_owned_counter():
    """no_improvement fires when the algorithm's own counter reaches the limit."""
    algo = build("MoUMDA", stop_without_improvement_in_gens=3)
    assert algo._front_unchanged_gens is None and algo.stop_condition() is False
    algo._front_unchanged_gens = 2
    assert algo.stop_condition() is False and algo.stop_trigger == ""
    algo._front_unchanged_gens = 3
    assert algo.stop_condition() is True and algo.stop_trigger == "no_improvement"


def frozen_nsga2(**params):
    """NSGA-II without crossover or mutation: offspring are unevaluated clones, so the front never changes."""
    algo = build("NSGA2", **params)
    algo.cxpb = 0.0
    algo.mutpb = 0.0
    return algo


def test_frozen_front_stops_after_limit_observations():
    """With a front that never changes, no_improvement fires once the counter reaches L. Generation 0
    is the first observation, so the run stops after generation L - 1."""
    algo = frozen_nsga2(stop_without_improvement_in_gens=4, gen_limit=50)
    algo.run()
    assert (algo.gens, algo.stop_trigger) == (3, "no_improvement")
    assert algo._front_unchanged_gens == 4


def test_generation_zero_counts_for_no_improvement_in_every_algorithm():
    """A MoUMDA_KMeans population frozen from generation 0 (one genotype, noise-free objectives) now
    stops after generation L - 1, as NSGA-II always did; before generation 0 was universal it ran L."""
    algo = build("MoUMDA_KMeans", noise=0.0, stop_without_improvement_in_gens=4, gen_limit=50)
    for ind in algo.population:
        ind[:] = A_BITS
        ind.fitness.values = algo.toolbox.evaluate(ind)
    algo.run()
    assert (algo.gens, algo.stop_trigger) == (3, "no_improvement")
    assert algo._front_unchanged_gens == 4


def test_a_front_that_changes_after_generation_zero_stops_as_before():
    """Once the front changes, the counter is the same as without a generation-0 observation: a run
    whose g1 front differs from its g0 front stops exactly L observations after that change."""
    signatures = [frozenset({tuple(A_BITS)}), frozenset({tuple(B_BITS)}), frozenset({tuple(B_BITS)}),
                  frozenset({tuple(B_BITS)})]

    def counters(observed):
        algo = build("MoUMDA", noise=0.0)
        found = []
        for signature in observed:
            algo.population = [_crafted(list(g), (10.0, 5.0)) for g in signature]
            algo._update_front_stagnation(algo.population)
            found.append(algo._front_unchanged_gens)
        return found

    with_generation_zero = counters(signatures)
    without_generation_zero = counters(signatures[1:])
    assert with_generation_zero == [1, 1, 2, 3]
    assert with_generation_zero[1:] == without_generation_zero


# ------------------------------------------------------------------------------ algorithm-owned stagnation


def observing(name):
    """The algorithm class, checking after every observation that the algorithm-owned stagnation
    state agrees with the legacy definition: the signature is front_sig(ParetoFront(source)), computed
    independently with deap, and the counter is the run length of equal consecutive signatures."""
    cls = ALGORITHMS[name][0]

    def _update_front_stagnation(self, source):
        super(checked, self)._update_front_stagnation(source)
        pareto_front = tools.ParetoFront()
        pareto_front.update(source)
        expected = front_sig(list(pareto_front))
        self.checks.append((self._front_signature == expected, source is self.population))
        self.signatures.append(expected)
        self.checks.append(self._front_unchanged_gens == counters_of(self.signatures)[-1])

    # a fresh class per call, so its `checks` list (filled from the constructor on) is per test
    checked = type(f"Checked{cls.__name__}", (cls,), {"_update_front_stagnation": _update_front_stagnation,
                                                     "checks": [], "signatures": []})
    return checked


@pytest.mark.parametrize("noise", [0.0, 1.0])
@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_algorithm_owned_stagnation_matches_the_legacy_definition(name, noise):
    cls, init_args = ALGORITHMS[name]
    ALGORITHMS[name] = (observing(name), init_args)
    try:
        algo = build(name, noise=noise, gen_limit=25)
    finally:
        ALGORITHMS[name] = (cls, init_args)
    algo.run()
    signature_checks = [c for c in algo.checks if isinstance(c, tuple)]
    counter_checks = [c for c in algo.checks if not isinstance(c, tuple)]
    assert len(signature_checks) == len(counter_checks) == algo.gens + 1 > 1
    assert all(ok for ok, _ in signature_checks)
    assert all(counter_checks)
    assert all(is_population for _, is_population in signature_checks), "the source is always the population"


@pytest.mark.parametrize("noise", [0.0, 1.0])
@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_no_improvement_stop_is_independent_of_recording(name, noise):
    """Stopping reads only algorithm state: with recording (the evaluation log) switched on or off, the
    same solutions are evaluated and the run stops at the same generation with the same trigger."""
    def run(recording):
        algo = build(name, noise=noise, gen_limit=200, stop_without_improvement_in_gens=3,
                     log_evaluations=recording)
        algo.run()
        return (algo.fitness_function[0].log, algo.gens, algo.evals, algo.stop_trigger,
                algo._front_unchanged_gens, rng_state())

    recorded, unrecorded = run(True), run(False)
    assert recorded == unrecorded


# ------------------------------------------------------------------------------ progress reporting


def test_progress_line_is_algorithm_owned_and_changes_nothing(capsys):
    """verbose_rate = k prints one line every k generations (0 prints nothing), from the algorithm's own
    state: the generation, cumulative evaluations, the current noisy front's members and distinct
    genotypes, and whether its genotype set changed. It needs no recorder, and the run (evaluations,
    stop, stagnation, both RNGs) is identical with it on or off."""
    def run(verbose_rate):
        algo = build("MoUMDA", noise=1.0, gen_limit=6, verbose_rate=verbose_rate)
        algo.run()
        return algo, (algo.fitness_function[0].log, algo.gens, algo.evals, algo.stop_trigger,
                      algo._front_signature, algo._front_unchanged_gens, rng_state())

    quiet, quiet_state = run(0)
    assert capsys.readouterr().out == ""
    loud, loud_state = run(2)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[SeedSig")]
    assert loud_state == quiet_state and loud.eval_log is None
    assert [int(line.split("gen ")[1].split(" ")[0]) for line in lines] == [0, 2, 4, 6]
    last = lines[-1]
    assert f"evals {loud.evals}" in last
    assert f"noisy front {loud._front_size} members, {len(loud._front_signature)} genotypes" in last
    expected = "changed" if loud._front_unchanged_gens == 1 else f"unchanged for {loud._front_unchanged_gens}"
    assert last.endswith(expected)
    assert loud._front_size == len(noisy_front(loud.population))
