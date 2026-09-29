"""MO evaluation logging and provenance (MO recording plan v6, Work Group 2).

Pins the evaluator-level event stream and its provenance tags:

    every genuine algorithm evaluation      exactly one event (x, x~, f(x), y), initial population included
    nothing else                            no event for the legacy recorder's clean re-evaluations,
                                            for duplicates rejected before evaluation, or for any extra
                                            evaluation (there is none)
    x vs x~                                 kept apart: posterior noise x~ == x, prior noise may differ
    f(x)                                    computed during the genuine evaluation, equal in value and type
                                            to the clean evaluator; zero extra evaluations, no RNG
    genotype interning                      one id per genotype; interning order is not discovery order
    canonical observation ids               exact (x, x~, y) twins share one; any difference separates
    ind._eval_tag                           (eval_id, orig_geno_id, wvalues): survives clone, replaced on
                                            re-evaluation, validated on resolve, inert for deap
    non-interference                        logging on vs off: identical search, state and RNG streams

The reference-#3 baselines are also re-run with logging on, without being re-recorded, in
test_mo_algorithms.py (characterisation_logged) and test_reproducibility.py (mo.json).

Runs in-process and writes nothing.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from deap import creator, tools

from harness.mo_runs import (
    ALGORITHM_NAMES,
    COCZ_WEIGHTS,
    KP_WEIGHTS,
    ScriptedPriorCOCZ,
    build,
    check_exactly_one_log,
    cocz_objectives,
    knapsack,
    kp_objectives,
    rng_state,
    step_run,
)
from noisyvis.problems import (
    countingOnesCountingZeros,
    eval_noisy_kp_v1_mo,
    eval_noisy_kp_v1_mo_violation,
)
from noisyvis.tracking.logger import clear_active_logger, get_active_logger, set_active_logger
from noisyvis.tracking.mo_logger import (
    EvaluationLogError,
    MOEvaluationLogger,
    ProvenanceError,
)


@pytest.fixture(autouse=True)
def no_active_logger():
    """Every test starts and ends with no active logger, so a leak fails the leaking test."""
    clear_active_logger()
    yield
    assert get_active_logger() is None, "an active logger leaked out of the test"


# ------------------------------------------------------------------------------ logger units

def test_same_genotype_same_id_and_distinct_genotypes_never_collide():
    log = MOEvaluationLogger(KP_WEIGHTS)
    a, b = log.intern([1, 0, 1]), log.intern([0, 1, 1])
    assert a != b
    assert log.intern((1, 0, 1)) == a and log.intern([0, 1, 1]) == b
    assert log.genotypes == [(1, 0, 1), (0, 1, 1)]
    # the key is the raw gene tuple, not int()-cast: real-valued genes keep their values
    c, d = log.intern([0.1, 0.25, 1.5]), log.intern([0.1, 0.2500001, 1.5])
    assert len({a, b, c, d}) == 4
    assert log.intern([0.4, 0.9, 0.0]) != log.intern([0.0, 0.0, 0.0])  # int() would merge these


def test_numerically_equal_genes_intern_together():
    """Documented limit: raw tuples compare and hash by numeric value, so 1, 1.0, True and np.int64(1)
    are the same gene. That is intended: they are numerically the same genotype."""
    log = MOEvaluationLogger(KP_WEIGHTS)
    ids = {log.intern([1, 0]), log.intern([1.0, 0.0]), log.intern([True, False]),
           log.intern([np.int64(1), np.int64(0)])}
    assert len(ids) == 1


def test_scope_requires_exactly_one_log_and_commits_nothing_on_failure():
    log = MOEvaluationLogger(KP_WEIGHTS)
    sentinel = object()

    def silent(ind):
        return (1.0, 2.0)

    def twice(ind):
        get_active_logger().log_mo_eval(ind, (1.0, 2.0), (1.0, 2.0))
        get_active_logger().log_mo_eval(ind, (1.0, 2.0), (1.0, 2.0))
        return (1.0, 2.0)

    def mismatched(ind):
        get_active_logger().log_mo_eval(ind, (1.0, 2.0), (1.0, 2.0))
        return (1.0, 3.0)

    def mutating(ind):
        get_active_logger().log_mo_eval(ind, (1.0, 2.0), (1.0, 2.0))
        ind[0] = 1 - ind[0]
        return (1.0, 2.0)

    def failing(ind):
        get_active_logger().log_mo_eval(ind, (1.0, 2.0), (1.0, 2.0))
        raise ValueError("evaluator failed")

    set_active_logger(sentinel)
    try:
        for fn, match in ((silent, "logged 0 times"), (twice, "logged 2 times"),
                          (mismatched, "logged y="), (mutating, "modified the individual")):
            with pytest.raises(EvaluationLogError, match=match):
                log.evaluate(fn, [1, 0, 1], {})
            assert get_active_logger() is sentinel, "the previous active logger is restored"
            assert len(log) == 0 and log.genotypes == [] and log._open is None
        with pytest.raises(ValueError, match="evaluator failed"):
            log.evaluate(failing, [1, 0, 1], {})
        assert get_active_logger() is sentinel and len(log) == 0 and log._open is None

        # still usable afterwards
        def good(ind):
            get_active_logger().log_mo_eval(ind, (1.0, 2.0), (1.0, 2.0))
            return (1.0, 2.0)
        assert log.evaluate(good, [1, 0, 1], {}) == ((1.0, 2.0), 0)
        assert get_active_logger() is sentinel
    finally:
        clear_active_logger()


def test_logging_outside_a_genuine_evaluation_fails_loudly():
    log = MOEvaluationLogger(COCZ_WEIGHTS)
    with pytest.raises(EvaluationLogError, match="outside a genuine evaluation"):
        log.log_mo_eval([1, 0], (1, 1), (1, 1))
    # an MO logger made active by hand still refuses an evaluation that is not in its scope
    set_active_logger(log)
    try:
        with pytest.raises(EvaluationLogError, match="outside a genuine evaluation"):
            countingOnesCountingZeros([1, 0, 1], noise_intensity=0)
    finally:
        clear_active_logger()
    assert len(log) == 0


def test_nested_evaluations_are_refused():
    outer, other = MOEvaluationLogger(COCZ_WEIGHTS), MOEvaluationLogger(COCZ_WEIGHTS)

    def nesting(logger):
        def fn(ind):
            logger.evaluate(countingOnesCountingZeros, list(ind), {})
            return countingOnesCountingZeros(ind)
        return fn

    for inner in (outer, other):
        with pytest.raises(EvaluationLogError, match="nested MO evaluation"):
            outer.evaluate(nesting(inner), [1, 0, 1], {})
        assert len(outer) == len(other) == 0
        assert outer._open is None and other._open is None and get_active_logger() is None


def test_an_evaluator_that_does_not_report_cannot_be_logged():
    """log_evaluations=True requires an evaluator that reports through log_mo_eval: a silent one fails
    at the first evaluation (the initial population) instead of producing a log with gaps."""
    with pytest.raises(EvaluationLogError, match="logged 0 times"):
        build("MoUMDA", prob="kp1", fitness_function=(kp_objectives, {"items_dict": knapsack()[2]}))


# ------------------------------------------------------------------------------ exactly one log

@pytest.mark.parametrize("prob", ["kp0", "kp1", "kpviol1", "cocz1", "prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_every_evaluation_is_logged_exactly_once(name, prob):
    """After construction and after every generation: one event per counted evaluation and per
    evaluator call, initial population included; the logged x and y are exactly the evaluator's inputs
    and outputs, in order; every population member resolves to its own event."""
    algo = build(name, prob=prob)
    fitness = algo.fitness_function[0]
    size = 1 if name == "SEMO" else 12
    assert len(algo.eval_log) == algo.evals == fitness.calls == size  # generation 0 included

    step_run(algo, check_exactly_one_log)
    assert algo.gens > 0

    log = algo.eval_log
    assert [log.event(i).x for i in range(len(log))] == [x for x, _ in fitness.log]
    assert [repr(log.event(i).observed) for i in range(len(log))] == [repr(y) for _, y in fitness.log]


@pytest.mark.parametrize("name", ["MoUMDA_prevent_duplicates", "MoUMDA_noDuplicates"])
def test_duplicates_rejected_before_evaluation_are_not_logged(name):
    """Duplicate-free variants reject duplicate candidates before evaluating them: those candidates are
    never evaluated, never counted and never logged, and each population holds distinct genotypes."""
    algo = build(name, prob="kp0")

    def distinct_and_counted(algo):
        check_exactly_one_log(algo)
        assert len({tuple(ind) for ind in algo.population}) == len(algo.population)

    step_run(algo, distinct_and_counted)


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_run_logs_every_evaluation_exactly_once(name):
    algo = build(name, prob="kp1")
    algo.run()
    check_exactly_one_log(algo)


# ------------------------------------------------------------------------------ event semantics

@pytest.mark.parametrize("prob", ["kp0", "kp1", "kpviol1", "cocz1"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_posterior_events_record_x_and_the_clean_objectives(name, prob):
    """Posterior noise: x~ == x, and f(x) equals the clean evaluator (noise_intensity=0) in value and
    type, compared by repr."""
    algo = build(name, prob=prob)
    step_run(algo)
    tf, tf_kwargs = algo.true_fitness_function
    log = algo.eval_log
    for i in range(len(log)):
        event = log.event(i)
        assert event.x_tilde == event.x and event.eval_geno == event.orig_geno
        assert repr(event.true_objectives) == repr(tf(list(event.x), **tf_kwargs)), event


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_prior_events_record_the_evaluated_genotype(name):
    """Pure prior noise: y = f(x~), f(x) is the deterministic value of x, and x~ is exactly the genotype
    the evaluator evaluated. The runs really do produce x~ != x."""
    algo = build(name, prob="prior")
    step_run(algo)
    log, prior = algo.eval_log, algo.fitness_function[0].fn
    items_dict = knapsack()[2]
    assert [log.event(i).x_tilde for i in range(len(log))] == prior.evaluated
    for i in range(len(log)):
        event = log.event(i)
        assert event.observed == kp_objectives(event.x_tilde, items_dict)
        assert event.true_objectives == kp_objectives(event.x, items_dict)
    assert any(log.event(i).x_tilde != log.event(i).x for i in range(len(log)))


def scripted_semo(script=()):
    """A logged SEMO on counting ones / counting zeros with scripted prior noise, for exact cases."""
    fitness = ScriptedPriorCOCZ()
    algo = build("SEMO", prob="cocz1", opt_weights=COCZ_WEIGHTS,
                 fitness_function=(fitness, {}), true_fitness_function=(cocz_objectives, {}))
    fitness.script.extend(script)
    return algo


def evaluated(algo, bits):
    """A genuine evaluation through the algorithm's own toolbox.evaluate, fitness then assigned."""
    ind = creator.Individual(bits)
    ind.fitness.values = algo.toolbox.evaluate(ind)
    return ind


X = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  # bits 2 and 5 are both 0


def test_same_x_same_y_different_x_tilde_are_distinct_events_and_observations():
    """Flipping bit 2 or bit 5 of the same x gives the same y from a different x~: two events, two
    canonical observations. (x, y) is never the provenance key."""
    algo = scripted_semo([[2], [5]])
    first_new = len(algo.eval_log)
    a, b = evaluated(algo, X), evaluated(algo, X)
    log = algo.eval_log
    ea, eb = log.event(first_new), log.event(first_new + 1)
    assert ea.x == eb.x == tuple(X) and ea.observed == eb.observed
    assert ea.x_tilde != eb.x_tilde and ea.x_tilde[2] == 1 and eb.x_tilde[5] == 1
    assert ea.obs_id != eb.obs_id and ea.eval_id != eb.eval_id

    # the retained individuals resolve to their own events, not merely to one matching (x, y)
    assert log.resolve(a) == ea.eval_id and log.resolve(b) == eb.eval_id
    assert log.event(log.resolve(b)).x_tilde == eb.x_tilde
    assert log.resolve(algo.toolbox.clone(b)) == eb.eval_id


def test_interning_is_not_discovery():
    """A genotype first seen as some evaluation's x~ is interned then; it is discovered as a search
    solution only at the first event whose orig_geno it is, which the event columns keep apart."""
    g = list(X)
    g[2] = 1
    algo = scripted_semo([[2]])
    start = len(algo.eval_log)
    evaluated(algo, X)                      # x = X, x~ = g: g interned here, as x~ only
    log = algo.eval_log
    g_id = log.intern(g)
    assert len(log.genotypes) == log.genotypes.index(tuple(g)) + 1
    assert log.eval_geno[start] == g_id and g_id not in log.orig_geno

    evaluated(algo, g)                      # later, g itself is generated as an x
    assert log.intern(g) == g_id, "the same genotype keeps its id"
    first_as_original = log.orig_geno.index(g_id)
    first_seen = min(i for i in range(len(log)) if g_id in (log.orig_geno[i], log.eval_geno[i]))
    assert (first_seen, first_as_original) == (start, start + 1)


def test_canonical_observations_collapse_only_exact_twins():
    # exact (x, x~, y) twins: clean posterior evaluations of the same x
    algo = build("SEMO", prob="kp0")
    start = len(algo.eval_log)
    evaluated(algo, X), evaluated(algo, X)
    log = algo.eval_log
    assert log.obs_id[start] == log.obs_id[start + 1]
    assert log.obs_first_eval[log.obs_id[start + 1]] == start
    assert len(log) == start + 2, "every evaluation keeps its own event"

    # repeated noisy evaluations of the same x: different y, different observations
    algo = build("SEMO", prob="kp1")
    start = len(algo.eval_log)
    evaluated(algo, X), evaluated(algo, X)
    log = algo.eval_log
    assert log.observed[start] != log.observed[start + 1]
    assert log.obs_id[start] != log.obs_id[start + 1]


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_clean_posterior_observations_are_one_per_genotype(name):
    """At zero posterior noise y is a function of x, so canonical observations and generated genotypes
    correspond one to one, however often a genotype is re-evaluated."""
    algo = build(name, prob="kp0")
    step_run(algo)
    log = algo.eval_log
    assert len(log.obs_first_eval) == len(set(log.orig_geno))
    assert len({(log.orig_geno[i], log.obs_id[i]) for i in range(len(log))}) == len(set(log.orig_geno))


# ------------------------------------------------------------------------------ zero-noise contract

def evaluator_cases():
    n_items, capacity, items_dict = knapsack()
    kp = {"items_dict": items_dict, "capacity": capacity}
    cases = []
    for fn in (eval_noisy_kp_v1_mo, eval_noisy_kp_v1_mo_violation):
        for bits in ([1, 0, 1, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1, 0, 0]):
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


@pytest.mark.parametrize("noise", [0, 0.5])
@pytest.mark.parametrize("fn,bits,kwargs", EVALUATOR_CASES, ids=EVALUATOR_IDS)
def test_logging_inside_the_evaluator_changes_no_value_type_or_rng(fn, bits, kwargs, noise):
    """With an MO logger recording the call, each evaluator returns exactly what it returns without one
    (repr: values and types) and leaves both RNGs in the same state (untouched at zero noise). The
    logged f(x) equals the clean evaluation in value and type; x~ is x."""
    random.seed(3)
    np.random.seed(3)
    before = rng_state()
    plain = fn(list(bits), noise_intensity=noise, **kwargs)
    plain_state = rng_state()

    random.seed(3)
    np.random.seed(3)
    log = MOEvaluationLogger(KP_WEIGHTS)
    observed, eval_id = log.evaluate(fn, list(bits), dict(kwargs, noise_intensity=noise))
    assert repr(observed) == repr(plain)
    assert rng_state() == plain_state
    if noise == 0:
        assert plain_state == before

    event = log.event(eval_id)
    clean = fn(list(bits), noise_intensity=0, **kwargs)
    assert repr(event.true_objectives) == repr(clean)
    assert repr(event.observed) == repr(plain)
    assert event.x == event.x_tilde == tuple(bits)


# ------------------------------------------------------------------------------ non-interference

def snapshot(algo):
    """Everything the search is: generation state, population, algorithm-specific state, both RNGs."""
    state = {
        "gens": algo.gens,
        "evals": algo.evals,
        "stop_trigger": algo.stop_trigger,
        "front_unchanged": (algo._front_signature, algo._front_unchanged_gens),
        "population": [(tuple(ind), repr(ind.fitness.values)) for ind in algo.population],
        "rng": rng_state(),
    }
    if hasattr(algo, "archive"):
        state["archive"] = [(tuple(ind), repr(ind.fitness.values)) for ind in algo.archive]
    for name in ("probability_vector", "_prepared_probability_vector", "cluster_labels", "cluster_centers",
                 "cluster_sizes", "cluster_probability_vectors"):
        if hasattr(algo, name):
            state[name] = repr(getattr(algo, name))
    return state


def legacy_record(algo):
    return repr((algo.n_gens_pareto_best, algo.noisy_pf_noisy_hypervolumes, algo.noisy_pf_true_hypervolumes,
                 algo.true_pf_hypervolumes, algo.pareto_fitnesses, algo.pareto_true_fitnesses,
                 algo.true_pareto_fitnesses, [[tuple(i) for i in f] for f in algo.pareto_solutions],
                 [[tuple(i) for i in f] for f in algo.true_pareto_solutions]))


def trace(name, prob, log):
    algo = build(name, prob=prob, log=log)
    states = [snapshot(algo)]  # after construction: the initial evaluations
    step_run(algo, lambda algo: states.append(snapshot(algo)))
    fitness = algo.fitness_function[0]
    evaluated_x_tilde = getattr(fitness.fn, "evaluated", None)
    return {"states": states, "calls": [(x, repr(y)) for x, y in fitness.log],
            "x_tilde": evaluated_x_tilde, "legacy": legacy_record(algo), "algo": algo}


@pytest.mark.parametrize("prob", ["kp0", "kp1", "prior"])
@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_logging_on_and_off_give_the_identical_search(name, prob):
    """With logging on vs off (same seed and config): the generated x, the evaluated x~ and the returned
    y sequences; the populations; the PA archive; the probability vectors; the KMeans state;
    gens/evals/stop; the stagnation state; the legacy recorder's output; and the Python and NumPy RNG
    states after construction and after every generation are all identical."""
    on, off = trace(name, prob, True), trace(name, prob, False)
    assert off["algo"].eval_log is None and on["algo"].eval_log is not None
    assert on["calls"] == off["calls"]
    assert on["x_tilde"] == off["x_tilde"]
    assert len(on["states"]) == len(off["states"])
    for got, want in zip(on["states"], off["states"]):
        assert got == want, f"generation {want['gens']} differs with logging on"
    assert on["legacy"] == off["legacy"]
    if prob == "prior":
        assert on["x_tilde"] and any(x != xt for (x, _), xt in zip(on["calls"], on["x_tilde"]))


def test_disabled_logging_is_the_plain_evaluation_path():
    algo = build("NSGA2", prob="kp1", log=False)
    step_run(algo)
    assert algo.eval_log is None
    assert not any(hasattr(ind, "_eval_tag") for ind in algo.population)


def test_interleaved_runs_keep_separate_logs():
    """Two logged runs stepped alternately: each log holds exactly its own run's evaluations."""
    a, b = build("MoUMDA", prob="kp1", seed=1), build("NSGA2", prob="kp1", seed=2)
    a._observe_generation(), b._observe_generation()
    for _ in range(5):
        for algo in (a, b):
            algo.gens += 1
            algo.perform_generation()
            algo._observe_generation()
            check_exactly_one_log(algo)
    for algo in (a, b):
        log, fitness = algo.eval_log, algo.fitness_function[0]
        assert [log.event(i).x for i in range(len(log))] == [x for x, _ in fitness.log]


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_legacy_recorder_clean_evaluations_are_not_logged(name):
    """The legacy recorder re-evaluates fronts cleanly through true_fitness_function, outside
    toolbox.evaluate. Those calls see no active MO logger and add no event: the log holds exactly the
    algorithm's counted evaluations."""
    algo = build(name, prob="kp1")
    tf, tf_kwargs = algo.true_fitness_function
    seen = []

    def spy(ind, **kwargs):
        seen.append(get_active_logger())
        return tf(ind, **kwargs)

    algo.true_fitness_function = (spy, tf_kwargs)
    step_run(algo, check_exactly_one_log)
    assert seen, "the legacy recorder made clean evaluations"
    assert all(active is None for active in seen)


# ------------------------------------------------------------------------------ _eval_tag

def test_tag_lifecycle_evaluate_commit_tag_then_assign():
    """evaluator returns y -> event committed -> tag attached from y -> caller assigns fitness.values
    = y -> resolve() validates. Between tagging and assignment the individual has a tag but not yet
    that fitness; that is intended, and only resolve() checks it."""
    algo = build("SEMO", prob="kp1")
    ind = creator.Individual(X)
    y = algo.toolbox.evaluate(ind)
    eval_id = len(algo.eval_log) - 1
    assert algo.eval_log.observed[eval_id] == tuple(y)
    assert ind._eval_tag == (eval_id, algo.eval_log.intern(X), tuple(v * w for v, w in zip(y, KP_WEIGHTS)))
    assert not ind.fitness.valid
    with pytest.raises(ProvenanceError, match="fitness changed"):
        algo.eval_log.resolve(ind)
    ind.fitness.values = y
    assert ind._eval_tag[2] == ind.fitness.wvalues
    assert algo.eval_log.resolve(ind) == eval_id


def test_clone_keeps_the_tag_and_reevaluation_replaces_it():
    algo = build("SEMO", prob="kp1")
    parent = evaluated(algo, X)
    parent_id = algo.eval_log.resolve(parent)

    clone = algo.toolbox.clone(parent)
    assert clone._eval_tag == parent._eval_tag
    assert algo.eval_log.resolve(clone) == parent_id, "an unmodified clone is the parent's evaluation"

    clone[0] = 1 - clone[0]
    del clone.fitness.values
    clone.fitness.values = algo.toolbox.evaluate(clone)
    clone_id = algo.eval_log.resolve(clone)
    assert clone_id == len(algo.eval_log) - 1 != parent_id
    assert algo.eval_log.event(clone_id).x == tuple(clone)
    assert algo.eval_log.resolve(parent) == parent_id


def test_stale_genotype_or_fitness_is_a_provenance_error():
    algo = build("SEMO", prob="kp1")

    ind = evaluated(algo, X)
    ind[0] = 1 - ind[0]  # genotype changed without re-evaluation
    with pytest.raises(ProvenanceError, match="genotype changed"):
        algo.eval_log.resolve(ind)

    ind = evaluated(algo, X)
    ind.fitness.values = (1.0, 2.0)  # fitness assigned without evaluation
    with pytest.raises(ProvenanceError, match="fitness changed"):
        algo.eval_log.resolve(ind)

    ind = evaluated(algo, X)
    del ind.fitness.values
    with pytest.raises(ProvenanceError, match="fitness changed"):
        algo.eval_log.resolve(ind)

    with pytest.raises(ProvenanceError, match="no _eval_tag"):
        algo.eval_log.resolve(creator.Individual(X))


def test_corrupt_or_inconsistent_tags_are_rejected():
    """A tag that disagrees with this log is rejected. The tag carries no run identity, so a tag from
    another run is only caught when it disagrees; individuals are resolved against their own run's log."""
    algo = build("SEMO", prob="kp1")
    ind = evaluated(algo, X)
    eval_id, orig_geno, wvalues = ind._eval_tag
    other_geno = algo.eval_log.intern([1] * 10)

    for tag, match in (((len(algo.eval_log), orig_geno, wvalues), "does not match this run"),
                       ((-1, orig_geno, wvalues), "does not match this run"),
                       ((eval_id, other_geno, wvalues), "does not match this run"),
                       ((eval_id, orig_geno, (0.0, 0.0)), "do not match evaluation")):
        ind._eval_tag = tag
        with pytest.raises(ProvenanceError, match=match):
            algo.eval_log.resolve(ind)

    # a tag from another run's log that disagrees with this one
    other = build("SEMO", prob="kp1", seed=7)
    foreign = evaluated(other, [0, 1] * 5)
    with pytest.raises(ProvenanceError):
        algo.eval_log.resolve(foreign)


def test_unmodified_nsga2_clones_resolve_to_their_parents_evaluation():
    """NSGA-II clones its tournament winners; a clone that is neither crossed nor mutated keeps its
    fitness and is never evaluated. With cxpb = mutpb = 0 every offspring is such a clone: no new
    evaluations happen, yet every survivor, clone objects included, resolves to an initial evaluation."""
    algo = build("NSGA2", prob="kp1", cxpb=0.0, mutpb=0.0)
    initial = {id(ind) for ind in algo.population}
    for _ in range(3):
        algo.gens += 1
        algo.perform_generation()
        check_exactly_one_log(algo)
    assert algo.evals == len(algo.eval_log) == 12
    clones = [ind for ind in algo.population if id(ind) not in initial]
    assert clones, "the survivors include never-evaluated clones"
    for ind in algo.population:
        assert algo.eval_log.resolve(ind) < 12


def tagged_and_untagged_population(seed=11):
    """Two equal populations (same genotypes and fitness, including duplicates and ties), one carrying
    arbitrary tags and one carrying none."""
    rng = random.Random(seed)
    build("SEMO", prob="kp1", log=False)  # ensures creator.Individual has the KP weights
    rows = [([rng.randint(0, 1) for _ in range(10)], (float(rng.randint(0, 5)), float(rng.randint(0, 5))))
            for _ in range(10)]
    rows += rows[:4]  # exact duplicates
    pops = []
    for tagged in (False, True):
        pop = []
        for k, (bits, values) in enumerate(rows):
            ind = creator.Individual(bits)
            ind.fitness.values = values
            if tagged:
                ind._eval_tag = (1000 - k, k * 7, (-1.0, 99.0))
            pop.append(ind)
        pops.append(pop)
    return pops


def test_tag_has_no_effect_on_deap():
    """Dominance, equality, ParetoFront, sortNondominated, selNSGA2 and selTournamentDCD give identical
    results (and consume identical RNG) with and without tags."""
    plain, tagged = tagged_and_untagged_population()
    index = [{id(ind): i for i, ind in enumerate(pop)} for pop in (plain, tagged)]

    def positions(pop_index, selected):
        return [index[pop_index][id(ind)] for ind in selected]

    assert [[a.fitness.dominates(b.fitness) for b in plain] for a in plain] == \
        [[a.fitness.dominates(b.fitness) for b in tagged] for a in tagged]
    assert all(p == t for p, t in zip(plain, tagged))

    fronts = []
    for pop in (plain, tagged):
        pf = tools.ParetoFront()
        pf.update(pop)
        fronts.append([(tuple(ind), ind.fitness.values) for ind in pf])
    assert fronts[0] == fronts[1]

    assert [positions(0, f) for f in tools.sortNondominated(plain, len(plain))] == \
        [positions(1, f) for f in tools.sortNondominated(tagged, len(tagged))]

    assert positions(0, tools.selNSGA2(plain, 6)) == positions(1, tools.selNSGA2(tagged, 6))

    results = []
    for k, pop in enumerate((plain, tagged)):
        survivors = tools.selNSGA2(pop, 8)  # assigns crowding distances
        random.seed(5)
        results.append((positions(k, tools.selTournamentDCD(survivors, 8)), random.getstate()))
    assert results[0] == results[1]
