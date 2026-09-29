"""Shared fixtures for the in-process MO evaluation-logging tests (MO recording plan v6, Work Groups 2 and 3).

    knapsack / problem / build   construct one MO algorithm on a 10-bit problem as the MO runner would,
                                 with evaluation logging on or off
    Counting                     wraps a fitness function and keeps every (genotype, returned vector)
    PriorBitflipKP, ScriptedPriorCOCZ
                                 test-only prior-noise evaluators (x~ may differ from x)
    step_run                     OptimisationAlgorithm.run(), step by step, with an observer
    check_exactly_one_log        one event per counted evaluation; every population member resolves
    Lab                          crafted evaluations (scripted x~, f(x), y) committed through the real
                                 MOEvaluationLogger onto real Individuals, for exact constructed cases
    direct_hv                    hypervolume written out independently, with the legacy convention

Imported by test modules only; nothing here runs at import time except the imports.
"""

from __future__ import annotations

import functools
import random

import numpy as np

from deap import creator
from deap.tools._hypervolume import hv

from noisyvis.algorithms.multi_objective import (
    MoUMDA,
    MoUMDA_KMeans,
    MoUMDA_noDuplicates,
    MoUMDA_ParetoArchive,
    NSGA2,
    SEMO,
)
from noisyvis.algorithms.operators import binary_attribute
from noisyvis.problems import (
    countingOnesCountingZeros,
    eval_noisy_kp_v1_mo,
    eval_noisy_kp_v1_mo_violation,
    load_problem_KP,
)
from noisyvis.tracking.logger import get_active_logger
from noisyvis.tracking.mo_logger import MOEvaluationLogger

KP_WEIGHTS = (1.0, -1.0)
COCZ_WEIGHTS = (1.0, 1.0)
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
        self.__name__ = getattr(fn, "__name__", type(fn).__name__)

    def __call__(self, individual, **kwargs):
        self.calls += 1
        result = self.fn(individual, **kwargs)
        self.log.append((tuple(individual), tuple(result)))
        return result


# ------------------------------------------------------------------------------ test-only prior noise

def kp_objectives(bits, items_dict, **_):
    """Deterministic knapsack (value, weight), no penalty. Never logs."""
    return (sum(items_dict[i][0] * bits[i] for i in range(len(bits))),
            sum(items_dict[i][1] * bits[i] for i in range(len(bits))))


def cocz_objectives(bits, **_):
    """Deterministic counting ones, counting zeros. Never logs."""
    ones = sum(bits)
    return (ones, len(bits) - ones)


def report(evaluated, true_objectives, observed):
    # the same duck-typed hook the production MO evaluators use
    log_mo_eval = getattr(get_active_logger(), "log_mo_eval", None)
    if log_mo_eval is not None:
        log_mo_eval(evaluated, true_objectives, observed)


class PriorBitflipKP:
    """
    Test-only pure prior noise on the knapsack: x~ is x with each bit flipped with probability flip_prob
    (Python RNG); y = kp(x~) and f(x) = kp(x). Keeps its own list of x~, independent of any logger.
    """
    __name__ = "PriorBitflipKP"

    def __init__(self):
        self.evaluated = []

    def __call__(self, individual, items_dict, flip_prob):
        noisy = [1 - b if random.random() < flip_prob else b for b in individual]
        self.evaluated.append(tuple(noisy))
        observed = kp_objectives(noisy, items_dict)
        report(noisy, kp_objectives(individual, items_dict), observed)
        return observed


class ScriptedPriorCOCZ:
    """
    Test-only prior noise with scripted flips: each call pops the next list of bit positions to flip
    (none once the script is empty). y = cocz(x~), f(x) = cocz(x). No RNG.
    """
    __name__ = "ScriptedPriorCOCZ"

    def __init__(self, script=()):
        self.script = list(script)

    def __call__(self, individual):
        flips = self.script.pop(0) if self.script else []
        noisy = [1 - b if i in flips else b for i, b in enumerate(individual)]
        observed = cocz_objectives(noisy)
        report(noisy, cocz_objectives(individual), observed)
        return observed


# ------------------------------------------------------------------------------ problems and algorithms

def problem(name):
    """(fitness function, fit params, true fitness function, weights, ref point) of a named problem."""
    n_items, capacity, items_dict = knapsack()
    kp = {"items_dict": items_dict, "capacity": capacity}
    kp_ref = [0.0, 2 * sum(float(v[1]) for v in items_dict.values())]
    if name in ("kp0", "kp1"):
        params = dict(kp, noise_intensity=0 if name == "kp0" else 1)
        return (Counting(eval_noisy_kp_v1_mo), params, (eval_noisy_kp_v1_mo, dict(params, noise_intensity=0)),
                KP_WEIGHTS, kp_ref)
    if name == "kpviol1":
        params = dict(kp, noise_intensity=1, penalty=1)
        return (Counting(eval_noisy_kp_v1_mo_violation), params,
                (eval_noisy_kp_v1_mo_violation, dict(params, noise_intensity=0)), KP_WEIGHTS, kp_ref)
    if name == "cocz1":
        params = {"noise_intensity": 1.0}
        return (Counting(countingOnesCountingZeros), params,
                (countingOnesCountingZeros, {"noise_intensity": 0}), COCZ_WEIGHTS, [-10.0, -10.0])
    if name == "prior":
        return (Counting(PriorBitflipKP()), {"items_dict": items_dict, "flip_prob": 0.2},
                (kp_objectives, {"items_dict": items_dict}), KP_WEIGHTS, kp_ref)
    raise KeyError(name)


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


def build(name, prob="kp1", log=True, seed=1, **params):
    """Construct one MO algorithm on a 10-bit problem as the MO runner would, logging on or off."""
    cls, init_args = ALGORITHMS[name]
    init_args = {**init_args, **{k: params.pop(k) for k in list(params) if k in init_args}}
    fitness, fit_params, true_fitness, weights, ref_point = problem(prob)
    kwargs = dict(
        sol_length=knapsack()[0],
        opt_weights=weights,
        attr_function=binary_attribute,
        fitness_function=(fitness, fit_params),
        true_fitness_function=true_fitness,
        ref_point=ref_point,
        gen_limit=12,
        stop_without_improvement_in_gens=5,
        log_evaluations=log,
    )
    kwargs.update(params)
    random.seed(seed)
    np.random.seed(seed)
    return cls(**init_args, **kwargs)


def step_run(algo, observe=lambda algo: None):
    """OptimisationAlgorithm.run(), step by step, calling observe after generation 0 and each update."""
    algo._observe_generation()
    observe(algo)
    while not algo.stop_condition():
        algo.gens += 1
        algo.perform_generation()
        algo._observe_generation()
        observe(algo)


def check_exactly_one_log(algo):
    """One event per counted evaluation, and every population member resolves to its event."""
    fitness = algo.fitness_function[0]
    assert len(algo.eval_log) == algo.evals == fitness.calls, (len(algo.eval_log), algo.evals, fitness.calls)
    for ind in algo.population:
        assert algo.eval_log.event(algo.eval_log.resolve(ind)).x == tuple(ind)


ALGORITHM_NAMES = sorted(ALGORITHMS)


# ------------------------------------------------------------------------------ constructed cases

def direct_hv(values, weights, ref_point):
    """Hypervolume written out independently, with the legacy recorder's convention."""
    if not values:
        return 0.0
    sign = np.array([-1.0 if w > 0 else 1.0 for w in weights])
    return float(hv.hypervolume(np.array(values, dtype=float) * sign, np.array(ref_point, dtype=float) * sign))


LAB_REF = (0.0, 1000.0)


class Lab:
    """
    Crafted evaluations with scripted x~, f(x) and y, committed through MOEvaluationLogger.evaluate onto
    real Individuals, then observed generation by generation. Knapsack weights: maximise the first
    objective, minimise the second.
    """

    def __init__(self):
        build("SEMO", prob="kp1", log=False)  # creator.Individual with the knapsack weights
        self.log = MOEvaluationLogger(KP_WEIGHTS, LAB_REF)

    def evaluate(self, bits, true, observed, evaluated=None):
        ind = creator.Individual(bits)

        def fn(individual):
            get_active_logger().log_mo_eval(individual if evaluated is None else evaluated, true, observed)
            return observed

        y, eval_id = self.log.evaluate(fn, ind, {})
        self.log.tag(ind, eval_id)
        ind.fitness.values = y
        return ind

    def observe(self, population):
        self.log.observe_generation(self.log.n_generations, population)

    def geno(self, bits):
        return self.log.intern(bits)

    def obs(self, ind):
        return self.log.obs_id[self.log.resolve(ind)]

    def hv(self, values):
        return direct_hv(values, KP_WEIGHTS, LAB_REF)


A, B, C = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]
F_A, F_B, F_C = (100, 50), (90, 60), (110, 55)  # true: A dominates B; C is incomparable with A
