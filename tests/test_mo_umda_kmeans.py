"""MoUMDA_KMeans: the clustering multi-objective UMDA.

NSGA-II selects μ = select_size parents from λ = pop_size = 2μ, K-means splits them into
k = floor(sqrt(μ)) clusters on their stored objective vectors, each non-empty cluster of q_i parents
fits a margin-free probability vector on its genotypes and samples 2*q_i offspring, and the offspring
replace the whole population. Only the generic stop criteria apply.

K-means is replaced by fixed labels (`_cluster` patched) wherever a test checks an algorithmic
invariant, so the invariants never depend on a particular clustering outcome. Runs in-process: it
writes nothing, and the end-to-end check calls the MO runner's per-seed function, which writes nothing either.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

import noisyvis.algorithms
import noisyvis.problems
from deap import creator
from noisyvis.algorithms.multi_objective import MoUMDA_KMeans, OptimisationAlgorithm
from noisyvis.algorithms.multi_objective import umda
from noisyvis.algorithms.operators import binary_attribute

WORKSPACE = Path(__file__).resolve().parents[1]
N = 10
WEIGHTS = (1.0, -1.0)


# ------------------------------------------------------------------------------ fixtures


class Counted:
    """A fitness function that counts its calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, individual, **kwargs):
        self.calls += 1
        return self.fn(individual)


def true_objectives(individual):
    # maximise the number of ones, minimise their positional weight
    return (float(sum(individual)), float(sum(i * bit for i, bit in enumerate(individual))))


def noisy_objectives(individual):
    t = true_objectives(individual)
    return (t[0] + random.gauss(0, 1), t[1] + random.gauss(0, 1))


def build(pop_size=12, select_size=6, seed=1, fitness=None, true_fitness=None, **params):
    fitness = fitness if fitness is not None else Counted(noisy_objectives)
    true_fitness = true_fitness if true_fitness is not None else Counted(true_objectives)
    random.seed(seed)
    np.random.seed(seed)
    kwargs = dict(
        sol_length=N,
        opt_weights=WEIGHTS,
        attr_function=binary_attribute,
        fitness_function=(fitness, {}),
        true_fitness_function=(true_fitness, {}),
        ref_point=[0.0, 1000.0],
        gen_limit=100,
    )
    kwargs.update(params)
    return MoUMDA_KMeans(pop_size=pop_size, select_size=select_size, **kwargs)


def individual(bits, fitness):
    ind = creator.Individual(bits)
    ind.fitness.values = fitness
    return ind


def genotypes(individuals):
    return [list(ind) for ind in individuals]


# ------------------------------------------------------------------------------ A. parameters


@pytest.mark.parametrize("mu", [1, 3, 4, 6, 9, 50])
def test_sizes_and_cluster_count(mu):
    algo = build(pop_size=2 * mu, select_size=mu)
    assert algo.select_size == mu
    assert algo.pop_size == 2 * mu
    assert algo.n_clusters == math.floor(math.sqrt(mu))
    assert algo.type == "MoUMDA_KMeans"
    assert algo.name == f"MoUMDA_KMeans(λ={2 * mu}, μ={mu}, k={algo.n_clusters})"
    assert algo.prob_margin is False and algo.prevent_duplicates is False


def test_default_select_size_is_half_pop_size():
    assert build(pop_size=12, select_size=None).select_size == 6


@pytest.mark.parametrize("pop_size, select_size", [(11, None), (12, 5), (12, 7), (0, 0)])
def test_rejects_sizes_other_than_lambda_equals_two_mu(pop_size, select_size):
    fitness = Counted(noisy_objectives)
    with pytest.raises(ValueError):
        build(pop_size=pop_size, select_size=select_size, fitness=fitness)
    assert fitness.calls == 0


def test_rejects_starting_solution():
    fitness = Counted(noisy_objectives)
    with pytest.raises(ValueError, match="starting_solution"):
        build(starting_solution=[1] * N, fitness=fitness)
    assert fitness.calls == 0


@pytest.mark.parametrize("option", [{"prob_margin": True}, {"prob_margin": False}, {"margin_scale": 2.0},
                                    {"prevent_duplicates": True}, {"prevent_duplicates": False}])
def test_margin_and_duplicate_options_are_not_accepted(option):
    with pytest.raises(TypeError):
        build(**option)


# ------------------------------------------------------------------------------ B. initialisation


def test_initial_population_is_lambda_binary_individuals_evaluated_once():
    fitness, true_fitness = Counted(noisy_objectives), Counted(true_objectives)
    algo = build(fitness=fitness, true_fitness=true_fitness)
    assert len(algo.population) == 12
    assert algo.evals == 12 and fitness.calls == 12 and true_fitness.calls == 0
    assert all(type(bit) is int and bit in (0, 1) for ind in algo.population for bit in ind)
    assert all(ind.fitness.valid for ind in algo.population)
    assert algo.gens == 0 and algo.probability_vector is None


def test_initial_population_is_bernoulli_half_whatever_attr_function():
    algo = build(pop_size=2000, select_size=1000, attr_function=lambda: 1)
    bits = np.array(genotypes(algo.population))
    assert abs(bits.mean() - 0.5) < 0.01
    assert 0 < bits.mean(axis=0).min() and bits.mean(axis=0).max() < 1


def test_initial_population_is_sampled_from_the_all_half_vector():
    algo = build(seed=3)
    random.seed(3)
    np.random.seed(3)
    random.randint(0, 10**6)  # seed_signature, drawn by OptimisationAlgorithm.__post_init__
    expected = umda.sample_from_probability_vector(np.full(N, 0.5), 12)
    assert genotypes(algo.population) == genotypes(expected)


# ------------------------------------------------------------------------------ C/D. selection and clustering


def test_selection_is_nsga2_of_mu_and_kmeans_sees_their_stored_objectives():
    fitness, true_fitness = Counted(noisy_objectives), Counted(true_objectives)
    algo = build(fitness=fitness, true_fitness=true_fitness)
    before = list(algo.population)
    seen = {}

    real_select = umda.tools.selNSGA2

    def select(population, k):
        seen["select_args"] = (list(population), k)
        seen["parents"] = real_select(population, k)
        return seen["parents"]

    def cluster(points):
        seen["points"] = np.array(points, copy=True)
        seen["calls_during_clustering"] = (fitness.calls, true_fitness.calls)
        return np.zeros(len(points), dtype=int), np.zeros((algo.n_clusters, 2))

    with mock.patch.object(umda.tools, "selNSGA2", side_effect=select) as spy, \
            mock.patch.object(algo, "_cluster", side_effect=cluster):
        algo.perform_generation()

    assert spy.call_count == 1
    population, k = seen["select_args"]
    assert k == 6 and all(a is b for a, b in zip(population, before)) and len(population) == 12
    assert len(seen["parents"]) == 6

    points = seen["points"]
    assert points.shape == (6, 2)  # (μ, M), not (μ, n)
    np.testing.assert_array_equal(points, [p.fitness.values for p in seen["parents"]])
    # the noisy stored values, not true fitness
    assert not np.array_equal(points, [true_objectives(p) for p in seen["parents"]])
    assert seen["calls_during_clustering"] == (12, 0)
    assert fitness.calls == 24 and true_fitness.calls == 0


def test_kmeans_settings_and_raw_objective_points():
    algo = build(pop_size=18, select_size=9)
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                       [100.0, 100.0], [100.1, 100.0], [100.0, 100.1],
                       [0.0, 1000.0], [0.1, 1000.0], [0.0, 1000.1]])
    with mock.patch.object(umda, "KMeans", wraps=umda.KMeans) as spy:
        labels, centers = algo._cluster(points)
    kwargs = spy.call_args.kwargs
    assert kwargs["n_clusters"] == 3
    assert (kwargs["init"], kwargs["n_init"], kwargs["max_iter"], kwargs["tol"], kwargs["algorithm"]) == \
        ("k-means++", 10, 300, 1e-4, "lloyd")
    assert isinstance(kwargs["random_state"], int)
    groups = {frozenset(np.flatnonzero(labels == label)) for label in set(labels)}
    assert groups == {frozenset({0, 1, 2}), frozenset({3, 4, 5}), frozenset({6, 7, 8})}
    assert centers.shape == (3, 2)


def test_fewer_distinct_points_than_clusters_leaves_empty_clusters_and_still_samples_lambda():
    algo = build(pop_size=18, select_size=9)
    parents = [individual([1] * N, (5.0, 5.0)) for _ in range(9)]
    with mock.patch.object(algo, "_select_parents", return_value=parents):
        algo.perform_generation()
    assert sum(algo.cluster_sizes) == 9 and len(algo.cluster_sizes) == 3
    assert len(algo.population) == 18


# ------------------------------------------------------------------------------ E/F. cluster models and sampling


def controlled_parents():
    """Nine parents: cluster 0 has 4, cluster 1 has 5, cluster 2 is empty."""
    zero = [[1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 1, 0]]
    one = [[0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
           [0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
           [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
           [0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 1]]
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1])
    rows = iter(zero), iter(one)
    parents = [individual(next(rows[label]), (float(i), float(-i))) for i, label in enumerate(labels)]
    return parents, labels, np.mean(zero, axis=0), np.mean(one, axis=0)


def test_one_margin_free_model_per_cluster_sampling_two_q_i_each():
    algo = build(pop_size=18, select_size=9)
    parents, labels, p0, p1 = controlled_parents()
    sampler = mock.Mock(wraps=umda.sample_from_probability_vector)
    with mock.patch.object(algo, "_select_parents", return_value=parents), \
            mock.patch.object(algo, "_cluster", return_value=(labels, np.zeros((3, 2)))), \
            mock.patch.object(umda, "sample_from_probability_vector", sampler):
        algo.perform_generation()

    assert algo.cluster_sizes == [4, 5, 0]
    vectors = algo.cluster_probability_vectors
    assert len(vectors) == 3 and vectors[2] is None
    np.testing.assert_array_equal(vectors[0], p0)
    np.testing.assert_array_equal(vectors[1], p1)
    for p in vectors[:2]:  # no margin: exact 0 and 1 survive
        assert (p == 0.0).any() and (p == 1.0).any()

    assert [c.args[1] for c in sampler.call_args_list] == [8, 10]
    np.testing.assert_array_equal(sampler.call_args_list[0].args[0], p0)
    np.testing.assert_array_equal(sampler.call_args_list[1].args[0], p1)
    assert all(not c.kwargs.get("prevent_duplicates", False) for c in sampler.call_args_list)

    # offspring in cluster order; fixed positions copied exactly (no mutation, no crossover)
    assert len(algo.population) == 18
    for block, p in ((algo.population[:8], p0), (algo.population[8:], p1)):
        for ind in block:
            fixed = (p == 0.0) | (p == 1.0)
            np.testing.assert_array_equal(np.array(ind)[fixed], p[fixed])
    np.testing.assert_array_equal(algo.cluster_labels, labels)
    assert algo.probability_vector is None


def test_duplicates_are_allowed():
    algo = build(pop_size=18, select_size=9)
    parents = [individual([1, 0] * (N // 2), (float(i), float(i))) for i in range(9)]
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    with mock.patch.object(algo, "_select_parents", return_value=parents), \
            mock.patch.object(algo, "_cluster", return_value=(labels, np.zeros((3, 2)))):
        algo.perform_generation()
    assert genotypes(algo.population) == [[1, 0] * (N // 2)] * 18


# ------------------------------------------------------------------------------ G. replacement and evaluations


def test_full_generational_replacement_and_lambda_evaluations_per_generation():
    fitness = Counted(noisy_objectives)
    algo = build(fitness=fitness)
    for generation in range(1, 4):
        old = {id(ind) for ind in algo.population}
        algo.gens += 1
        algo.perform_generation()
        assert len(algo.population) == 12
        assert not old & {id(ind) for ind in algo.population}
        assert algo.evals == 12 * (generation + 1) and fitness.calls == algo.evals


def test_failed_evaluation_commits_nothing():
    calls = {"n": 0}

    def failing(individual):
        calls["n"] += 1
        if calls["n"] > 12 + 5:
            raise RuntimeError("evaluation failed")
        return noisy_objectives(individual)

    algo = build(fitness=Counted(failing))
    population, sizes = list(algo.population), algo.cluster_sizes
    with pytest.raises(RuntimeError):
        algo.perform_generation()
    assert algo.population == population and algo.evals == 12 and algo.cluster_sizes is sizes


# ------------------------------------------------------------------------------ H. stopping


def converged_run(**params):
    algo = build(fitness=Counted(true_objectives), **params)
    for ind in algo.population:
        ind[:] = [1, 0] * (N // 2)
        ind.fitness.values = true_objectives(ind)
    return algo


def test_converged_cluster_models_do_not_stop_the_run():
    algo = converged_run(gen_limit=5)
    algo.probability_vector = np.ones(N)  # would stop ordinary MoUMDA; must be ignored here
    assert algo.stop_condition() is False
    algo.probability_vector = None
    algo.run()
    assert algo.gens == 5 and algo.stop_trigger == "gen_limit"
    assert all(p is None or np.all((p == 0) | (p == 1)) for p in algo.cluster_probability_vectors)
    assert algo.probability_vector is None


def test_generic_stop_criteria_still_apply():
    by_evals = build(eval_limit=36)
    by_evals.run()
    assert (by_evals.gens, by_evals.evals, by_evals.stop_trigger) == (2, 36, "eval_limit")

    stalled = converged_run(stop_without_improvement_in_gens=3)
    stalled.run()
    assert stalled.stop_trigger == "no_improvement" and stalled.gens == 3

    assert MoUMDA_KMeans.stop_condition is not umda.MoUMDABase.stop_condition
    algo = build()
    with mock.patch.object(OptimisationAlgorithm, "stop_condition", return_value=True) as generic:
        assert algo.stop_condition() is True
    generic.assert_called_once()


# ------------------------------------------------------------------------------ reproducibility


def run_seeded(seed):
    algo = build(pop_size=18, select_size=9, seed=seed, gen_limit=6)
    with mock.patch.object(umda, "KMeans", wraps=umda.KMeans) as spy:
        algo.run()
    return {
        "population": genotypes(algo.population),
        "fitness": [ind.fitness.values for ind in algo.population],
        "cluster_sizes": algo.cluster_sizes,
        "labels": algo.cluster_labels.tolist(),
        "hv": algo.true_pf_hypervolumes,
        "kmeans_seeds": [c.kwargs["random_state"] for c in spy.call_args_list],
    }


def test_identically_seeded_runs_are_identical():
    first, second, other = run_seeded(7), run_seeded(7), run_seeded(8)
    assert first == second
    assert first != other
    assert len(first["kmeans_seeds"]) == 6 and len(set(first["kmeans_seeds"])) == 6
    assert first["kmeans_seeds"] != other["kmeans_seeds"]


# ------------------------------------------------------------------------------ end to end


def test_default_config_runs_through_the_mo_runner():
    from hydra.utils import instantiate  # noqa: F401  (the runner's own dependency)
    from noisyvis.experiments.config.workflows import resolve_mo_config
    from noisyvis.experiments.runner import mo_algo_data_single

    cfg = OmegaConf.create(yaml.safe_load((WORKSPACE / "tests" / "configs" / "mo_knapsack.yaml").read_text()))
    fragment = yaml.safe_load((WORKSPACE / "configs/defaults/multiobjective/algos/moumda_kmeans.yaml").read_text())
    cfg.algo = OmegaConf.create(fragment["algo"])
    cfg.run.max_gens = 3
    cfg.run.eval_limit = None
    cfg = resolve_mo_config(cfg)

    fitness_fn = getattr(noisyvis.problems, cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)
    true_fit_params = dict(fit_params, noise_intensity=0)
    algo_params = {
        "sol_length": cfg.problem.dimensions,
        "opt_weights": tuple(cfg.problem.weights),
        "eval_limit": cfg.run.eval_limit,
        "attr_function": getattr(noisyvis.algorithms, cfg.problem.attr_function),
        "starting_solution": None,
        "target_stop": None,
        "gen_limit": cfg.run.max_gens,
        "stop_without_improvement_in_gens": None,
        "fitness_function": (fitness_fn, fit_params),
        "true_fitness_function": (fitness_fn, true_fit_params),
        "ref_point": cfg.problem.get("ref_point", None),
        "verbose_rate": 0,
    }
    prob_info = {key: None for key in ("name", "type", "goal", "dimensions", "opt_global", "mean_value",
                                       "mean_weight", "PID", "experiment_name", "experiment_description")}
    row = mo_algo_data_single(prob_info, cfg.algo.init_args, algo_params, seed=1)
    assert row["algo_type"] == "MoUMDA_KMeans"
    assert row["algo_name"] == "MoUMDA_KMeans(λ=100, μ=50, k=7)"
    assert (row["n_gens"], row["n_evals"], row["stop_trigger"]) == (3, 400, "gen_limit")
    assert row["true_pf_hypervolumes"]
