"""Multi-objective algorithm contracts, characterised before the MO package split (MO refactor, Stage 0).

`noisyvis.algorithms.multi_objective` is about to become a package (`base`, `umda`, `semo`, `nsga2`),
gain a `MoUMDABase`, stop MoUMDA runs whose probability vector has converged, and have its
`MoUMDA_noDuplicates` defect fixed. Before anything moves, this module pins what exists today:

1. Every public MO name still resolves at its flat `noisyvis.algorithms.multi_objective.<name>` path
   (the Hydra `_target_` spelling every config uses) to the object the `noisyvis.algorithms`
   namespace exports. Location-agnostic: the defining module may be the package or a submodule.
2. Every configured MO algorithm target instantiates through the MO resolver and the runner's
   parameter construction and runs two generations.
3. Behavioural characterisation against the frozen baseline `baselines/mo_algorithms.json`:
   MoUMDA and MoUMDA_ParetoArchive with and without probability margins (the archive fingerprinted
   every generation), and NSGA-II. SEMO is already pinned by `baselines/mo.json`. There is
   deliberately no MoUMDA_noDuplicates baseline: its current behaviour is the defect below.
8. `_update_archive_nondominated` keeps genotype duplicates. Under noisy evaluation copies of one
   genotype can carry different observed objectives; whether the archive should deduplicate them is
   an open research question, so the refactor must not change it.

A defect pinned by strict xfails from Stage 0 and fixed in MO refactor Stage 5 (the xfails removed):

9. `MoUMDA_noDuplicates` never passed `prevent_duplicates` to the sampler, so its offspring contained
   duplicate genotypes: it behaved exactly like `MoUMDA` and differed only in `name`/`type`.
10. Its initial population was built by the generic initialiser, so it was not duplicate-free either.

Since Stage 5 duplicate-free runs check, before each generation, whether the probability vector that
generation would use can support pop_size distinct genotypes, and stop with
`probability_vector_converged` or `insufficient_unique_support` if not.

The characterisation is a strict *prefix* contract for later stages, not only an equality one: runs
whose probability vector fully converges will stop there once the convergence stop exists (Stage 4),
so every case records per-generation fingerprints and whether the vector that generated each
generation had collapsed to 0/1. Until then the whole record must match exactly.

The baseline was captured from the untouched pre-refactor tree (commit recorded in the file) by
running `run_probe()` twice in fresh processes and requiring identical output. It must never be
re-derived to make a test pass. Imports of the science packages happen only in the probe subprocess.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.run_isolated import HARNESS_DIR, WORKSPACE, child_env, make_temp_root

BASELINE_PATH = Path(__file__).resolve().parent / "baselines" / "mo_algorithms.json"

MO_PREFIX = "noisyvis.algorithms.multi_objective."

# Public names a config, the runner or a user can reach at the flat MO path today.
PUBLIC_MO_NAMES = (
    "OptimisationAlgorithm", "MoUMDABase", "SEMO", "MoUMDA", "MoUMDA_noDuplicates", "MoUMDA_ParetoArchive", "NSGA2",
    "mo_umda_update_full", "mo_umda_update_with_archive", "record_pareto_data", "front_sig",
    "mut_flip_one_bit",
)
MO_ALGORITHM_CLASSES = frozenset({"SEMO", "MoUMDA", "MoUMDA_noDuplicates", "MoUMDA_ParetoArchive", "NSGA2"})

# B1 (known_broken_configs.py): the one MO target that does not exist.
KNOWN_UNRESOLVABLE_TARGETS = frozenset({MO_PREFIX + "MoMuPlusLamdaEA"})

# Check 2 shrinks every run to this, so the sweep stays fast. 12 keeps NSGA-II's
# selTournamentDCD happy (it needs a population divisible by four).
CONFIG_SWEEP_GENS = 2
CONFIG_SWEEP_POP_SIZE = 12
CONFIG_SWEEP_SELECT_SIZE = 6

# ------------------------------------------------------------------ characterisation spec

SEEDS = [1, 2]

# prob_margin is always explicit, so a change of class default can never alter what a case means.
# The margin-off cases record every generation, so their series are prefix-comparable per generation.
CHARACTERISATION_CASES = {
    "moumda_margin_on": {
        "init_args": {"_target_": MO_PREFIX + "MoUMDA", "pop_size": 20, "select_size": 10, "prob_margin": True},
        "gen_limit": 30,
        "record_every_gen": False,
    },
    "moumda_margin_off": {
        "init_args": {"_target_": MO_PREFIX + "MoUMDA", "pop_size": 20, "select_size": 10, "prob_margin": False},
        "gen_limit": 150,
        "record_every_gen": True,
    },
    "pareto_archive_margin_on": {
        "init_args": {"_target_": MO_PREFIX + "MoUMDA_ParetoArchive", "pop_size": 20, "select_size": 10,
                      "prob_margin": True},
        "gen_limit": 30,
        "record_every_gen": False,
    },
    "pareto_archive_margin_off": {
        "init_args": {"_target_": MO_PREFIX + "MoUMDA_ParetoArchive", "pop_size": 20, "select_size": 10,
                      "prob_margin": False},
        "gen_limit": 150,
        "record_every_gen": True,
    },
    "nsga2": {
        "init_args": {"_target_": MO_PREFIX + "NSGA2", "pop_size": 20, "cxpb": 0.9, "mutpb": 0.2,
                      "mate_op": "deap.tools.cxTwoPoint", "mutate_op": "deap.tools.mutFlipBit",
                      "mutate_params": {"indpb": 0.1}},
        "gen_limit": 30,
        "record_every_gen": False,
    },
}

# ------------------------------------------------------------------ helper expectations (checks 4/14)

# binary_support_at_least(k, required) is exactly 2**k >= required.
SUPPORT_CASES = [(6, 100), (7, 100), (6, 64), (6, 65), (0, 1), (0, 2), (1, 2), (1, 3), (10_000, 100)]
EXPECTED_SUPPORT = {"6,100": False, "7,100": True, "6,64": True, "6,65": False, "0,1": True, "0,2": False,
                    "1,2": True, "1,3": False, "10000,100": True}

# no_duplicates_stop_reason(vector with k free bits, pop_size): convergence wins over support.
REASON_CASES = [(0, 10), (0, 1), (3, 10), (4, 10), (6, 100), (7, 100)]
EXPECTED_REASONS = {"0,10": "probability_vector_converged", "0,1": "probability_vector_converged",
                    "3,10": "insufficient_unique_support", "4,10": None,
                    "6,100": "insufficient_unique_support", "7,100": None}

# Frozen outputs of direct legacy-wrapper calls on the paths the classes never reach: duplicate
# rejection, real-valued genes and a non-empty archive. Captured at commit 3395480 (MO refactor
# Stage 1), where both wrappers were still AST-identical to the pre-refactor module. `next_draw` is
# the next np.random draw after the call, so it pins how much RNG each call consumed. Never
# re-derive these to make a test pass.
LEGACY_WRAPPER_OUTPUTS = {
    "archive binary empty": {
        "archive_sha": "19554dfe5411b60ca450eb44bd818766625a924d58a0cf4e59f3bc8a86a97bc8",
        "gene_type": "int", "next_draw": 0.22407863694631935, "size": 12,
        "solutions_sha": "2b2422cd0b6cf241965828e7802ff73ca36ef19d4d249a0d9e85f6dddb301d15",
    },
    "archive binary seeded": {
        "archive_sha": "174da8f63f779296cf3652cc2b03e4ee963cc8a0cbfdfbca1a892f14db9a72af",
        "gene_type": "int", "next_draw": 0.22407863694631935, "size": 12,
        "solutions_sha": "df827a07cd7afad75e85e0ab58871bacdb884b34e41e6b488134bb783a4367a8",
    },
    "archive real": {
        "archive_sha": "1a642cbd6f70ea9ee68c32509ba0e63d3fb5a5faa5c08a9e96c6fdc88bb46c04",
        "gene_type": "float", "next_draw": 0.7963907007825485, "size": 12,
        "solutions_sha": "9365ea5cdd2a9fd530242c42b5722580ecb23c6bbefcb43e4006843b67cab672",
    },
    "full binary margin": {
        "gene_type": "int", "next_draw": 0.22407863694631935, "size": 12,
        "solutions_sha": "f312317a701e6483e05fcfc7079385b6d5e6aba1c3b4e549b6e0b3b2813ed808",
    },
    "full binary unique": {
        "gene_type": "int", "next_draw": 0.3290495438867621, "size": 12,
        "solutions_sha": "2f1a4f19324df29b8c5b51a09dd3e0141bc7db19ba98ed2692bd372e86fddb31",
    },
    "full real": {
        "gene_type": "float", "next_draw": 0.7963907007825485, "size": 12,
        "solutions_sha": "8de677e7902d4839f8489daa4d361353a371870ab8396c2d51c76c52d8e5db3c",
    },
    "full real unique": {
        "gene_type": "float", "next_draw": 0.7963907007825485, "size": 12,
        "solutions_sha": "8de677e7902d4839f8489daa4d361353a371870ab8396c2d51c76c52d8e5db3c",
    },
}

# ------------------------------------------------------------------------------ the probe

# Runs in a harness subprocess (fresh temp root, write fence installed). It only reports what it
# observes; every expectation lives in this file or in the frozen baseline.
_PROBE = r'''
import hashlib
import importlib.util
import json
import random
import sys
import traceback
from pathlib import Path

ARGS = json.loads(sys.argv[1])

spec = importlib.util.spec_from_file_location("_harness_fence", ARGS["fence"])
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)
fence.install(ARGS["workspace"])

import deap
import hydra
import numpy as np
import yaml
from deap import creator, tools
from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import OmegaConf

import noisyvis.algorithms
import noisyvis.problems
from noisyvis.experiments.config.workflows import resolve_mo_config

WORKSPACE = Path(ARGS["workspace"])
MO_PREFIX = ARGS["mo_prefix"]
BASE_CONFIG = WORKSPACE / "tests" / "configs" / "mo_knapsack.yaml"
algorithms = sys.modules["noisyvis.algorithms"]
problems = sys.modules["noisyvis.problems"]


def section(fn):
    """Run one probe section; an exception is reported, never allowed to hide other sections."""
    try:
        return fn()
    except Exception:
        return {"error": traceback.format_exc()}


# ---------------------------------------------------------------- shared helpers

def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    raw.pop("hydra", None)  # Hydra runtime settings; not part of the experiment config
    return OmegaConf.create(raw)


def runner_algo_params(cfg):
    """The algo_params of run_mo_experiment (src/noisyvis/experiments/runner.py), built the same way."""
    fitness_fn = getattr(problems, cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)
    true_fit_params = fit_params.copy()
    true_fit_params["noise_intensity"] = 0
    cfg_start_sol = getattr(cfg.algo, "starting_solution", None)
    start_sol = list(cfg_start_sol) if cfg_start_sol is not None else None
    return {
        "sol_length": cfg.problem.dimensions,
        "opt_weights": tuple(cfg.problem.weights),
        "eval_limit": cfg.run.eval_limit,
        "attr_function": getattr(sys.modules["noisyvis.algorithms"], cfg.problem.attr_function),
        "starting_solution": start_sol,
        "target_stop": None,
        "gen_limit": cfg.run.max_gens,
        "stop_without_improvement_in_gens": cfg.run.get("stop_without_improvement_in_gens", None),
        "fitness_function": (fitness_fn, fit_params),
        "true_fitness_function": (fitness_fn, true_fit_params),
        "ref_point": cfg.problem.get("ref_point", None),
        "verbose_rate": cfg.run.get("verbose_rate", 0),
    }


def seed_all(seed):
    """What mo_algo_data_single does before instantiating."""
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return (random.getstate(), repr(np.random.get_state()))


def sha(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def genotypes(individuals):
    return [[int(x) for x in ind] for ind in individuals]


def fitness_rows(rows):
    return [[float(v) for v in row] for row in rows]


def duplicate_genotypes(individuals):
    keys = [tuple(int(x) for x in ind) for ind in individuals]
    return len(keys) - len(set(keys))


def umda_module():
    """Wherever the UMDA family lives (the flat module today, a submodule after the split)."""
    return sys.modules[_locate(MO_PREFIX + "MoUMDA_ParetoArchive").__module__]


def case_config(case):
    cfg = load_config(BASE_CONFIG)
    cfg.run.max_gens = case["gen_limit"]
    cfg.run.eval_limit = None
    cfg.algo = OmegaConf.create({"init_args": case["init_args"]})
    return resolve_mo_config(cfg)


def build(case, seed):
    cfg = case_config(case)
    params = runner_algo_params(cfg)
    params["record_every_gen"] = case["record_every_gen"]
    params.update(case.get("params", {}))  # optional overrides; the characterisation cases have none
    seed_all(seed)
    return instantiate(cfg.algo.init_args, **params)


# ---------------------------------------------------------------- 1. public import paths

def public_names():
    found = {}
    for name in ARGS["public_names"]:
        obj = _locate(MO_PREFIX + name)
        found[name] = {
            "module": getattr(obj, "__module__", None),
            "callable": callable(obj),
            "is_class": isinstance(obj, type),
            "is_package_namespace_object": obj is getattr(algorithms, name, None),
        }
    return found


# ---------------------------------------------------------------- 2. configured targets

def config_sweep():
    outcomes = {}
    for where in ("configs", "tests/configs"):
        for path in sorted((WORKSPACE / where).rglob("*.yaml")):
            if not path.exists():  # a dangling symlink: config hygiene, reported not run
                outcomes[str(path.relative_to(WORKSPACE))] = {"dangling": True}
                continue
            raw = yaml.safe_load(path.read_text())
            if not isinstance(raw, dict):
                continue
            target = ((raw.get("algo") or {}).get("init_args") or {}).get("_target_")
            if not (isinstance(target, str) and target.startswith(MO_PREFIX)):
                continue
            relative = str(path.relative_to(WORKSPACE))
            if target in ARGS["known_unresolvable"]:
                outcomes[relative] = {"target": target, "skipped": "known unresolvable (B1)"}
                continue
            try:
                if "problem" in raw:
                    cfg = load_config(path)
                else:  # an algo-only fragment (configs/defaults/...): run it on the MO test problem
                    cfg = load_config(BASE_CONFIG)
                    cfg.algo = OmegaConf.create(raw["algo"])
                cfg.run.max_gens = ARGS["sweep_gens"]
                cfg.run.eval_limit = None
                cfg.run.use_noise_dependent_eval_limit = False
                cfg.run.stop_without_improvement_in_gens = None
                cfg = resolve_mo_config(cfg)
                if "pop_size" in cfg.algo.init_args:
                    cfg.algo.init_args.pop_size = ARGS["sweep_pop_size"]
                if "select_size" in cfg.algo.init_args:
                    cfg.algo.init_args.select_size = ARGS["sweep_select_size"]
                params = runner_algo_params(cfg)
                seed_all(1)
                algo = instantiate(cfg.algo.init_args, **params)
                algo.run()
                outcomes[relative] = {"target": target, "class": type(algo).__name__, "name": algo.name,
                                      "type": algo.type, "gens": algo.gens, "evals": algo.evals,
                                      "stop_trigger": algo.stop_trigger}
            except Exception:
                outcomes[relative] = {"target": target, "error": traceback.format_exc()}
    return outcomes


# ---------------------------------------------------------------- 3. characterisation

def expected_generating_vector(algo):
    """The probability vector the next generation will sample from, computed independently.

    Recomputes, without committing anything, exactly what perform_generation is about to compute:
    selNSGA2 parents (for the archive variant, the non-dominated archive update), their mean, and
    the margin clip. Consumes no RNG, which is asserted.
    """
    before = rng_state()
    parents = tools.selNSGA2(algo.population, algo.select_size)
    source = parents
    if hasattr(algo, "archive"):
        source = umda_module()._update_archive_nondominated(algo.archive, parents) or parents
    probs = np.mean(source, axis=0)
    if algo.prob_margin:
        eps = algo.margin_scale / float(algo.sol_length)
        probs = np.clip(probs, eps, 1.0 - eps)
    if rng_state() != before:
        raise AssertionError("the vector probe consumed RNG")
    return probs


def generating_vector_collapsed(algo):
    """Whether the vector the next generation will sample from is entirely 0/1; None with no UMDA model."""
    if not hasattr(algo, "select_size"):
        return None
    probs = expected_generating_vector(algo)
    return bool(np.all((probs == 0.0) | (probs == 1.0)))


def generation_fingerprint(algo, collapsed):
    fp = {
        "gen": algo.gens,
        "evals": algo.evals,
        "population_genotypes_sha": sha(genotypes(algo.population)),
        "population_fitnesses_sha": sha(fitness_rows(ind.fitness.values for ind in algo.population)),
        "population_duplicate_genotypes": duplicate_genotypes(algo.population),
    }
    if collapsed is not None:
        fp["generating_vector_collapsed"] = collapsed
    if hasattr(algo, "archive"):
        fp["archive_size"] = len(algo.archive)
        fp["archive_genotypes_sha"] = sha(genotypes(algo.archive))
        fp["archive_fitnesses_sha"] = sha(fitness_rows(ind.fitness.values for ind in algo.archive))
        fp["archive_duplicate_genotypes"] = duplicate_genotypes(algo.archive)
    return fp


def run_summary(algo):
    return {
        "class": type(algo).__name__,
        "name": algo.name,
        "type": algo.type,
        "gens": algo.gens,
        "evals": algo.evals,
        "stop_trigger": algo.stop_trigger,
        "noisy_pf_noisy_hypervolumes": [float(x) for x in algo.noisy_pf_noisy_hypervolumes],
        "noisy_pf_true_hypervolumes": [float(x) for x in algo.noisy_pf_true_hypervolumes],
        "true_pf_hypervolumes": [float(x) for x in algo.true_pf_hypervolumes],
        "n_gens_pareto_best": [int(x) for x in algo.n_gens_pareto_best],
        "pareto_solutions_sha": [sha(genotypes(front)) for front in algo.pareto_solutions],
        "pareto_fitnesses_sha": [sha(fitness_rows(front)) for front in algo.pareto_fitnesses],
        "pareto_true_fitnesses_sha": [sha(fitness_rows(front)) for front in algo.pareto_true_fitnesses],
        "true_pareto_solutions_sha": [sha(genotypes(front)) for front in algo.true_pareto_solutions],
        "true_pareto_fitnesses_sha": [sha(fitness_rows(front)) for front in algo.true_pareto_fitnesses],
        "final_population_genotypes_sha": sha(genotypes(algo.population)),
        "final_population_fitnesses_sha": sha(fitness_rows(ind.fitness.values for ind in algo.population)),
    }


def drive(algo):
    """OptimisationAlgorithm.run(), step by step, fingerprinting every generation."""
    per_generation = []
    while not algo.stop_condition():
        collapsed = generating_vector_collapsed(algo)
        algo.gens += 1
        algo.perform_generation()
        algo.record_state_pareto(algo.population)
        per_generation.append(generation_fingerprint(algo, collapsed))
    return per_generation


def characterisation():
    cases = {}
    for name, case in ARGS["cases"].items():
        cases[name] = {}
        for seed in ARGS["seeds"]:
            algo = build(case, seed)
            per_generation = drive(algo)
            cases[name][str(seed)] = {"summary": run_summary(algo), "per_generation": per_generation}
    return cases


def run_equivalence():
    """The stepped loop above is run() exactly: a plain run() must give the identical summary."""
    seed = ARGS["seeds"][0]
    found = {}
    for name, case in ARGS["cases"].items():
        stepped = build(case, seed)
        drive(stepped)
        plain = build(case, seed)
        plain.run()
        found[name] = run_summary(plain) == run_summary(stepped)
    return found


# ---------------------------------------------------------------- 8. archive duplicate semantics

def archive_duplicate_semantics():
    build(ARGS["cases"]["pareto_archive_margin_on"], 1)  # creator.Individual with weights (1, -1)
    update = umda_module()._update_archive_nondominated

    def individual(bits, fitness):
        ind = creator.Individual(bits)
        ind.fitness.values = fitness
        return ind

    named = {
        "a": individual([1, 0, 1], (10.0, 5.0)),  # archive member
        "b": individual([1, 0, 1], (10.0, 5.0)),  # same genotype, identical fitness
        "c": individual([1, 0, 1], (12.0, 6.0)),  # same genotype, noisy and mutually non-dominated with a
        "d": individual([1, 0, 1], (8.0, 6.0)),   # same genotype, dominated by a
        "e": individual([0, 1, 1], (9.0, 4.0)),   # other genotype, non-dominated
    }
    archive = [named["a"]]
    candidates = [named[k] for k in ("b", "c", "d", "e")]
    archive_before = list(archive)
    result = update(archive, candidates)
    names_by_id = {id(obj): key for key, obj in named.items()}
    return {
        "kept": sorted(names_by_id[id(obj)] for obj in result),
        "kept_duplicate_genotypes": duplicate_genotypes(result),
        "returns_new_list": result is not archive,
        "archive_argument_unchanged": len(archive) == len(archive_before)
                                      and all(x is y for x, y in zip(archive, archive_before)),
        "empty_inputs": update([], []),
    }


# ---------------------------------------------------------------- 9/10. noDuplicates defect

def no_duplicates_generation():
    """Inject a population whose probability vector has exactly 4 free bits (16 genotypes) and
    select_size == pop_size (so the vector is the exact population mean), then run one generation."""
    case = {"init_args": {"_target_": MO_PREFIX + "MoUMDA_noDuplicates", "pop_size": 10, "select_size": 10,
                          "prob_margin": False},
            "gen_limit": 5, "record_every_gen": False}
    algo = build(case, 1)
    injected = []
    for value in range(10):
        bits = [(value >> b) & 1 for b in range(4)] + [0] * (algo.sol_length - 4)
        ind = creator.Individual(bits)
        ind.fitness.values = algo.toolbox.evaluate(ind)
        injected.append(ind)
    algo.population = injected
    probs = np.mean(injected, axis=0)
    evals_before = algo.evals
    counting = Counting(algo.fitness_function[0])
    algo.fitness_function = (counting, algo.fitness_function[1])
    seed_all(1)
    algo.perform_generation()
    keys = [tuple(int(x) for x in ind) for ind in algo.population]
    return {
        "free_bits": int(np.count_nonzero((probs > 0.0) & (probs < 1.0))),
        "population_size": len(keys),
        "unique_genotypes": len(set(keys)),
        "all_within_support": all(not any(key[4:]) for key in keys),
        "evals_added": algo.evals - evals_before,
        "evaluations": counting.calls,
    }


class Counting:
    """Wraps a callable and counts its calls."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def no_duplicates_initial_population():
    """sol_length 4 has 16 genotypes, so 10 unique initial individuals are possible (with rejections)."""
    cfg = case_config({"init_args": {"_target_": MO_PREFIX + "MoUMDA_noDuplicates", "pop_size": 10,
                                     "select_size": 5, "prob_margin": False},
                       "gen_limit": 5})
    params = runner_algo_params(cfg)
    params["sol_length"] = 4
    attribute = Counting(params["attr_function"])
    fitness = Counting(params["fitness_function"][0])
    params["attr_function"] = attribute
    params["fitness_function"] = (fitness, params["fitness_function"][1])
    seed_all(1)
    algo = instantiate(cfg.algo.init_args, **params)
    keys = [tuple(int(x) for x in ind) for ind in algo.population]
    return {"population_size": len(keys), "unique_genotypes": len(set(keys)), "evals": algo.evals,
            "candidates_drawn": attribute.calls // 4, "evaluations": fitness.calls}


# ---------------------------------------------------------------- 4/14. helpers and direct calls

def ten_bit_vector(free_bits):
    """A 10-bit probability vector: `free_bits` positions at 0.5, the rest fixed at 1."""
    return np.array([0.5] * free_bits + [1.0] * (10 - free_bits))


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return "ValueError"
    return "no error"


def helper_contracts():
    um = umda_module()
    vectors = {
        "partial": np.array([0.0, 1.0, 0.5, 1.0]),
        "collapsed": np.array([0.0, 1.0, 1.0, 0.0]),
        "empty": np.array([]),
        "margin_clipped": np.clip(np.array([0.0, 1.0, 1.0, 0.0]), 0.1, 0.9),
    }
    copies = {name: v.copy() for name, v in vectors.items()}
    before = rng_state()

    converged = {name: um.is_probability_vector_converged(v) for name, v in vectors.items()}
    converged["None"] = um.is_probability_vector_converged(None)
    free_bits = {name: um.count_free_bits(v) for name, v in vectors.items()}
    support = {f"{k},{required}": um.binary_support_at_least(k, required)
               for k, required in ARGS["support_cases"]}
    reasons = {f"{k},{pop}": um.no_duplicates_stop_reason(ten_bit_vector(k), pop)
               for k, pop in ARGS["reason_cases"]}
    # Direct misuse fails fast, before any draw: impossible duplicate-free support.
    guards = {
        "sample k=3 pop=10": raises_value_error(
            lambda: um.sample_from_probability_vector(ten_bit_vector(3), 10, prevent_duplicates=True)),
        "sample k=0 pop=10": raises_value_error(
            lambda: um.sample_from_probability_vector(ten_bit_vector(0), 10, prevent_duplicates=True)),
    }
    pure = {"inputs_unchanged": all(np.array_equal(vectors[n], copies[n]) for n in vectors),
            "rng_unchanged": rng_state() == before}

    build(ARGS["cases"]["moumda_margin_on"], 1)  # creator.Individual with weights (1, -1)
    collapsed_population = []
    for i in range(10):
        ind = creator.Individual([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        ind.fitness.values = (float(i), float(i))
        collapsed_population.append(ind)
    guards["mo_umda_update_full collapsed"] = raises_value_error(
        lambda: um.mo_umda_update_full(10, collapsed_population, 10, 5, None, prob_margin=False,
                                       prevent_duplicates=True))

    seed_all(1)
    sampled = um.sample_from_probability_vector(ten_bit_vector(4), 10, prevent_duplicates=True)
    unique_sample = {"size": len(sampled), "unique": len({tuple(ind) for ind in sampled}),
                     "within_support": all(all(bit == 1 for bit in ind[4:]) for ind in sampled)}
    return {"converged": converged, "free_bits": free_bits, "support": support, "reasons": reasons,
            "guards": guards, "pure": pure, "unique_sample": unique_sample,
            "constants": [um.PROBABILITY_VECTOR_CONVERGED, um.INSUFFICIENT_UNIQUE_SUPPORT]}


def legacy_wrapper_outputs():
    """Direct calls of the two legacy wrappers on paths the classes never reach (duplicate
    prevention, real-valued genes, a non-empty archive): output, gene type and RNG consumption."""
    um = umda_module()
    build(ARGS["cases"]["moumda_margin_on"], 1)  # creator.Individual with weights (1, -1)
    generator = np.random.RandomState(12345)

    def population(make_genes):
        found = []
        for i in range(12):
            ind = creator.Individual(make_genes())
            ind.fitness.values = (float(sum(ind)), float(i % 5))
            found.append(ind)
        return found

    binary = population(lambda: [int(b) for b in generator.randint(0, 2, size=10)])
    # 4 free bits (16 genotypes) for 12 unique offspring: duplicate rejection really happens.
    narrow = population(lambda: [int(b) for b in generator.randint(0, 2, size=4)] + [1] * 6)
    real = population(lambda: [float(x) for x in generator.uniform(-5.0, 5.0, size=6)])
    # Archive members non-dominated by every population member (value above 10, or weight below 0).
    seeded_archive = []
    for bits, fitness in (([0] * 10, (11.0, 4.0)), ([1] * 10, (0.0, -1.0))):
        ind = creator.Individual(bits)
        ind.fitness.values = fitness
        seeded_archive.append(ind)

    def fingerprint(solutions, archive=None):
        found = {"size": len(solutions), "gene_type": type(solutions[0][0]).__name__,
                 "solutions_sha": sha([list(s) for s in solutions]),
                 "next_draw": float(np.random.rand())}
        if archive is not None:
            found["archive_sha"] = sha([[list(a), list(a.fitness.values)] for a in archive])
        return found

    calls = {
        "full binary margin": lambda: um.mo_umda_update_full(10, binary, 12, 6, None, prob_margin=True),
        "full binary unique": lambda: um.mo_umda_update_full(10, narrow, 12, 6, None, prob_margin=False,
                                                             prevent_duplicates=True),
        "full real": lambda: um.mo_umda_update_full(6, real, 12, 6, None),
        "full real unique": lambda: um.mo_umda_update_full(6, real, 12, 6, None, prevent_duplicates=True),
        "archive binary empty": lambda: um.mo_umda_update_with_archive(10, binary, 12, 6, None, archive=[]),
        "archive binary seeded": lambda: um.mo_umda_update_with_archive(10, binary, 12, 6, None,
                                                                        archive=seeded_archive, prob_margin=False),
        "archive real": lambda: um.mo_umda_update_with_archive(6, real, 12, 6, None, archive=[]),
    }
    found = {}
    for name, call in calls.items():
        seed_all(7)
        out = call()
        found[name] = fingerprint(*out) if isinstance(out, tuple) else fingerprint(out)
    return found


# ---------------------------------------------------------------- 15. commit on success

class FailingFitness:
    """A fitness function that raises on its `fail_on`-th call."""

    def __init__(self, fail_on):
        self.calls = 0
        self.fail_on = fail_on

    def __call__(self, individual, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("injected evaluation failure")
        return (float(sum(individual)), float(sum(individual)))


def commit_on_success():
    found = {}
    for name in ("moumda_margin_on", "moumda_margin_off", "pareto_archive_margin_on"):
        algo = build(ARGS["cases"][name], 1)
        record = {"vector_before_first_generation": None if algo.probability_vector is None else "set"}

        # A successful generation stores exactly the vector it sampled from.
        expected = expected_generating_vector(algo)
        algo.gens += 1
        algo.perform_generation()
        algo.record_state_pareto(algo.population)
        record["stored_vector_is_generating_vector"] = bool(np.array_equal(algo.probability_vector, expected))

        # A generation whose evaluation raises commits nothing of its own.
        before = {"population": algo.population, "genotypes": genotypes(algo.population), "evals": algo.evals,
                  "vector": algo.probability_vector, "archive": getattr(algo, "archive", None)}
        algo.fitness_function = (FailingFitness(3), {})
        try:
            algo.perform_generation()
            record["raised"] = None
        except RuntimeError as exc:
            record["raised"] = str(exc)
        record["population_object_kept"] = algo.population is before["population"]
        record["population_genotypes_kept"] = genotypes(algo.population) == before["genotypes"]
        record["evals_kept"] = algo.evals == before["evals"]
        record["vector_object_kept"] = algo.probability_vector is before["vector"]
        if hasattr(algo, "archive"):
            # The archive update is part of constructing the model, before evaluation (today's timing).
            record["archive_updated_before_evaluation"] = algo.archive is not before["archive"]
        found[name] = record
    return found


# ---------------------------------------------------------------- 5/6/7. convergence stop

COLLAPSED_GENOTYPE = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]


def moumda_case(target="MoUMDA", pop_size=20, select_size=10, init_args=None, **params):
    return {"init_args": dict({"_target_": MO_PREFIX + target, "pop_size": pop_size, "select_size": select_size,
                               "prob_margin": False}, **(init_args or {})),
            "gen_limit": 50, "record_every_gen": True, "params": params}


def inject_identical_population(algo, genotype):
    """Replace the population with copies of one genotype, each with its own noisy evaluation."""
    injected = []
    for _ in range(algo.pop_size):
        ind = creator.Individual(list(genotype))
        ind.fitness.values = algo.toolbox.evaluate(ind)
        injected.append(ind)
    algo.population = injected


def records(algo):
    return len(algo.true_pf_hypervolumes)


def stop_check(algo):
    """One stop_condition() call, reporting whether it drew RNG."""
    before = rng_state()
    stopped = algo.stop_condition()
    return stopped, rng_state() == before


def step(algo):
    algo.gens += 1
    algo.perform_generation()
    algo.record_state_pareto(algo.population)


def ordinary_collapse():
    algo = build(moumda_case(), 1)
    inject_identical_population(algo, COLLAPSED_GENOTYPE)
    found = {"first_check": stop_check(algo), "evals_before": algo.evals}
    step(algo)
    found.update({
        "gens": algo.gens, "evals": algo.evals, "records": records(algo),
        "trigger_before_next_check": algo.stop_trigger,
        "vector": [float(p) for p in algo.probability_vector],
        "genotypes": sorted({tuple(int(x) for x in ind) for ind in algo.population}),
        "distinct_fitnesses": len({tuple(ind.fitness.values) for ind in algo.population}),
        "recorded_front_genotypes": sorted({tuple(int(x) for x in ind) for ind in algo.pareto_solutions[-1]}),
    })
    found["next_check"] = stop_check(algo)
    found["trigger"] = algo.stop_trigger
    found["gens_after"], found["evals_after"], found["records_after"] = algo.gens, algo.evals, records(algo)

    plain = build(moumda_case(), 1)
    inject_identical_population(plain, COLLAPSED_GENOTYPE)
    plain.run()
    found["run"] = {"gens": plain.gens, "evals": plain.evals, "records": records(plain),
                    "trigger": plain.stop_trigger}
    return found


def archive_collapse():
    algo = build(moumda_case("MoUMDA_ParetoArchive"), 1)
    dominant = creator.Individual(list(COLLAPSED_GENOTYPE))
    dominant.fitness.values = (1.0e6, -1.0e6)  # dominates every knapsack evaluation
    algo.archive = [dominant]
    diverse_parents = sorted({tuple(int(x) for x in ind) for ind in algo.population})
    found = {"first_check": stop_check(algo), "population_diverse": len(diverse_parents) > 1}
    archive_before_generation = algo.archive
    step(algo)
    found.update({
        "archive_replaced_by_generation": algo.archive is not archive_before_generation,
        "archive_genotypes": sorted(tuple(int(x) for x in ind) for ind in algo.archive),
        "gens": algo.gens, "evals": algo.evals, "records": records(algo),
        "trigger_before_next_check": algo.stop_trigger,
        "vector": [float(p) for p in algo.probability_vector],
        "population_genotypes": sorted({tuple(int(x) for x in ind) for ind in algo.population}),
        "population_evaluated": all(ind.fitness.valid for ind in algo.population),
        "recorded_front_genotypes": sorted({tuple(int(x) for x in ind) for ind in algo.pareto_solutions[-1]}),
    })
    archive_object = algo.archive
    archive_snapshot = [[list(a), list(a.fitness.values)] for a in algo.archive]
    checks = [stop_check(algo), stop_check(algo)]
    found.update({
        "next_checks": checks, "trigger": algo.stop_trigger,
        "archive_object_kept": algo.archive is archive_object,
        "archive_contents_kept": [[list(a), list(a.fitness.values)] for a in algo.archive] == archive_snapshot,
        "gens_after": algo.gens, "evals_after": algo.evals, "records_after": records(algo),
    })

    # The same diverse start on ordinary MoUMDA does not converge after one generation.
    ordinary = build(moumda_case(), 1)
    step(ordinary)
    found["ordinary_same_start_stops"] = ordinary.stop_condition()
    return found


def generic_precedence():
    """A converged probability_vector and a generic criterion on the same check: generic wins."""
    found = {}
    for name, params in (("gen_limit", {"gen_limit": 1}), ("eval_limit", {"eval_limit": 40}),
                         ("no_improvement", {"stop_without_improvement_in_gens": 1})):
        algo = build(moumda_case(**params), 1)
        inject_identical_population(algo, COLLAPSED_GENOTYPE)
        algo.run()
        found[name] = {"gens": algo.gens, "trigger": algo.stop_trigger,
                       "vector_converged": umda_module().is_probability_vector_converged(algo.probability_vector)}
    return found


def real_objectives(individual):
    return (float(sum(individual)), float(sum(individual)))


def real_valued_path():
    """Gaussian generations leave probability_vector None, so they never trigger the stop."""
    algo = _locate(MO_PREFIX + "MoUMDA")(
        pop_size=10, select_size=5, sol_length=3, opt_weights=(1.0, -1.0), gen_limit=50,
        attr_function=getattr(algorithms, "Rastrigin_attribute"), fitness_function=(real_objectives, {}),
    )
    for ind in algo.population:  # a collapsed real-valued population
        ind[:] = [0.5, 0.5, 0.5]
        ind.fitness.values = real_objectives(ind)
    algo.gens += 1
    algo.perform_generation()
    return {"vector": algo.probability_vector, "stops": algo.stop_condition(), "trigger": algo.stop_trigger,
            "gene_type": type(algo.population[0][0]).__name__}


# ---------------------------------------------------------------- duplicate-free MoUMDA (Stage 5)

def duplicate_free_case(**params):
    """MoUMDA_noDuplicates on the 10-bit knapsack with select_size == pop_size == 10, so the probability
    vector is exactly the mean of the (injected) population."""
    return moumda_case("MoUMDA_noDuplicates", pop_size=10, select_size=10, **params)


def inject_free_bits(algo, free_bits):
    """Individual i carries the low `free_bits` bits of i, the other bits fixed at 1: with 10
    individuals every one of those bits is mixed, so the vector has exactly `free_bits` free bits."""
    injected = []
    for value in range(algo.pop_size):
        ind = creator.Individual([(value >> b) & 1 for b in range(free_bits)] + [1] * (algo.sol_length - free_bits))
        ind.fitness.values = algo.toolbox.evaluate(ind)
        injected.append(ind)
    algo.population = injected


def state_snapshot(algo):
    return {"gens": algo.gens, "evals": algo.evals, "records": records(algo), "population": algo.population,
            "genotypes": genotypes(algo.population), "vector": algo.probability_vector, "rng": rng_state()}


def unchanged_since(algo, before):
    now = state_snapshot(algo)
    return {"gens": now["gens"] == before["gens"], "evals": now["evals"] == before["evals"],
            "records": now["records"] == before["records"],
            "population_object": now["population"] is before["population"],
            "population_genotypes": now["genotypes"] == before["genotypes"],
            "probability_vector_object": now["vector"] is before["vector"], "rng": now["rng"] == before["rng"]}


class Instrumented:
    """Counts parent selections, vector calculations and sampler calls in the UMDA module, and records
    which vector object each sampler call received. Restores the originals on exit."""

    def __enter__(self):
        um = umda_module()
        self.um, self.calls, self.sampled_vectors = um, {"select": 0, "vector": 0, "sample": 0}, []
        self.saved = {"select": um.tools.selNSGA2, "vector": um.calculate_probability_vector,
                      "sample": um.sample_from_probability_vector}

        def count(name, fn):
            def wrapped(*args, **kwargs):
                self.calls[name] += 1
                if name == "sample":
                    self.sampled_vectors.append(args[0])
                return fn(*args, **kwargs)
            return wrapped

        um.tools.selNSGA2 = count("select", self.saved["select"])
        um.calculate_probability_vector = count("vector", self.saved["vector"])
        um.sample_from_probability_vector = count("sample", self.saved["sample"])
        return self

    def __exit__(self, *exc):
        self.um.tools.selNSGA2 = self.saved["select"]
        self.um.calculate_probability_vector = self.saved["vector"]
        self.um.sample_from_probability_vector = self.saved["sample"]
        return False


def duplicate_free_preflight_stops():
    """support = 1 and 1 < support < pop_size: stop before the generation, committing nothing."""
    found = {}
    for name, free_bits in (("support_1", 0), ("support_8", 3)):
        algo = build(duplicate_free_case(), 1)
        inject_free_bits(algo, free_bits)
        before = state_snapshot(algo)
        with Instrumented() as probe_calls:
            stopped = algo.stop_condition()
        prepared = algo._prepared_probability_vector
        found[name] = {
            "stopped": stopped, "trigger": algo.stop_trigger, "unchanged": unchanged_since(algo, before),
            "calls": probe_calls.calls,
            "prepared_is_population_mean": bool(np.array_equal(prepared, np.mean(algo.population, axis=0))),
            "prepared_free_bits": umda_module().count_free_bits(prepared),
        }
        # run() from this state ends immediately, with no generation counted.
        plain = build(duplicate_free_case(), 1)
        inject_free_bits(plain, free_bits)
        evals = plain.evals
        plain.run()
        found[name]["run"] = {"gens": plain.gens, "evals_added": plain.evals - evals, "records": records(plain),
                              "trigger": plain.stop_trigger}
    return found


def duplicate_free_reuse():
    """Sufficient support (16 genotypes for 10): calculate once, inspect once, use once."""
    algo = build(duplicate_free_case(), 1)
    inject_free_bits(algo, 4)
    evals = algo.evals
    with Instrumented() as probe_calls:
        first = algo.stop_condition()
        prepared = algo._prepared_probability_vector
        after_first = dict(probe_calls.calls)
        second = algo.stop_condition()
        after_second = dict(probe_calls.calls)
        same_prepared = algo._prepared_probability_vector is prepared
        algo.gens += 1
        algo.perform_generation()
        after_generation = dict(probe_calls.calls)
        sampled = probe_calls.sampled_vectors
    keys = [tuple(int(x) for x in ind) for ind in algo.population]
    return {
        "checks": [first, second], "calls_after_first_check": after_first,
        "calls_after_second_check": after_second, "prepared_kept_between_checks": same_prepared,
        "calls_after_generation": after_generation,
        "sampler_received_prepared_object": len(sampled) == 1 and sampled[0] is prepared,
        "committed_vector_is_prepared_object": algo.probability_vector is prepared,
        "prepared_cleared": algo._prepared_probability_vector is None,
        "population_size": len(keys), "unique_genotypes": len(set(keys)),
        "within_support": all(all(bit == 1 for bit in key[4:]) for key in keys),
        "evals_added": algo.evals - evals,
    }


def duplicate_free_commit_on_success():
    """Evaluation fails after a successful preflight: nothing is committed, the prepared vector stays."""
    algo = build(duplicate_free_case(), 1)
    inject_free_bits(algo, 4)
    stopped = algo.stop_condition()
    prepared = algo._prepared_probability_vector
    before = state_snapshot(algo)
    algo.fitness_function = (FailingFitness(3), {})
    try:
        algo.perform_generation()
        raised = None
    except RuntimeError as exc:
        raised = str(exc)
    after = unchanged_since(algo, before)
    del after["rng"]  # sampling did draw RNG before evaluation failed
    return {"stopped": stopped, "raised": raised, "unchanged": after,
            "prepared_kept": algo._prepared_probability_vector is prepared}


def duplicate_free_direct_generation_guard():
    """perform_generation called directly (no stop check) on an impossible vector raises, commits nothing."""
    algo = build(duplicate_free_case(), 1)
    inject_free_bits(algo, 3)
    before = state_snapshot(algo)
    outcome = raises_value_error(algo.perform_generation)
    return {"outcome": outcome, "unchanged": unchanged_since(algo, before),
            "prepared": algo._prepared_probability_vector}


def duplicate_free_construction():
    """Constructor semantics and duplicate-free initialisation, including the invalid configurations."""
    cfg = case_config(duplicate_free_case())

    def construct(target, **overrides):
        params = runner_algo_params(cfg)
        params.update(overrides.pop("params", {}))
        seed_all(1)
        return _locate(MO_PREFIX + target)(pop_size=overrides.pop("pop_size", 10), select_size=5, **overrides,
                                           **params)

    def outcome(fn):
        try:
            algo = fn()
        except ValueError as exc:
            return {"error": "ValueError", "message": str(exc)}
        keys = [tuple(int(x) for x in ind) for ind in algo.population]
        return {"name": algo.name, "type": algo.type, "prevent_duplicates": algo.prevent_duplicates,
                "population_size": len(keys), "unique_genotypes": len(set(keys)), "evals": algo.evals}

    start = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return {
        "default": outcome(lambda: construct("MoUMDA_noDuplicates")),
        "explicit_true": outcome(lambda: construct("MoUMDA_noDuplicates", prevent_duplicates=True)),
        "explicit_false": outcome(lambda: construct("MoUMDA_noDuplicates", prevent_duplicates=False)),
        "moumda_flag_true": outcome(lambda: construct("MoUMDA", prevent_duplicates=True)),
        "capacity_3_bits": outcome(lambda: construct("MoUMDA_noDuplicates", params={"sol_length": 3})),
        "starting_solution_pop_10": outcome(lambda: construct("MoUMDA_noDuplicates",
                                                              params={"starting_solution": start})),
        "starting_solution_pop_1": outcome(lambda: construct("MoUMDA_noDuplicates", pop_size=1,
                                                             params={"starting_solution": start})),
        "ordinary_3_bits": outcome(lambda: construct("MoUMDA", params={"sol_length": 3})),
    }


def duplicate_free_natural_runs():
    """Natural margin-off runs: every recorded population is duplicate-free, and MoUMDA(prevent_duplicates=True)
    is the same algorithm as MoUMDA_noDuplicates for the same seed and inputs."""
    found = {}
    for seed in ARGS["seeds"]:
        runs = {}
        for target, extra in (("MoUMDA_noDuplicates", {}), ("MoUMDA", {"prevent_duplicates": True})):
            algo = build(moumda_case(target, init_args=extra, gen_limit=150), seed)
            initial_duplicates = duplicate_genotypes(algo.population)
            per_generation = drive(algo)
            summary = run_summary(algo)
            summary.pop("class")
            runs[target] = {"summary": summary, "per_generation": per_generation,
                            "initial_duplicates": initial_duplicates}
        a, b = runs["MoUMDA_noDuplicates"], runs["MoUMDA"]
        found[str(seed)] = {
            "initial_duplicate_genotypes": a["initial_duplicates"],
            "gens": a["summary"]["gens"], "trigger": a["summary"]["stop_trigger"],
            "max_duplicate_genotypes": max((g["population_duplicate_genotypes"] for g in a["per_generation"]),
                                           default=0),
            "records": len(a["summary"]["true_pf_hypervolumes"]),
            "flag_spelling_identical": a["summary"] == b["summary"] and a["per_generation"] == b["per_generation"]
                                       and a["initial_duplicates"] == b["initial_duplicates"],
        }
    return found


report = {
    "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "deap": deap.__version__,
                    "hydra": hydra.__version__},
    "duplicate_free_preflight_stops": section(duplicate_free_preflight_stops),
    "duplicate_free_reuse": section(duplicate_free_reuse),
    "duplicate_free_commit_on_success": section(duplicate_free_commit_on_success),
    "duplicate_free_direct_generation_guard": section(duplicate_free_direct_generation_guard),
    "duplicate_free_construction": section(duplicate_free_construction),
    "duplicate_free_natural_runs": section(duplicate_free_natural_runs),
    "ordinary_collapse": section(ordinary_collapse),
    "archive_collapse": section(archive_collapse),
    "generic_precedence": section(generic_precedence),
    "real_valued_path": section(real_valued_path),
    "commit_on_success": section(commit_on_success),
    "helper_contracts": section(helper_contracts),
    "legacy_wrapper_outputs": section(legacy_wrapper_outputs),
    "public_names": section(public_names),
    "config_sweep": section(config_sweep),
    "characterisation": section(characterisation),
    "run_equivalence": section(run_equivalence),
    "archive_duplicate_semantics": section(archive_duplicate_semantics),
    "no_duplicates_generation": section(no_duplicates_generation),
    "no_duplicates_initial_population": section(no_duplicates_initial_population),
}
print(json.dumps(report))
'''


def run_probe() -> dict:
    """Run the probe once in a fresh harness subprocess and return its report.

    Also used, unchanged, to capture `baselines/mo_algorithms.json`.
    """
    args = {
        "fence": str(HARNESS_DIR / "fence.py"),
        "workspace": str(WORKSPACE),
        "mo_prefix": MO_PREFIX,
        "public_names": list(PUBLIC_MO_NAMES),
        "known_unresolvable": sorted(KNOWN_UNRESOLVABLE_TARGETS),
        "sweep_gens": CONFIG_SWEEP_GENS,
        "sweep_pop_size": CONFIG_SWEEP_POP_SIZE,
        "sweep_select_size": CONFIG_SWEEP_SELECT_SIZE,
        "cases": CHARACTERISATION_CASES,
        "seeds": SEEDS,
        "support_cases": SUPPORT_CASES,
        "reason_cases": REASON_CASES,
    }
    root = make_temp_root()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE, json.dumps(args)],
            cwd=str(root),
            env=child_env(root),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert completed.returncode == 0, f"MO algorithm probe failed:\n{completed.stderr[-4000:]}"
        return json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def probe() -> dict:
    return run_probe()


def _section(probe: dict, name: str):
    """A probe section; a crash inside it is a real failure, never an expected (xfail) one."""
    found = probe[name]
    if isinstance(found, dict) and "error" in found:
        raise RuntimeError(f"probe section {name!r} crashed:\n{found['error']}")
    return found


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE_PATH.is_file(), f"missing frozen baseline {BASELINE_PATH}; it is never re-derived by a test run"
    return json.loads(BASELINE_PATH.read_text())


# ------------------------------------------------------------------------------ 1. imports


def test_public_mo_names_resolve_at_flat_paths(probe):
    found = _section(probe, "public_names")

    assert set(found) == set(PUBLIC_MO_NAMES)
    for name, info in found.items():
        module = info["module"]
        assert module == MO_PREFIX[:-1] or module.startswith(MO_PREFIX), (
            f"{MO_PREFIX}{name} is defined in {module}, outside the multi_objective package"
        )
        assert info["callable"], f"{MO_PREFIX}{name} is not callable"
        assert info["is_class"] is (name in MO_ALGORITHM_CLASSES | {"OptimisationAlgorithm", "MoUMDABase"}), name
        assert info["is_package_namespace_object"], (
            f"noisyvis.algorithms.{name} is not the object at {MO_PREFIX}{name}"
        )


# ------------------------------------------------------------------------------ 2. configs


def test_every_configured_mo_target_instantiates_and_runs(probe, note):
    outcomes = _section(probe, "config_sweep")

    # Dangling config symlinks are a config-hygiene problem owned by test_config_resolution.py;
    # they cannot hold an MO target, so they are reported here rather than failed.
    dangling = sorted(path for path, o in outcomes.items() if o.get("dangling"))
    if dangling:
        note(f"test_mo_algorithms: skipped dangling config symlinks: {dangling}")
    outcomes = {path: o for path, o in outcomes.items() if not o.get("dangling")}

    assert len(outcomes) > 20, f"only {len(outcomes)} MO configs found; the walk looks broken"
    errors = {path: o["error"] for path, o in outcomes.items() if "error" in o}
    assert not errors, "configured MO targets failed to instantiate or run:\n" + "\n".join(
        f"--- {path}\n{error}" for path, error in errors.items()
    )

    skipped = {o["target"] for o in outcomes.values() if "skipped" in o}
    assert skipped == set(KNOWN_UNRESOLVABLE_TARGETS), f"unexpected skips: {sorted(skipped)}"

    ran = [o for o in outcomes.values() if "skipped" not in o]
    assert {o["class"] for o in ran} == MO_ALGORITHM_CLASSES, sorted({o["class"] for o in ran})
    # Every run reaches the generation limit, except that a MoUMDA-family run may legitimately stop
    # earlier: ordinary/ParetoArchive runs on the post-generation convergence stop (MO refactor Stage 4)
    # after at least one completed generation, duplicate-free runs on the pre-generation support stops
    # (Stage 5) after any number of generations.
    def ran_as_expected(o):
        if o["gens"] == CONFIG_SWEEP_GENS and o["stop_trigger"] == "gen_limit":
            return True
        if o["gens"] >= CONFIG_SWEEP_GENS:
            return False
        if o["type"] == "MoUMDA_noDuplicates":
            return o["stop_trigger"] in {"probability_vector_converged", "insufficient_unique_support"}
        return (o["stop_trigger"] == "probability_vector_converged" and o["gens"] >= 1
                and o["class"] in {"MoUMDA", "MoUMDA_ParetoArchive"})

    wrong = {path: o for path, o in outcomes.items() if "skipped" not in o and not ran_as_expected(o)}
    assert not wrong, f"configs that did not run as expected: {wrong}"
    early = sorted(f"{path} ({o['stop_trigger']})" for path, o in outcomes.items()
                   if "skipped" not in o and o["stop_trigger"] != "gen_limit")
    if early:
        note(f"test_mo_algorithms: config sweep runs that stopped early: {early}")


# ------------------------------------------------------------------------------ 3. characterisation


def test_baseline_spec_matches_this_module(baseline):
    """The frozen baseline was captured with exactly these cases and seeds."""
    assert baseline["seeds"] == SEEDS
    assert baseline["cases_spec"] == CHARACTERISATION_CASES


def test_stepped_loop_is_run(probe):
    """The characterisation steps the loop itself; that loop must be exactly `run()`."""
    equivalence = _section(probe, "run_equivalence")
    assert equivalence == {name: True for name in CHARACTERISATION_CASES}, equivalence


def first_collapsed_generation(run: dict):
    """The first baseline generation sampled from a fully 0/1 probability vector, or None."""
    return next((g["gen"] for g in run["per_generation"] if g.get("generating_vector_collapsed")), None)


def truncated_at_convergence(want: dict, stop_gen: int) -> dict:
    """What a pre-refactor baseline run becomes once the convergence stop (MO refactor Stage 4) ends
    it right after generation `stop_gen`, the first one sampled from a converged vector.

    Only defined for runs that record every generation (one record per generation), which is why
    the margin-off characterisation cases do.
    """
    summary = want["summary"]
    per_generation = want["per_generation"][:stop_gen]
    records = stop_gen
    runs, remaining = [], records  # n_gens_pareto_best: the baseline's run lengths cut at `records`
    for length in summary["n_gens_pareto_best"]:
        runs.append(min(length, remaining))
        remaining -= runs[-1]
        if remaining == 0:
            break
    truncated = dict(summary)
    truncated.update({
        "gens": stop_gen,
        "evals": per_generation[-1]["evals"],
        "stop_trigger": "probability_vector_converged",
        "n_gens_pareto_best": runs,
        "final_population_genotypes_sha": per_generation[-1]["population_genotypes_sha"],
        "final_population_fitnesses_sha": per_generation[-1]["population_fitnesses_sha"],
    })
    for key in ("noisy_pf_noisy_hypervolumes", "noisy_pf_true_hypervolumes", "true_pf_hypervolumes",
                "pareto_solutions_sha", "pareto_fitnesses_sha", "pareto_true_fitnesses_sha",
                "true_pareto_solutions_sha", "true_pareto_fitnesses_sha"):
        truncated[key] = summary[key][:records]
    return {"summary": truncated, "per_generation": per_generation}


@pytest.mark.parametrize("case", sorted(CHARACTERISATION_CASES))
def test_characterisation_matches_baseline(probe, baseline, case):
    """Exact, except that a run whose probability vector fully converges now stops right after the
    first generation sampled from it: its record must then be exactly that prefix of the baseline."""
    observed = _section(probe, "characterisation")[case]
    expected = baseline["cases"][case]

    assert set(observed) == set(expected) == {str(seed) for seed in SEEDS}
    for seed in sorted(expected):
        got, want = observed[seed], expected[seed]
        stop_gen = first_collapsed_generation(want)
        if stop_gen is not None and stop_gen < want["summary"]["gens"]:
            assert CHARACTERISATION_CASES[case]["record_every_gen"], f"{case}: prefix needs per-generation records"
            want = truncated_at_convergence(want, stop_gen)
        assert got["summary"] == want["summary"], (
            f"{case} seed {seed}: run summary differs from the frozen pre-refactor baseline"
        )
        assert len(got["per_generation"]) == len(want["per_generation"]), (
            f"{case} seed {seed}: {len(got['per_generation'])} generations, baseline has "
            f"{len(want['per_generation'])}"
        )
        for got_gen, want_gen in zip(got["per_generation"], want["per_generation"]):
            assert got_gen == want_gen, (
                f"{case} seed {seed}: generation {want_gen['gen']} differs from the frozen baseline"
            )


def test_natural_convergence_points_are_inside_the_baseline_window(baseline):
    """The margin-off MoUMDA baseline runs really do collapse inside their window, so the prefix
    comparison above exercises the convergence stop; margin-on runs and the archive variant do not."""
    collapse = {case: {seed: first_collapsed_generation(run) for seed, run in runs.items()}
                for case, runs in baseline["cases"].items()}
    assert collapse["moumda_margin_off"] == {"1": 30, "2": 42}
    for case in ("moumda_margin_on", "pareto_archive_margin_on", "pareto_archive_margin_off", "nsga2"):
        assert set(collapse[case].values()) == {None}, (case, collapse[case])


# ------------------------------------------------------------------------------ 8. archive


def test_pareto_archive_update_keeps_genotype_duplicates(probe):
    """Deliberately unchanged by the refactor: the archive does not deduplicate by genotype."""
    semantics = _section(probe, "archive_duplicate_semantics")

    # b duplicates a exactly; c is a noisy copy mutually non-dominated with a; d is dominated.
    assert semantics["kept"] == ["a", "b", "c", "e"]
    assert semantics["kept_duplicate_genotypes"] == 2
    assert semantics["returns_new_list"] is True
    assert semantics["archive_argument_unchanged"] is True
    assert semantics["empty_inputs"] == []


# ------------------------------------------------------------------------------ 4. helpers


def test_probability_vector_helpers(probe):
    found = _section(probe, "helper_contracts")

    assert found["constants"] == ["probability_vector_converged", "insufficient_unique_support"]
    # Exact 0/1 comparison: a margin-clipped vector is not converged; empty and None are not either.
    assert found["converged"] == {"partial": False, "collapsed": True, "empty": False,
                                  "margin_clipped": False, "None": False}
    assert found["free_bits"] == {"partial": 1, "collapsed": 0, "empty": 0, "margin_clipped": 4}
    assert found["support"] == EXPECTED_SUPPORT
    assert found["reasons"] == EXPECTED_REASONS
    assert found["pure"] == {"inputs_unchanged": True, "rng_unchanged": True}


# ------------------------------------------------------------------------------ 14. direct calls


def test_direct_calls_with_impossible_unique_support_raise(probe):
    """Direct misuse fails fast instead of entering an impossible rejection loop."""
    found = _section(probe, "helper_contracts")

    assert found["guards"] == {"sample k=3 pop=10": "ValueError", "sample k=0 pop=10": "ValueError",
                               "mo_umda_update_full collapsed": "ValueError"}
    # With sufficient support (16 genotypes for 10) the sampler returns unique in-support genotypes.
    assert found["unique_sample"] == {"size": 10, "unique": 10, "within_support": True}


def test_legacy_wrappers_unchanged(probe):
    assert _section(probe, "legacy_wrapper_outputs") == LEGACY_WRAPPER_OUTPUTS


# ------------------------------------------------------------------------------ 5/6/7. convergence stop

COLLAPSED = (1, 0, 1, 1, 0, 0, 1, 0, 1, 0)
NO_RNG_NOT_STOPPED = [False, True]  # stop_check(): (stopped, RNG untouched)
NO_RNG_STOPPED = [True, True]


def test_ordinary_moumda_keeps_the_converged_generation_then_stops(probe):
    found = _section(probe, "ordinary_collapse")

    # Generation 1 runs normally from the collapsed parents: sampled, evaluated, counted, recorded.
    assert found["first_check"] == NO_RNG_NOT_STOPPED
    assert (found["gens"], found["evals"], found["records"]) == (1, found["evals_before"] + 20, 1)
    assert found["vector"] == [float(bit) for bit in COLLAPSED]
    assert found["genotypes"] == [list(COLLAPSED)]
    assert found["distinct_fitnesses"] > 1, "noisy copies of one genotype should get different observed fitness"
    assert found["recorded_front_genotypes"] == [list(COLLAPSED)]
    # Only the following check stops, and it neither draws RNG nor advances the run.
    assert found["trigger_before_next_check"] == ""
    assert found["next_check"] == NO_RNG_STOPPED
    assert found["trigger"] == "probability_vector_converged"
    assert (found["gens_after"], found["evals_after"], found["records_after"]) == (1, found["evals"], 1)
    assert found["run"] == {"gens": 1, "evals": found["evals"], "records": 1,
                            "trigger": "probability_vector_converged"}


def test_pareto_archive_keeps_the_converged_generation_and_stop_checks_leave_the_archive(probe):
    found = _section(probe, "archive_collapse")

    assert found["first_check"] == NO_RNG_NOT_STOPPED
    assert found["population_diverse"] is True
    # The real archive update happens inside the generation, as before; its single genotype gives
    # a converged vector, whose sampled population is evaluated, and the archive is recorded.
    assert found["archive_replaced_by_generation"] is True
    assert found["archive_genotypes"] == [list(COLLAPSED)]
    assert (found["gens"], found["records"]) == (1, 1)
    assert found["vector"] == [float(bit) for bit in COLLAPSED]
    assert found["population_genotypes"] == [list(COLLAPSED)]
    assert found["population_evaluated"] is True
    assert found["recorded_front_genotypes"] == [list(COLLAPSED)]
    assert found["trigger_before_next_check"] == ""
    # The following checks stop without drawing RNG or touching the archive, however often called.
    assert found["next_checks"] == [NO_RNG_STOPPED, NO_RNG_STOPPED]
    assert found["trigger"] == "probability_vector_converged"
    assert found["archive_object_kept"] is True and found["archive_contents_kept"] is True
    assert (found["gens_after"], found["evals_after"], found["records_after"]) == (1, found["evals"], 1)
    assert found["ordinary_same_start_stops"] is False


def test_generic_stop_criteria_take_precedence_over_convergence(probe):
    found = _section(probe, "generic_precedence")

    assert found == {
        "gen_limit": {"gens": 1, "trigger": "gen_limit", "vector_converged": True},
        "eval_limit": {"gens": 1, "trigger": "eval_limit", "vector_converged": True},
        "no_improvement": {"gens": 1, "trigger": "no_improvement", "vector_converged": True},
    }


def test_real_valued_generations_never_trigger_the_convergence_stop(probe):
    assert _section(probe, "real_valued_path") == {"vector": None, "stops": False, "trigger": "",
                                                   "gene_type": "float"}


# ------------------------------------------------------------------------------ 15. commit on success


def test_generation_commits_only_after_successful_evaluation(probe):
    """probability_vector always describes a completed generation; a failed one commits nothing."""
    found = _section(probe, "commit_on_success")

    committed_nothing = {
        "vector_before_first_generation": None,
        "stored_vector_is_generating_vector": True,
        "raised": "injected evaluation failure",
        "population_object_kept": True,
        "population_genotypes_kept": True,
        "evals_kept": True,
        "vector_object_kept": True,
    }
    assert found["moumda_margin_on"] == committed_nothing
    assert found["moumda_margin_off"] == committed_nothing
    # The archive variant updates its archive while constructing the model, before evaluation,
    # exactly as before the refactor; that update is not rolled back.
    assert found["pareto_archive_margin_on"] == dict(committed_nothing, archive_updated_before_evaluation=True)


# ------------------------------------------------------------------------------ 9/10. duplicate-free MoUMDA
# Strict xfails from Stage 0 until Stage 5 fixed the defect.


def test_no_duplicates_generation_is_duplicate_free(probe):
    found = _section(probe, "no_duplicates_generation")

    # Preconditions of the scenario itself.
    if found["free_bits"] != 4 or not found["all_within_support"] or found["population_size"] != 10:
        raise RuntimeError(f"the injected scenario is not what the test assumes: {found}")

    assert found["unique_genotypes"] == found["population_size"], found
    # Exactly the accepted offspring are evaluated and counted.
    assert found["evals_added"] == found["evaluations"] == 10, found


def test_no_duplicates_initial_population_is_duplicate_free(probe):
    found = _section(probe, "no_duplicates_initial_population")

    if found["population_size"] != 10:
        raise RuntimeError(f"the initialisation scenario is not what the test assumes: {found}")

    assert found["unique_genotypes"] == found["population_size"], found
    # Duplicates were drawn and rejected, and rejected candidates were never evaluated.
    assert found["candidates_drawn"] > 10, found
    assert found["evals"] == found["evaluations"] == 10, found


_PREFLIGHT_UNCHANGED = {"gens": True, "evals": True, "records": True, "population_object": True,
                        "population_genotypes": True, "probability_vector_object": True, "rng": True}


def test_duplicate_free_preflight_stops_before_the_generation(probe):
    found = _section(probe, "duplicate_free_preflight_stops")

    for name, trigger, free_bits in (("support_1", "probability_vector_converged", 0),
                                     ("support_8", "insufficient_unique_support", 3)):
        result = found[name]
        assert result["stopped"] is True and result["trigger"] == trigger, (name, result)
        # Nothing but the prepared vector changes: no RNG, evaluation, counter, record or population.
        assert result["unchanged"] == _PREFLIGHT_UNCHANGED, (name, result)
        # One selection, one vector calculation, no sampling.
        assert result["calls"] == {"select": 1, "vector": 1, "sample": 0}, (name, result)
        # The classified vector is cached: exactly the mean of the (select_size == pop_size) parents.
        assert result["prepared_is_population_mean"] is True
        assert result["prepared_free_bits"] == free_bits
        assert result["run"] == {"gens": 0, "evals_added": 0, "records": 0, "trigger": trigger}, (name, result)


def test_duplicate_free_generation_uses_the_prepared_vector_once(probe):
    found = _section(probe, "duplicate_free_reuse")

    assert found["checks"] == [False, False]
    # The first check selects parents and calculates the vector once; nothing afterwards repeats it.
    assert found["calls_after_first_check"] == {"select": 1, "vector": 1, "sample": 0}
    assert found["calls_after_second_check"] == {"select": 1, "vector": 1, "sample": 0}
    assert found["prepared_kept_between_checks"] is True
    assert found["calls_after_generation"] == {"select": 1, "vector": 1, "sample": 1}
    # The classified array itself is sampled and then committed as the generation's vector.
    assert found["sampler_received_prepared_object"] is True
    assert found["committed_vector_is_prepared_object"] is True
    assert found["prepared_cleared"] is True
    assert (found["population_size"], found["unique_genotypes"], found["within_support"]) == (10, 10, True)
    assert found["evals_added"] == 10


def test_duplicate_free_generation_commits_only_after_successful_evaluation(probe):
    found = _section(probe, "duplicate_free_commit_on_success")

    assert found["stopped"] is False
    assert found["raised"] == "injected evaluation failure"
    assert found["unchanged"] == {key: True for key in _PREFLIGHT_UNCHANGED if key != "rng"}
    assert found["prepared_kept"] is True


def test_direct_duplicate_free_generation_with_impossible_support_raises(probe):
    found = _section(probe, "duplicate_free_direct_generation_guard")

    assert found["outcome"] == "ValueError"
    assert found["unchanged"] == _PREFLIGHT_UNCHANGED
    assert found["prepared"] is None


def test_duplicate_free_construction(probe):
    found = _section(probe, "duplicate_free_construction")

    valid = {"name": "MoUMDA_noDuplicates(p=10, μ=5)", "type": "MoUMDA_noDuplicates", "prevent_duplicates": True,
             "population_size": 10, "unique_genotypes": 10, "evals": 10}
    assert found["default"] == valid
    assert found["explicit_true"] == valid
    assert found["moumda_flag_true"] == valid
    assert found["explicit_false"] == {"error": "ValueError",
                                       "message": "MoUMDA_noDuplicates requires prevent_duplicates=True."}
    assert found["capacity_3_bits"]["error"] == "ValueError", found["capacity_3_bits"]
    assert found["starting_solution_pop_10"]["error"] == "ValueError", found["starting_solution_pop_10"]
    assert found["starting_solution_pop_1"] == dict(valid, name="MoUMDA_noDuplicates(p=1, μ=5)",
                                                    population_size=1, unique_genotypes=1, evals=1)
    # Duplicates remain allowed without the flag: 10 individuals over 8 genotypes.
    ordinary = found["ordinary_3_bits"]
    assert ordinary["type"] == "MoUMDA" and ordinary["prevent_duplicates"] is False
    assert ordinary["population_size"] == 10 and ordinary["unique_genotypes"] <= 8


def test_duplicate_free_natural_runs(probe, note):
    found = _section(probe, "duplicate_free_natural_runs")

    for seed, result in found.items():
        assert result["initial_duplicate_genotypes"] == 0, (seed, result)
        assert result["max_duplicate_genotypes"] == 0, (seed, result)
        assert result["trigger"] in {"gen_limit", "probability_vector_converged", "insufficient_unique_support"}
        assert result["records"] == result["gens"], (seed, result)
        # MoUMDA(prevent_duplicates=True) is the same algorithm as MoUMDA_noDuplicates.
        assert result["flag_spelling_identical"] is True, (seed, result)
    note(f"test_mo_algorithms: duplicate-free natural runs: "
         f"{ {seed: (r['gens'], r['trigger']) for seed, r in found.items()} }")
