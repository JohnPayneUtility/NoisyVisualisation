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

Known defect, pinned as strict xfails until the stage that fixes it (MO refactor Stage 5):

9. `MoUMDA_noDuplicates` never passes `prevent_duplicates` to the sampler, so its offspring contain
   duplicate genotypes: it behaves exactly like `MoUMDA` and differs only in `name`/`type`.
10. Its initial population is built by the generic initialiser, so it is not duplicate-free either.

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
    "OptimisationAlgorithm", "SEMO", "MoUMDA", "MoUMDA_noDuplicates", "MoUMDA_ParetoArchive", "NSGA2",
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

def generating_vector_collapsed(algo):
    """Whether the probability vector the next generation will sample from is entirely 0/1.

    Recomputes, without committing anything, exactly what perform_generation is about to compute:
    selNSGA2 parents (for the archive variant, the non-dominated archive update), their mean, and
    the margin clip. Consumes no RNG, which is asserted. None for algorithms with no UMDA model.
    """
    if not hasattr(algo, "select_size"):
        return None
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
        raise AssertionError("the collapse probe consumed RNG")
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
    seed_all(1)
    algo.perform_generation()
    keys = [tuple(int(x) for x in ind) for ind in algo.population]
    return {
        "free_bits": int(np.count_nonzero((probs > 0.0) & (probs < 1.0))),
        "population_size": len(keys),
        "unique_genotypes": len(set(keys)),
        "all_within_support": all(not any(key[4:]) for key in keys),
    }


def no_duplicates_initial_population():
    """sol_length 4 has 16 genotypes, so 10 unique initial individuals are possible."""
    cfg = case_config({"init_args": {"_target_": MO_PREFIX + "MoUMDA_noDuplicates", "pop_size": 10,
                                     "select_size": 5, "prob_margin": False},
                       "gen_limit": 5})
    params = runner_algo_params(cfg)
    params["sol_length"] = 4
    seed_all(1)
    algo = instantiate(cfg.algo.init_args, **params)
    keys = [tuple(int(x) for x in ind) for ind in algo.population]
    return {"population_size": len(keys), "unique_genotypes": len(set(keys)), "evals": algo.evals}


report = {
    "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "deap": deap.__version__,
                    "hydra": hydra.__version__},
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
        assert info["is_class"] is (name in MO_ALGORITHM_CLASSES | {"OptimisationAlgorithm"}), name
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
    wrong = {path: o for path, o in outcomes.items()
             if "skipped" not in o and (o["gens"] != CONFIG_SWEEP_GENS or o["stop_trigger"] != "gen_limit")}
    assert not wrong, f"configs that did not run exactly {CONFIG_SWEEP_GENS} generations: {wrong}"


# ------------------------------------------------------------------------------ 3. characterisation


def test_baseline_spec_matches_this_module(baseline):
    """The frozen baseline was captured with exactly these cases and seeds."""
    assert baseline["seeds"] == SEEDS
    assert baseline["cases_spec"] == CHARACTERISATION_CASES


def test_stepped_loop_is_run(probe):
    """The characterisation steps the loop itself; that loop must be exactly `run()`."""
    equivalence = _section(probe, "run_equivalence")
    assert equivalence == {name: True for name in CHARACTERISATION_CASES}, equivalence


@pytest.mark.parametrize("case", sorted(CHARACTERISATION_CASES))
def test_characterisation_matches_baseline(probe, baseline, case):
    observed = _section(probe, "characterisation")[case]
    expected = baseline["cases"][case]

    assert set(observed) == set(expected) == {str(seed) for seed in SEEDS}
    for seed in sorted(expected):
        got, want = observed[seed], expected[seed]
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


# ------------------------------------------------------------------------------ 9/10. known defect

_NO_DUPLICATES_DEFECT = (
    "Known defect, fixed in MO refactor Stage 5: MoUMDA_noDuplicates never passes prevent_duplicates "
    "to the sampler and initialises through the generic initialiser, so its populations contain "
    "duplicate genotypes."
)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_NO_DUPLICATES_DEFECT)
def test_no_duplicates_generation_is_duplicate_free(probe):
    found = _section(probe, "no_duplicates_generation")

    # Preconditions of the scenario itself; they hold before and after the fix.
    if found["free_bits"] != 4 or not found["all_within_support"] or found["population_size"] != 10:
        raise RuntimeError(f"the injected scenario is not what the test assumes: {found}")

    assert found["unique_genotypes"] == found["population_size"], found


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_NO_DUPLICATES_DEFECT)
def test_no_duplicates_initial_population_is_duplicate_free(probe):
    found = _section(probe, "no_duplicates_initial_population")

    if found["population_size"] != 10 or found["evals"] != 10:
        raise RuntimeError(f"the initialisation scenario is not what the test assumes: {found}")

    assert found["unique_genotypes"] == found["population_size"], found
