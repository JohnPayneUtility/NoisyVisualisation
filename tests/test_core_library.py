"""Core-library contracts, characterised before the Stage 8 moves (plan Stage 8, Checkpoint 0).

Stage 8 moves the logger to `noisyvis.tracking`, the LON/CoLON builders to `noisyvis.networks`, renames
the two algorithm modules, repoints the `src.algorithms.*` forwarders and dedupes the AST-identical D4
operators into `noisyvis.algorithms.operators`. The reproducibility baselines catch most regressions
that could cause, but not these:

1. A split active-logger singleton. If `set` and `get` land on different module objects the SO
   baselines fail loudly, but a `clear_active_logger` on a stale copy is silent.
2. The configured `attr_function` names must keep resolving through the `noisyvis.algorithms` namespace
   to the same definitions.
3. Every D4 definition a consumer reaches today (package namespace, SO module, MO module, the
   fitness evaluators that call `random_bit_flip`, the LON and CoLON builder modules) must keep the same
   definition, whichever copy survives the dedupe.
4. Every configured algorithm target must resolve to its canonical module's object, whichever spelling
   (`src.algorithms.*` or `noisyvis.algorithms.*`) the config uses, and the canonical modules must keep
   the frozen namespace. Until Stage 12 Checkpoint C deletes them, a separate temporary test and probe
   also pin that the two `src.algorithms.*` forwarders re-export exactly those canonical objects.
5. The LON, CoLON and compression entry points must move without any edit to their bodies.

Every test is location-agnostic: modules are found through the objects runtime code actually uses,
never by hard-coding pre- or post-move paths, so the same tests hold before and after each Stage 8
checkpoint. Stage 9 Checkpoint 0 removed the one remaining hard-coded path, the
`noisyvis.problems.FitnessFunctions` import: the fitness consumers are now the `__globals__` of the two
evaluators that call `random_bit_flip` and of `OneMax_fitness`, reached through the `noisyvis.problems`
namespace, so the tests also hold across the Stage 9 split of that module.

The EXPECTED values are frozen literals captured from the untouched pre-Stage-8 tree (commit
PRE_STAGE8_COMMIT), under the runner's Python 3.11. They were computed from a `git archive` of that
commit, not from the implementation under test, and must never be re-derived to make a test pass.
AST hashes are `sha256(ast.dump(<FunctionDef|ClassDef>))`, so they ignore comments, formatting and
line numbers but pin every statement, in order. `ast.dump` output depends on the Python version, so a
runner Python upgrade legitimately needs the hashes re-captured from PRE_STAGE8_COMMIT.

Imports of the science packages happen in one subprocess, so the pytest process never imports them.
The subprocess only reports what it observes; every expectation lives in this file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from harness.run_isolated import HARNESS_DIR, WORKSPACE, child_env, make_temp_root
from legacy_paths import LEGACY_TO_CANONICAL, canonicalise

PRE_STAGE8_COMMIT = "465ca0689be53f1a38aa24df977fedd7303857b3"

# --------------------------------------------------------------- frozen at PRE_STAGE8_COMMIT

LOGGER_API_AST = {
    "set_active_logger": "19a2e25853e73de6bfe113a99495dcf23fe1d8fe9f0805ddb67ac7eb25fb9a0b",
    "get_active_logger": "0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f",
    "clear_active_logger": "a876c96f177e76b775a7585d4f9ec2db44f278cf7d16d9886b5ec44833015002",
    "ExperimentLogger": "46f7737302ba750d58e01fb8c699bbf7ae9656178b377bfeece7805c6c90ac30",
}

# One hash per name: the copies in Algorithms.py / MOAlgorithms.py / LONs.py were verified identical.
D4_AST = {
    "binary_attribute": "dcc0ff4798ab7d1735c9ff0a1d7aa4a0e0e1dbf406e0245ed2723bb01297ea08",
    "Rastrigin_attribute": "936cbdf479b46e98f031c395de3c9717ad38b14c87c1b7347fc1d51258d43c09",
    "mutSwapBit": "6e0c71bcda31c6595928be489eaec6f6ad4aa7b27cc9ffe1d76fd2fcffd71c53",
    "random_bit_flip": "9b0d90c2dce16fd6c4d322178bc6e179dea2727f737c06ecfee3bad240bd203a",
    "complementary_crossover": "4d539f133049e136f82376eb3ba0281b35e06972c0cbf2d1eb3ad8db1d155fa5",
}

LON_ENTRY_POINT_AST = {
    "BinaryLON": "2cda6b976a0712bf6dc56b2afad2e1be3a9babf38c8a27de3c1651c1c3b141f7",
    "BinaryCoLON": "6f6c6179dd0f7fafa649ed4297140057727657e3e6dc2a42eb5670c85f0ca87c",
    "compress_lon_aggregated": "dfa5c67219564d38012b8857416c74ada57d3df53c85d94b97006c3f0d826b1f",
}

CONFIGURED_ATTR_FUNCTIONS = frozenset({"binary_attribute", "Rastrigin_attribute"})

# Which D4 names each consumer reaches today. The MO module never had random_bit_flip; the fitness
# evaluators and the LON/CoLON builder modules use only random_bit_flip. The two evaluators that call it
# are consumers through their own __globals__, wherever Stage 9 places them.
D4_CONSUMERS = {
    "package": frozenset(D4_AST),
    "single_objective_module": frozenset(D4_AST),
    "multi_objective_module": frozenset(D4_AST) - {"random_bit_flip"},
    "OneMax_prior_mult_bitflip_fitness_globals": frozenset({"random_bit_flip"}),
    "eval_noisy_kp_prior_mult_bitflip_globals": frozenset({"random_bit_flip"}),
    "BinaryLON_module": frozenset({"random_bit_flip"}),
    "BinaryCoLON_module": frozenset({"random_bit_flip"}),
}

# Public namespace of each forwarder: every module-level binding of the forwarded module.
FORWARDER_PUBLIC_NAMES = {
    "Algorithms": frozenset({
        "ABC", "Any", "Callable", "CompactGA", "ExperimentLogger", "List", "MuPlusLamdaEA",
        "MuPlusLamdaEA_estimated", "MuPlusLamdaEA_forgetful", "OptimisationAlgorithm", "Optional",
        "PCEA", "Rastrigin_attribute", "Tuple", "UMDA", "UMDA_estimated", "abstractmethod",
        "algorithms", "base", "binary_attribute", "clear_active_logger", "complementary_crossover",
        "creator", "dataclass", "field", "median", "mutSwapBit", "np", "optuna", "random",
        "random_bit_flip", "set_active_logger", "tools", "umda_update_full",
    }),
    "MOAlgorithms": frozenset({
        "ABC", "Any", "Callable", "List", "MoUMDA", "MoUMDA_ParetoArchive", "MoUMDA_noDuplicates",
        "NSGA2", "OptimisationAlgorithm", "Optional", "Rastrigin_attribute", "SEMO", "Tuple",
        "abstractmethod", "algorithms", "base", "binary_attribute", "complementary_crossover",
        "creator", "dataclass", "field", "front_sig", "hypervolume", "mo_umda_update_full",
        "mo_umda_update_with_archive", "mutSwapBit", "mut_flip_one_bit", "np", "optuna", "pydoc",
        "random", "record_pareto_data", "tools",
    }),
}

# A class defined only in each forwarded module, used to find the canonical module at runtime.
FORWARDER_ANCHOR = {"Algorithms": "MuPlusLamdaEA", "MOAlgorithms": "SEMO"}

# B1 (known_broken_configs.py) is the only config target expected not to resolve.
FORWARDER_UNRESOLVABLE_TARGETS = {
    "Algorithms": frozenset(),
    "MOAlgorithms": frozenset({"src.algorithms.MOAlgorithms.MoMuPlusLamdaEA"}),
}

# ------------------------------------------------------------------------------ the probe

# Shared by the main probe and the temporary forwarder probe. It imports no compatibility module.
_PROBE_PRELUDE = """
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path

spec = importlib.util.spec_from_file_location("_harness_fence", %(fence)r)
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)
fence.install(%(workspace)r)

spec = importlib.util.spec_from_file_location("_legacy_paths", %(legacy_paths)r)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

import yaml
from hydra._internal.utils import _locate

WORKSPACE = Path(%(workspace)r)
FORWARDER_ANCHOR = %(anchors)r
D4_NAMES = %(d4_names)r


def ast_sha256(obj):
    body = ast.parse(textwrap.dedent(inspect.getsource(obj))).body
    if len(body) != 1:
        raise AssertionError(f"getsource({obj!r}) parsed to {len(body)} statements")
    return hashlib.sha256(ast.dump(body[0]).encode()).hexdigest()


def describe(obj):
    if obj is None:
        return None
    return {"module": getattr(obj, "__module__", None), "name": getattr(obj, "__name__", None),
            "callable": callable(obj), "ast": ast_sha256(obj) if callable(obj) else None}


def optional_import(name):
    try:
        found = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    return importlib.import_module(name) if found is not None else None


def config_values(key):
    def walk(node, out):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key and isinstance(v, str) and v:
                    out.append(v)
                else:
                    walk(v, out)
        elif isinstance(node, list):
            for item in node:
                walk(item, out)
    found = {}
    for where in ("configs", "tests/configs"):
        values = []
        for path in sorted((WORKSPACE / where).rglob("*.yaml")):
            walk(yaml.safe_load(path.read_text()), values)
        found[where] = values
    return found
"""

# The main probe never imports a compatibility module, so it keeps working once Stage 12 deletes them.
_PROBE_MAIN = """
# Every current consumer, imported the way runtime code imports it.
import noisyvis.problems
import noisyvis.experiments.runner as runner
import noisyvis.experiments.lon_runner as lon_runner
import noisyvis.algorithms

# Fitness consumers through the dynamic namespace the runners use, never through a module path.
problems = sys.modules["noisyvis.problems"]
fitness_globals = {name: getattr(problems, name).__globals__
                   for name in ("OneMax_fitness", "OneMax_prior_mult_bitflip_fitness",
                                "eval_noisy_kp_prior_mult_bitflip")}
get_active_logger_via_fitness = fitness_globals["OneMax_fitness"]["get_active_logger"]

package = sys.modules["noisyvis.algorithms"]
so_module = sys.modules[package.MuPlusLamdaEA.__module__]
mo_module = sys.modules[package.SEMO.__module__]
lon_module = sys.modules[lon_runner.BinaryLON.__module__]
colon_module = sys.modules[lon_runner.BinaryCoLON.__module__]

# Old and new logger paths: import whichever exists, so a stray copy would be loaded and counted.
logger_paths = {name: optional_import(name) is not None
                for name in ("noisyvis.algorithms.Logger", "noisyvis.tracking.logger")}

report = {}

# 1. active-logger singleton
holders = sorted(
    name for name, module in list(sys.modules.items())
    if module is not None and name.split(".")[0] in ("noisyvis", "src") and "_active_logger" in vars(module)
)
logger = {"paths_present": logger_paths, "holders": holders}
if len(holders) == 1:
    holder = sys.modules[holders[0]]
    users = {
        "set_active_logger via single-objective module": so_module.set_active_logger,
        "get_active_logger via OneMax_fitness globals": get_active_logger_via_fitness,
        "clear_active_logger via experiments.runner": runner.clear_active_logger,
    }
    logger["shares_holder_globals"] = {label: fn.__globals__ is vars(holder) for label, fn in users.items()}
    state = {"initially_none": get_active_logger_via_fitness() is None}
    sentinel = object()
    so_module.set_active_logger(sentinel)
    state["get_sees_set"] = get_active_logger_via_fitness() is sentinel
    state["holder_state_set"] = vars(holder)["_active_logger"] is sentinel
    runner.clear_active_logger()
    state["get_sees_clear"] = get_active_logger_via_fitness() is None
    state["holder_state_cleared"] = vars(holder)["_active_logger"] is None
    logger["state"] = state
    logger["api_ast"] = {name: ast_sha256(getattr(holder, name))
                         for name in ("set_active_logger", "get_active_logger", "clear_active_logger", "ExperimentLogger")}
    logger["single_objective_uses_holder_class"] = so_module.ExperimentLogger is holder.ExperimentLogger
report["logger"] = logger

# 2. configured attr_function names through the dynamic namespace
attr_values = config_values("attr_function")
report["attr"] = {
    "counts": {where: len(values) for where, values in attr_values.items()},
    "names": {name: describe(getattr(sys.modules["noisyvis.algorithms"], name, None))
              for name in sorted({v for values in attr_values.values() for v in values})},
}

# 3. D4 definitions reached by every consumer
consumers = {
    "package": vars(package),
    "single_objective_module": vars(so_module),
    "multi_objective_module": vars(mo_module),
    "OneMax_prior_mult_bitflip_fitness_globals": fitness_globals["OneMax_prior_mult_bitflip_fitness"],
    "eval_noisy_kp_prior_mult_bitflip_globals": fitness_globals["eval_noisy_kp_prior_mult_bitflip"],
    "BinaryLON_module": vars(lon_module),
    "BinaryCoLON_module": vars(colon_module),
}
report["d4"] = {
    role: {"module": namespace["__name__"],
           "names": {name: describe(namespace[name]) for name in D4_NAMES if name in namespace}}
    for role, namespace in consumers.items()
}

# 4/5. configured algorithm targets, through their canonical modules. Path-neutral: a target is
# counted and resolved by its canonical spelling, whichever spelling the config uses.
targets = config_values("_target_")
canonical_report = {}
for name, anchor in FORWARDER_ANCHOR.items():
    prefix = legacy.LEGACY_TO_CANONICAL["src.algorithms." + name + "."]
    canonical = importlib.import_module(prefix[:-1])
    mine = {where: [t for t in map(legacy.canonicalise, values) if t.startswith(prefix)]
            for where, values in targets.items()}
    resolution = {}
    for target in sorted({t for values in mine.values() for t in values}):
        attribute = target.rsplit(".", 1)[1]
        try:
            obj = _locate(target)
        except Exception as exc:
            resolution[target] = "unresolvable: " + type(exc).__name__
            continue
        resolution[target] = "canonical" if obj is getattr(canonical, attribute, object()) else "different object"
    canonical_report[name] = {
        "canonical_module": canonical.__name__,
        "anchor_defined_in_canonical": getattr(canonical, anchor).__module__ == canonical.__name__,
        "canonical_public": sorted(n for n in vars(canonical) if not n.startswith("_")),
        "package_exports_canonical_anchor": getattr(package, anchor) is getattr(canonical, anchor),
        "target_counts": {where: len(values) for where, values in mine.items()},
        "resolution": resolution,
    }
report["canonical_targets"] = canonical_report

# 6. LON / CoLON / compression entry points, as the LON runner imports them
report["lon"] = {name: describe(getattr(lon_runner, name, None))
                 for name in ("BinaryLON", "BinaryCoLON", "compress_lon_aggregated")}

print(json.dumps(report))
"""

# TEMPORARY (Stage 12): the `src.algorithms.*` forwarders, imported only here. Deleted with the
# forwarders in Stage 12 Checkpoint C.
_FORWARDER_PROBE_BODY = """
import noisyvis.algorithms

package = sys.modules["noisyvis.algorithms"]
forwarders = {name: importlib.import_module("src.algorithms." + name) for name in FORWARDER_ANCHOR}

targets = config_values("_target_")
report = {}
for name, forwarder in forwarders.items():
    anchor = FORWARDER_ANCHOR[name]
    canonical = sys.modules[getattr(forwarder, anchor).__module__]
    public = sorted(n for n in vars(forwarder) if not n.startswith("_"))
    prefix = "src.algorithms." + name + "."
    canonical_prefix = legacy.LEGACY_TO_CANONICAL[prefix]
    # Every configured target of this family, in its legacy spelling, whichever spelling the config uses.
    mine = {where: [prefix + t[len(canonical_prefix):] for t in map(legacy.canonicalise, values)
                    if t.startswith(canonical_prefix)]
            for where, values in targets.items()}
    resolution = {}
    for target in sorted({t for values in mine.values() for t in values}):
        attribute = target.rsplit(".", 1)[1]
        try:
            obj = _locate(target)
        except Exception as exc:
            resolution[target] = "unresolvable: " + type(exc).__name__
            continue
        resolution[target] = ("canonical" if obj is getattr(canonical, attribute, object())
                              and obj is getattr(forwarder, attribute, object()) else "different object")
    report[name] = {
        "file": forwarder.__file__,
        "canonical_module": canonical.__name__,
        "public": public,
        "canonical_public": sorted(n for n in vars(canonical) if not n.startswith("_")),
        "not_identical": [n for n in public if getattr(forwarder, n) is not getattr(canonical, n, object())],
        "package_exports_canonical_anchor": getattr(package, anchor) is getattr(canonical, anchor),
        "target_counts": {where: len(values) for where, values in mine.items()},
        "resolution": resolution,
    }

print(json.dumps(report))
"""

_SUBSTITUTIONS = {
    "fence": str(HARNESS_DIR / "fence.py"),
    "legacy_paths": str(HARNESS_DIR.parent / "legacy_paths.py"),
    "workspace": str(WORKSPACE),
    "anchors": FORWARDER_ANCHOR,
    "d4_names": sorted(D4_AST),
}
_PROBE = (_PROBE_PRELUDE + _PROBE_MAIN) % _SUBSTITUTIONS
_FORWARDER_PROBE = (_PROBE_PRELUDE + _FORWARDER_PROBE_BODY) % _SUBSTITUTIONS


def _run_probe(code: str, label: str) -> dict:
    """Run one probe in a harness subprocess (fresh temp root, fence installed)."""
    root = make_temp_root()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(root),
            env=child_env(root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, f"{label} failed:\n{completed.stderr[-4000:]}"
        return json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def probe() -> dict:
    """Run the main probe once; it imports no compatibility module."""
    return _run_probe(_PROBE, "core-library probe")


@pytest.fixture(scope="module")
def forwarder_probe() -> dict:
    """TEMPORARY (Stage 12): the forwarder-only probe, deleted with the forwarders in Checkpoint C."""
    return _run_probe(_FORWARDER_PROBE, "core-library forwarder probe")


# ------------------------------------------------------------------------------ the tests


def test_active_logger_singleton_is_shared(probe):
    logger = probe["logger"]

    assert any(logger["paths_present"].values()), f"no logger module found: {logger['paths_present']}"
    assert len(logger["holders"]) == 1, (
        f"the active-logger state `_active_logger` lives in {logger['holders']}; exactly one module "
        "object may hold it, or set/get/clear can act on different singletons (R3)"
    )
    not_shared = [label for label, shared in logger["shares_holder_globals"].items() if not shared]
    assert not not_shared, f"these logger functions do not use the single holder's state: {not_shared}"
    assert logger["single_objective_uses_holder_class"], "the SO module's ExperimentLogger is not the holder's"

    assert logger["state"] == {
        "initially_none": True,
        "get_sees_set": True,
        "holder_state_set": True,
        "get_sees_clear": True,
        "holder_state_cleared": True,
    }, f"set/get/clear do not behave as one singleton: {logger['state']}"

    assert logger["api_ast"] == LOGGER_API_AST, "logger API definitions changed since the pre-Stage-8 tree"


def test_configured_attr_functions_resolve_in_algorithms_namespace(probe):
    attr = probe["attr"]

    assert attr["counts"]["configs"] > 0, "no attr_function found under configs/; the walk looks broken"
    assert set(attr["names"]) == CONFIGURED_ATTR_FUNCTIONS, (
        f"configured attr_function names changed: {sorted(attr['names'])}"
    )
    for name, found in attr["names"].items():
        assert found is not None, f"getattr(sys.modules['noisyvis.algorithms'], {name!r}) fails"
        assert found["callable"] and found["name"] == name, f"{name}: {found}"
        assert found["ast"] == D4_AST[name], f"{name} resolves to a changed definition ({found['module']})"


def test_d4_definitions_reached_by_every_consumer_are_pinned(probe):
    d4 = probe["d4"]

    assert set(d4) == set(D4_CONSUMERS)
    for role, expected_names in D4_CONSUMERS.items():
        reached = d4[role]["names"]
        missing = expected_names - set(reached)
        assert not missing, f"{role} ({d4[role]['module']}) no longer reaches {sorted(missing)}"
        for name in expected_names:
            found = reached[name]
            assert found["callable"] and found["name"] == name, f"{role}.{name}: {found}"
            assert found["ast"] == D4_AST[name], (
                f"{role} ({d4[role]['module']}) reaches a changed {name} defined in {found['module']}"
            )


@pytest.mark.parametrize("forwarder", sorted(FORWARDER_PUBLIC_NAMES))
def test_algorithm_config_targets_resolve_to_canonical_objects(probe, forwarder):
    """Permanent: every configured algorithm target resolves in its canonical module, whichever spelling."""
    report = probe["canonical_targets"][forwarder]
    expected_module = LEGACY_TO_CANONICAL[f"src.algorithms.{forwarder}."][:-1]

    assert report["canonical_module"] == expected_module, report["canonical_module"]
    assert report["canonical_module"].startswith("noisyvis.algorithms."), report["canonical_module"]
    assert report["anchor_defined_in_canonical"], (
        f"{FORWARDER_ANCHOR[forwarder]} is not defined in {report['canonical_module']}"
    )

    expected = FORWARDER_PUBLIC_NAMES[forwarder]
    assert set(report["canonical_public"]) == expected, (
        f"{report['canonical_module']} public namespace differs from the frozen forwarder namespace:\n"
        f"  missing: {sorted(expected - set(report['canonical_public']))}\n"
        f"  added:   {sorted(set(report['canonical_public']) - expected)}"
    )
    assert report["package_exports_canonical_anchor"], (
        f"noisyvis.algorithms.{FORWARDER_ANCHOR[forwarder]} is not the canonical module's class"
    )

    assert report["target_counts"]["configs"] > 0, "no config targets found; the walk looks broken"
    assert report["target_counts"]["tests/configs"] > 0, "no test-config targets found; the walk looks broken"
    unresolvable = {t for t, outcome in report["resolution"].items() if outcome.startswith("unresolvable")}
    different = {t for t, outcome in report["resolution"].items() if outcome == "different object"}
    assert not different, f"config targets resolve to non-canonical objects: {sorted(different)}"
    # The frozen literal keeps its historical spelling; B1 stays unresolvable under the canonical one.
    expected_unresolvable = {canonicalise(t) for t in FORWARDER_UNRESOLVABLE_TARGETS[forwarder]}
    assert unresolvable == expected_unresolvable, (
        f"unexpected set of unresolvable config targets: {sorted(unresolvable)}"
    )


# TEMPORARY (Stage 12): deleted with the forwarders in Checkpoint C.
@pytest.mark.parametrize("forwarder", sorted(FORWARDER_PUBLIC_NAMES))
def test_algorithm_forwarder_preserves_namespace_and_identity(forwarder_probe, forwarder):
    report = forwarder_probe[forwarder]

    assert report["file"] == str(WORKSPACE / "src" / "src" / "algorithms" / f"{forwarder}.py")
    assert report["canonical_module"].startswith("noisyvis.algorithms."), report["canonical_module"]
    assert report["canonical_module"] != f"src.algorithms.{forwarder}"

    expected = FORWARDER_PUBLIC_NAMES[forwarder]
    assert set(report["public"]) == expected, (
        f"src.algorithms.{forwarder} namespace changed:\n"
        f"  missing: {sorted(expected - set(report['public']))}\n"
        f"  added:   {sorted(set(report['public']) - expected)}"
    )
    assert set(report["canonical_public"]) == expected, (
        f"{report['canonical_module']} public namespace differs from the frozen forwarder namespace"
    )
    assert not report["not_identical"], (
        f"src.algorithms.{forwarder} re-exports different objects for: {report['not_identical']}"
    )
    assert report["package_exports_canonical_anchor"], (
        f"noisyvis.algorithms.{FORWARDER_ANCHOR[forwarder]} is not the canonical module's class"
    )

    assert report["target_counts"]["configs"] > 0, "no config targets found; the walk looks broken"
    assert report["target_counts"]["tests/configs"] > 0, "no test-config targets found; the walk looks broken"
    unresolvable = {t for t, outcome in report["resolution"].items() if outcome.startswith("unresolvable")}
    different = {t for t, outcome in report["resolution"].items() if outcome == "different object"}
    assert not different, f"config targets resolve to non-canonical objects: {sorted(different)}"
    assert unresolvable == FORWARDER_UNRESOLVABLE_TARGETS[forwarder], (
        f"unexpected set of unresolvable config targets: {sorted(unresolvable)}"
    )


def test_lon_entry_points_are_pinned(probe):
    lon = probe["lon"]

    for name, expected in LON_ENTRY_POINT_AST.items():
        found = lon[name]
        assert found is not None, f"noisyvis.experiments.lon_runner no longer imports {name}"
        assert found["callable"] and found["name"] == name, f"{name}: {found}"
        assert found["ast"] == expected, f"{name} ({found['module']}) changed since the pre-Stage-8 tree"
