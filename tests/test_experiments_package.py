"""Contracts of the `noisyvis.experiments` package and the removed compatibility modules (Stages 7, 12).

Stage 7 moved `run_helpers.py` to `noisyvis.experiments.hyperparams` and left the root
`run_helpers.py` as a forwarder for the Hydra config dotted paths `_target_: run_helpers.*`. Stage 12
rewrote those paths and deleted the forwarder, together with the `src/src/` compatibility package.
These tests pin four things:

1. `hyperparams` exposes exactly the namespace the original module had, and every configured
   hyperparams target resolves to the `hyperparams` object, whichever spelling (`run_helpers.*` or
   `noisyvis.experiments.hyperparams.*`) the config uses.
2. No Python source imports the root `run_helpers`.
3. Importing any `noisyvis.experiments` module has no side effects: it sets no MLflow tracking URI
   (a workflow must never inherit another's, R24) and creates no files. New modules added to the
   package are covered automatically.
4. The compatibility modules Stage 12 removed stay removed: none of them is importable from a
   runner-like `sys.path`, and neither `src/src/` nor the root `run_helpers.py` exists.

Imports of the package happen in subprocesses, so the pytest process never imports the science
packages.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

from harness.run_isolated import (
    HARNESS_DIR,
    MLFLOW_SENTINEL_URI,
    WORKSPACE,
    child_env,
    make_temp_root,
)

# The public namespace of the original root run_helpers.py (captured from commit 3e6e68e): its
# seven functions plus the two names its own imports bound. A star-import forwarder re-exports all
# of them, so the namespace the run scripts used to star-import is unchanged.
EXPECTED_PUBLIC_NAMES = {
    "DictConfig",
    "determine_pid_from_cfg",
    "dynamic_pop_size_1plambdaea",
    "dynamic_pop_size_PCEA",
    "dynamic_pop_size_UMDA",
    "dynamic_pop_size_mu",
    "dynamic_pop_size_mup1ea",
    "inverse_n_mut_rate",
    "np",
}


def _run_child(code: str, *, workspace_on_path: bool) -> dict:
    """Run `code` in a harness-style subprocess in a fresh temp root; it prints one JSON line."""
    root = make_temp_root()
    try:
        prelude = f"import sys\nsys.path.insert(0, {str(WORKSPACE)!r})\n" if workspace_on_path else ""
        completed = subprocess.run(
            [sys.executable, "-c", prelude + code],
            cwd=str(root),
            env=child_env(root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert completed.returncode == 0, f"child failed:\n{completed.stderr[-4000:]}"
        return json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)


# Every configured `_target_` of the hyperparams family, by canonical spelling, whichever spelling the
# config uses. It imports no compatibility module.
_TARGETS_PRELUDE = """
import importlib.util
import json
from pathlib import Path

import yaml
from hydra._internal.utils import _locate

spec = importlib.util.spec_from_file_location("_legacy_paths", %(legacy_paths)r)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

LEGACY_PREFIX = "run_helpers."
CANONICAL_PREFIX = legacy.LEGACY_TO_CANONICAL[LEGACY_PREFIX]

def public(module):
    return sorted(name for name in vars(module) if not name.startswith("_"))

def canonical_targets(directory):
    found = []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "_target_" and isinstance(value, str):
                    value = legacy.canonicalise(value)
                    if value.startswith(CANONICAL_PREFIX):
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    for path in sorted(Path(directory).rglob("*.yaml")):
        walk(yaml.safe_load(path.read_text()))
    return found

canonical_config_targets = {"configs": canonical_targets("/workspace/configs"),
                            "tests/configs": canonical_targets("/workspace/tests/configs")}
""" % {"legacy_paths": str(HARNESS_DIR.parent / "legacy_paths.py")}

# Permanent: runs WITHOUT /workspace on sys.path, so the root forwarder cannot be what resolves.
_CANONICAL_TARGET_PROBE = _TARGETS_PRELUDE + """
import noisyvis.experiments.hyperparams as hyperparams

resolution = {}
for target in sorted({t for ts in canonical_config_targets.values() for t in ts}):
    try:
        resolution[target] = _locate(target) is getattr(hyperparams, target[len(CANONICAL_PREFIX):])
    except Exception as exc:  # noqa: BLE001
        resolution[target] = "unresolvable: " + type(exc).__name__

print(json.dumps({
    "canonical_module": CANONICAL_PREFIX[:-1],
    "hyperparams_module": hyperparams.__name__,
    "hyperparams_names": public(hyperparams),
    "target_counts": {where: len(ts) for where, ts in canonical_config_targets.items()},
    "target_resolves_to_hyperparams_object": resolution,
}))
"""


def test_hyperparams_config_targets_resolve_to_canonical_objects():
    """Permanent: every configured hyperparams target resolves in noisyvis.experiments.hyperparams,
    whichever spelling (`run_helpers.*` or canonical) the config uses, without /workspace on sys.path."""
    report = _run_child(_CANONICAL_TARGET_PROBE, workspace_on_path=False)

    assert report["canonical_module"] == "noisyvis.experiments.hyperparams"
    assert report["hyperparams_module"] == report["canonical_module"]
    assert set(report["hyperparams_names"]) == EXPECTED_PUBLIC_NAMES

    assert report["target_counts"]["configs"] > 0, "no hyperparams config targets found; walk broken"
    unresolved = [t for t, same in report["target_resolves_to_hyperparams_object"].items() if same is not True]
    assert not unresolved, f"config targets not resolving to the hyperparams objects: {unresolved}"


def _imports_root_run_helpers(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name == "run_helpers" or alias.name.startswith("run_helpers.") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return node.level == 0 and (module == "run_helpers" or module.startswith("run_helpers."))
    return False


def test_no_python_source_imports_root_run_helpers():
    deps = WORKSPACE / "tests" / ".deps"
    files = [path for path in sorted(WORKSPACE.glob("run*.py")) if path.name != "run_helpers.py"]
    files += sorted((WORKSPACE / "src" / "noisyvis").rglob("*.py"))
    files += [path for path in sorted((WORKSPACE / "tests").rglob("*.py")) if deps not in path.parents]

    assert any(path.name == "run.py" for path in files)
    assert any("noisyvis" in path.parts for path in files)
    assert any(path.name == "test_experiments_package.py" for path in files)

    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if _imports_root_run_helpers(node):
                offenders.append(f"{path.relative_to(WORKSPACE)}:{node.lineno}")
    assert not offenders, (
        "Python source imports the root run_helpers forwarder, which exists only for Hydra config "
        f"dotted paths until Stage 12; import noisyvis.experiments.hyperparams instead: {offenders}"
    )


# The compatibility modules Stage 12 removed: the `src/src/` forwarders and the root forwarder.
REMOVED_COMPAT_MODULES = (
    "src.algorithms.Algorithms",
    "src.algorithms.MOAlgorithms",
    "src.problems.ProblemScripts",
    "src.problems.ViolationFunctions",
    "run_helpers",
)

_REMOVED_MODULES_PROBE = """
import importlib
import json
import sys

import noisyvis

outcomes = {}
for name in %(modules)r:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        outcomes[name] = "ModuleNotFoundError"
    except Exception as exc:  # noqa: BLE001
        outcomes[name] = f"{type(exc).__name__}: {exc}"
    else:
        outcomes[name] = f"imported from {getattr(module, '__file__', None)}"

try:
    import src
except ModuleNotFoundError:
    bare_src = {"imported": False}
else:
    bare_src = {"imported": True, "file": getattr(src, "__file__", None),
                "path": [str(entry) for entry in getattr(src, "__path__", [])]}

print(json.dumps({"sys_path_0": sys.path[0], "noisyvis_file": noisyvis.__file__,
                  "outcomes": outcomes, "bare_src": bare_src}))
""" % {"modules": REMOVED_COMPAT_MODULES}


def test_compatibility_shims_removed():
    """Stage 12 deleted the compatibility bridge, and nothing may make it importable again.

    The child is runner-like: `/workspace` first on sys.path (the run scripts' own directory) plus
    the editable install's `/workspace/src`. A stray `__pycache__/run_helpers.*.pyc` cannot bring the
    root module back, because Python never imports a cached bytecode file without its source.
    A bare `import src` may still succeed, but only as a namespace package that forwards to nothing.
    """
    assert not (WORKSPACE / "src" / "src").exists(), "the src/src compatibility package still exists"
    assert not (WORKSPACE / "run_helpers.py").exists(), "the root run_helpers.py forwarder still exists"

    report = _run_child(_REMOVED_MODULES_PROBE, workspace_on_path=True)

    assert report["sys_path_0"] == str(WORKSPACE), report["sys_path_0"]
    assert report["noisyvis_file"] == str(WORKSPACE / "src" / "noisyvis" / "__init__.py"), report["noisyvis_file"]
    importable = {name: outcome for name, outcome in report["outcomes"].items() if outcome != "ModuleNotFoundError"}
    assert set(report["outcomes"]) == set(REMOVED_COMPAT_MODULES)
    assert not importable, f"removed compatibility modules are importable again: {importable}"
    bare_src = report["bare_src"]
    assert not bare_src["imported"] or bare_src["file"] is None, (
        f"`src` is a regular package again, which would be a new compatibility bridge: {bare_src}"
    )


_IMPORT_SIDE_EFFECT_PROBE = """
import importlib
import importlib.util
import json
import os
import pkgutil

spec = importlib.util.spec_from_file_location("_harness_fence", %(fence)r)
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)
fence.install(%(workspace)r)

def listing():
    out = []
    for dirpath, dirnames, filenames in os.walk(".", followlinks=False):
        out.extend(os.path.join(dirpath, name) for name in dirnames + filenames)
    return sorted(out)

before = listing()
package = importlib.import_module("noisyvis.experiments")
modules = [package.__name__]
for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
    importlib.import_module(info.name)
    modules.append(info.name)
after = listing()

import mlflow
print(json.dumps({
    "modules": modules,
    "tracking_uri": mlflow.get_tracking_uri(),
    "files_unchanged": before == after,
    "new_paths": sorted(set(after) - set(before)),
    "workspace_on_path": any(os.path.realpath(p or ".") == %(workspace)r for p in __import__("sys").path),
}))
""" % {"fence": str(HARNESS_DIR / "fence.py"), "workspace": str(WORKSPACE)}


def test_experiments_modules_import_without_side_effects():
    report = _run_child(_IMPORT_SIDE_EFFECT_PROBE, workspace_on_path=False)

    assert not report["workspace_on_path"], "the probe must not see the root wrappers or forwarder"
    assert "noisyvis.experiments.hyperparams" in report["modules"], report["modules"]
    assert report["tracking_uri"] == MLFLOW_SENTINEL_URI, (
        f"importing {report['modules']} changed the MLflow tracking URI to {report['tracking_uri']!r}"
    )
    assert report["files_unchanged"], f"importing experiments modules created {report['new_paths']}"
