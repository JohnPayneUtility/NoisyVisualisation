"""Contracts of the `noisyvis.experiments` package and the root `run_helpers` forwarder (Stage 7).

Stage 7 moved `run_helpers.py` to `noisyvis.experiments.hyperparams`. The root `run_helpers.py`
stays until Stage 12 as a forwarder, only so the Hydra config dotted paths `_target_: run_helpers.*`
keep resolving. These tests pin three things:

1. The forwarder exposes exactly the namespace the original module had, as the *same objects*, and
   every `run_helpers.*` config target resolves to the `hyperparams` object.
2. No Python source imports the forwarder, which is the precondition for deleting it in Stage 12.
3. Importing any `noisyvis.experiments` module has no side effects: it sets no MLflow tracking URI
   (a workflow must never inherit another's, R24) and creates no files. New modules added to the
   package are covered automatically.

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


_FORWARDER_PROBE = """
import json
from pathlib import Path

import yaml
from hydra._internal.utils import _locate

import importlib

import noisyvis.experiments.hyperparams as hyperparams

# By name, so the Stage 12 source grep for forwarder imports stays empty.
run_helpers = importlib.import_module("run_helpers")

def public(module):
    return sorted(name for name in vars(module) if not name.startswith("_"))

def targets(directory):
    found = []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "_target_" and isinstance(value, str) and value.startswith("run_helpers."):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    for path in sorted(Path(directory).rglob("*.yaml")):
        walk(yaml.safe_load(path.read_text()))
    return found

names = public(run_helpers)
config_targets = {"configs": targets("/workspace/configs"), "tests/configs": targets("/workspace/tests/configs")}
resolution = {}
for target in sorted({t for ts in config_targets.values() for t in ts}):
    resolution[target] = _locate(target) is getattr(hyperparams, target.split(".", 1)[1])

print(json.dumps({
    "forwarder_file": run_helpers.__file__,
    "forwarder_names": names,
    "hyperparams_names": public(hyperparams),
    "identical": {name: getattr(run_helpers, name) is getattr(hyperparams, name) for name in names},
    "target_counts": {where: len(ts) for where, ts in config_targets.items()},
    "target_resolves_to_hyperparams_object": resolution,
}))
"""


def test_run_helpers_forwarder_preserves_namespace_and_identity():
    report = _run_child(_FORWARDER_PROBE, workspace_on_path=True)

    assert report["forwarder_file"] == str(WORKSPACE / "run_helpers.py")
    assert set(report["hyperparams_names"]) == EXPECTED_PUBLIC_NAMES
    assert set(report["forwarder_names"]) == EXPECTED_PUBLIC_NAMES
    not_identical = [name for name, same in report["identical"].items() if not same]
    assert not not_identical, f"run_helpers re-exports different objects for: {not_identical}"

    assert report["target_counts"]["configs"] > 0, "no run_helpers.* config targets found; walk broken"
    unresolved = [t for t, same in report["target_resolves_to_hyperparams_object"].items() if not same]
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
