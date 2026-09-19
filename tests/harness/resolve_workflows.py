"""Run the synthetic config-resolution cases through a set of resolver functions (plan §5.4, Stage 7).

Child process of `test_config_workflows.py`. Like `resolve_in_runner.py`, it runs with its cwd in a
harness temp root (knapsack loaders resolve instances through NOISYVIS_ROOT), puts /workspace first
on sys.path exactly as `python run.py` has it, and installs the write fence. After resolving, it
reports which compatibility modules got loaded; since Stage 12 the config targets are canonical
`noisyvis.*` paths and none may be.

A resolver is named by a spec:
    run.py:resolve_config_dependencies          a function in a run script, loaded with a
                                                non-"__main__" run name (imports run, main does not)
    package.module:function                     a function in an importable module

Each outcome records the resolved config in a type-preserving and key-order-preserving encoding
(dict insertion order is what shows where new keys such as `ref_point`, `PID` or a created
`mutate_params` were added), or the exception that resolution raised. It also records whether the
input config was left unmodified.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path("/workspace")
HARNESS_DIR = Path(__file__).resolve().parent
TESTS_DIR = HARNESS_DIR.parent
NON_MAIN_RUN_NAME = "noisyvis_stage7_workflow_check"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(obj):
    """JSON form that keeps dict key order, non-string keys, tuples, and int versus float."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        return {"__dict__": [[encode(k), encode(v)] for k, v in obj.items()]}
    if isinstance(obj, list):
        return [encode(item) for item in obj]
    if isinstance(obj, tuple):
        return {"__tuple__": [encode(item) for item in obj]}
    raise TypeError(f"cannot encode {type(obj)!r}: {obj!r}")


class ResolverLoader:
    def __init__(self):
        self._script_namespaces: dict = {}

    def get(self, spec: str):
        target, function = spec.rsplit(":", 1)
        if target.endswith(".py"):
            path = WORKSPACE / target
            if path not in self._script_namespaces:
                import runpy

                self._script_namespaces[path] = runpy.run_path(str(path), run_name=NON_MAIN_RUN_NAME)
            return self._script_namespaces[path][function]
        return getattr(importlib.import_module(target), function)


def resolve_case(resolver, case_cfg: dict) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(copy.deepcopy(case_cfg))
    before = OmegaConf.to_container(cfg)
    try:
        resolved = resolver(cfg)
    except Exception as exc:  # noqa: BLE001 - the raised type is the recorded behaviour
        outcome = {"raises": f"{type(exc).__module__}.{type(exc).__qualname__}"}
    else:
        outcome = {"resolved": encode(OmegaConf.to_container(resolved))}
    outcome["input_unchanged"] = OmegaConf.to_container(cfg) == before
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--resolvers", required=True, help="JSON: {workflow: [spec, ...]}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.chdir(Path(args.root).resolve())

    fence = _load_by_path("_harness_fence", HARNESS_DIR / "fence.py")
    cases_module = _load_by_path("_config_workflow_cases", TESTS_DIR / "config_workflow_cases.py")
    sys.path[0] = str(WORKSPACE)
    fence.install(str(WORKSPACE))

    resolvers = json.loads(Path(args.resolvers).read_text())
    cases = cases_module.build_cases()
    loader = ResolverLoader()

    results: dict = {}
    for workflow, specs in resolvers.items():
        results[workflow] = {}
        for spec in specs:
            resolver = loader.get(spec)
            results[workflow][spec] = {
                name: resolve_case(resolver, case_cfg) for name, case_cfg in cases[workflow].items()
            }

    # Stage 12: after every case has resolved, which compatibility modules got loaded (none may).
    compat = _load_by_path("_harness_compat_modules", HARNESS_DIR / "compat_modules.py")
    results["compat_modules_loaded"] = compat.compat_modules_loaded()

    Path(args.out).write_text(json.dumps(results, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
