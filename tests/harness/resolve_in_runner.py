"""Resolve config dotted paths and dynamic names inside a runner's own namespace (plan §5.6).

The dynamic lookups in the run scripts are `getattr(sys.modules['src.problems'], name)` and
`getattr(sys.modules['src.algorithms'], name)`. Those keys exist only because each script opens with
`from src.problems import *` and `from src.algorithms import *`. Importing either namespace
explicitly from the test would *register the key* and so hide exactly the failure this gate exists
to catch.

So this child process loads the runner itself, with a `run_name` other than "__main__" (its
top-level imports run, `main` does not), reads the `sys.modules[...]` keys out of that runner's own
source by AST, and resolves every name through them.

Dotted paths use the same mechanisms the runners use:
    _target_      -> hydra._internal.utils._locate, what instantiate()/call() use
    violation_fn  -> the runner's own _import_from_dotted (importlib + getattr)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path("/workspace")
HARNESS_DIR = Path(__file__).resolve().parent
NON_MAIN_RUN_NAME = "noisyvis_stage1_cfgcheck"


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"_harness_{name}", HARNESS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace_keys(script_path: Path) -> dict:
    """Read the sys.modules keys a runner actually uses, from its own source.

    Looks for `getattr(sys.modules['<key>'], <expr>)` and classifies by the expression:
    a fitness-function lookup or an attribute-generator lookup.
    """
    tree = ast.parse(script_path.read_text(), filename=str(script_path))
    keys: dict = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue

        target = node.args[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "modules"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "sys"
        ):
            continue

        key_node = target.slice
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue

        wanted = ast.unparse(node.args[1]).lower()
        if "fitness" in wanted:
            keys["fitness"] = key_node.value
        elif "attr" in wanted:
            keys["attr"] = key_node.value

    return keys


def _resolve_one(request: dict, namespace: dict, keys: dict) -> dict:
    kind = request["kind"]
    value = request["value"]
    result = dict(request)

    try:
        if kind == "target":
            from hydra._internal.utils import _locate

            _locate(value)
        elif kind == "violation":
            importer = namespace.get("_import_from_dotted")
            if importer is None:
                raise RuntimeError(
                    "this runner has no _import_from_dotted; violation_fn was dispatched to the "
                    "wrong runner"
                )
            importer(value)
        elif kind in ("fitness", "attr"):
            key = keys.get(kind)
            if key is None:
                raise RuntimeError(f"could not find the sys.modules key for {kind!r} in the runner")
            module = sys.modules[key]  # deliberately a dict lookup, never an import
            getattr(module, value)
        else:
            raise RuntimeError(f"unknown request kind {kind!r}")
    except Exception as exc:  # noqa: BLE001 - the error text is the test's payload
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["ok"] = True
    result["error"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    script_path = Path(args.script).resolve()
    os.chdir(root)

    fence = _load_sibling("fence")
    sys.path[0] = str(WORKSPACE)
    fence.install(str(WORKSPACE))

    import mlflow

    forced_uri = f"file:{root}/mlruns"
    original = mlflow.set_tracking_uri
    mlflow.set_tracking_uri = lambda uri, *a, **kw: original(forced_uri, *a, **kw)

    keys = namespace_keys(script_path)

    import runpy

    # run_name is NOT "__main__": top-level imports execute, main() does not.
    namespace = runpy.run_path(str(script_path), run_name=NON_MAIN_RUN_NAME)

    requests = json.loads(Path(args.requests).read_text())
    results = [_resolve_one(request, namespace, keys) for request in requests]

    Path(args.out).write_text(json.dumps({"keys": keys, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
