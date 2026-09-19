"""Resolve config dotted paths and dynamic names inside a runner's own namespace (plan §5.6).

The dynamic lookups in the run scripts are `getattr(sys.modules['src.problems'], name)` and
`getattr(sys.modules['src.algorithms'], name)`. Those keys exist only because each script opens with
`from src.problems import *` and `from src.algorithms import *`. Importing either namespace
explicitly from the test would *register the key* and so hide exactly the failure this gate exists
to catch.

So this child process loads the runner itself, with a `run_name` other than "__main__" (its
top-level imports run, `main` does not), reads the `sys.modules[...]` keys out of the runner's
source by AST, and resolves every name through them.

Stage 7 moves the run-script bodies into `noisyvis.experiments`, so a root script can be a thin
wrapper whose lookups live in a module it imports. The keys are therefore read from the runner
script **and** every `noisyvis.experiments` module it imports, followed transitively inside that
package. The scan is static: those modules are located on disk, never imported by this harness.
If two scanned files use different keys for the same kind of lookup, that is an error rather than
a choice. The `sys.modules` lookup itself stays a dict lookup, after the runner has been loaded.

Dotted paths use the same mechanisms the runners use:
    _target_      -> hydra._internal.utils._locate, what instantiate()/call() use
    violation_fn  -> the runner's _import_from_dotted (importlib + getattr): taken from the runner
                     namespace, or else from one of its already-loaded noisyvis.experiments modules
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
SRC_ROOT = WORKSPACE / "src"
HARNESS_DIR = Path(__file__).resolve().parent
NON_MAIN_RUN_NAME = "noisyvis_stage1_cfgcheck"
DELEGATE_PACKAGE = "noisyvis.experiments"


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"_harness_{name}", HARNESS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _in_delegate_package(dotted: str) -> bool:
    return dotted == DELEGATE_PACKAGE or dotted.startswith(DELEGATE_PACKAGE + ".")


def _module_file(dotted: str) -> Path | None:
    """The source file of a module under src/, located on disk without importing anything."""
    base = SRC_ROOT.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imported_delegates(path: Path, module_name: str | None) -> list:
    """noisyvis.experiments modules imported by one source file, in source order."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package = None
    if module_name is not None:
        package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if package is None:
                    continue
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            # `from pkg import name` may import a submodule called `name`
            names = [module] + [f"{module}.{alias.name}" for alias in node.names if alias.name != "*"]
        else:
            continue
        for name in names:
            if _in_delegate_package(name) and _module_file(name) is not None:
                found.append(name)
    return found


def delegate_modules(script_path: Path) -> list:
    """Every noisyvis.experiments module the runner imports, transitively within that package."""
    ordered: list = []
    pending = _imported_delegates(script_path, None)
    while pending:
        name = pending.pop(0)
        if name in ordered:
            continue
        ordered.append(name)
        pending.extend(_imported_delegates(_module_file(name), name))
    return ordered


def _keys_in_source(path: Path) -> list:
    """(kind, key) for every `getattr(sys.modules['<key>'], <expr>)` in one source file.

    The kind is classified by the looked-up expression: a fitness-function lookup or an
    attribute-generator lookup.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []

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
            found.append(("fitness", key_node.value))
        elif "attr" in wanted:
            found.append(("attr", key_node.value))

    return found


def namespace_keys(script_path: Path) -> tuple:
    """The sys.modules keys a runner actually uses, and the files they were read from.

    Scans the runner script and its noisyvis.experiments delegates. Conflicting keys for the same
    kind raise instead of silently picking one.
    """
    keys: dict = {}
    origins: dict = {}
    files = [(str(script_path), script_path)]
    files += [(name, _module_file(name)) for name in delegate_modules(script_path)]

    for label, path in files:
        for kind, key in _keys_in_source(path):
            if kind in keys and keys[kind] != key:
                raise RuntimeError(
                    f"conflicting sys.modules keys for {kind!r} lookups: {keys[kind]!r} in "
                    f"{origins[kind]} but {key!r} in {label}"
                )
            if kind not in keys:
                keys[kind] = key
                origins[kind] = label

    return keys, origins, [label for label, _ in files]


def _find_importer(namespace: dict, delegates: list):
    """The runner's `_import_from_dotted`: from its own namespace, else an already-loaded delegate."""
    importer = namespace.get("_import_from_dotted")
    if importer is not None:
        return importer
    for name in delegates:
        module = sys.modules.get(name)  # loaded by the runner itself; never imported here
        if module is not None and getattr(module, "_import_from_dotted", None) is not None:
            return module._import_from_dotted
    return None


def _resolve_one(request: dict, namespace: dict, keys: dict, delegates: list) -> dict:
    kind = request["kind"]
    value = request["value"]
    result = dict(request)

    try:
        if kind == "target":
            from hydra._internal.utils import _locate

            _locate(value)
        elif kind == "violation":
            importer = _find_importer(namespace, delegates)
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

    # No MLflow patch (Stage 6): the runner's import-time set_tracking_uri resolves into the temp
    # root through NOISYVIS_ROOT, and the environment carries the harness's unreachable sentinel.

    keys, key_origins, scanned = namespace_keys(script_path)
    delegates = scanned[1:]

    import runpy

    # run_name is NOT "__main__": top-level imports execute, main() does not.
    namespace = runpy.run_path(str(script_path), run_name=NON_MAIN_RUN_NAME)

    requests = json.loads(Path(args.requests).read_text())
    results = [_resolve_one(request, namespace, keys, delegates) for request in requests]

    # Stage 12: after the runner loaded and every request resolved, which compatibility modules got
    # loaded (none may).
    compat = _load_sibling("compat_modules")

    Path(args.out).write_text(
        json.dumps(
            {"keys": keys, "key_origins": key_origins, "scanned": scanned, "results": results,
             "compat_modules_loaded": compat.compat_modules_loaded()},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
