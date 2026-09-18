"""Pure-AST inventory of the dashboard and MLflow-app statements that Stage 11 redistributes.

Shared by `tests/test_dashboard_package.py` and the `tests/test_viz_package.py` probe, and loaded by
file path in both, so it never depends on the caller's `sys.path`. It imports nothing outside the
standard library and imports no `noisyvis` module: an inventory must be readable from a tree that
cannot be imported.

The unit is a **top-level statement**, keyed so that the same statement can be found again after it
moves to another module:

    ("cb", "<dash callback id>")   a callback, keyed by the output id Dash itself would build
    ("def", "<name>")             any other function or class
    ("assign", "<target>")        a module-level assignment
    ("if", "<test>")              a module-level `if`
    ("main",)                     the `if __name__ == "__main__":` entrypoint block
    ("doc",)                      the module docstring

Callback ids are built statically from the decorator, resolving the store-id constants of
`dashboard/layout/stores.py`, and are checked against the live `callback_map` keys by
`test_dashboard_package`.

Two transformations are expected during Stage 11 and are inverted here, so that a moved statement
still hashes to its PRE_STAGE_11 value:

    T1  the B9 renames (`RENAMES`), keyed by output id, which is stable across the rename
    T2  an in-body relative import, whose level changes with the module's package
        (`noisyvis.dashboard.callbacks` is one level deeper than `noisyvis.dashboard`)

and one is reconstructed:

    T3  `if __name__ == "__main__": app.run(...)` becoming `def main(): app.run(...)` plus a guard
        that does nothing but call `main()`
    T4  the MLflow page's `render_experiments` using `MLRUNS_DIR` instead of counting `parents[4]`
        (`invert_mlruns_dir`)

Nothing else may differ. Module docstrings are recorded separately: `DashboardHelpers`' docstring is
pinned, because A1 keeps that file's header verbatim; the `dataio` ones may change with the package's
dissolution.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

# --------------------------------------------------------------------------- PRE_STAGE_11 layout

PRE_STAGE_11_COMMIT = "fa8d250c65fb9b778a18262f165967476f824f98"

DASHBOARD_PY = "dashboard/Dashboard.py"
HELPERS_PY = "dashboard/DashboardHelpers.py"
DATAIO_INIT = "dataio/__init__.py"
TRANSFORMERS_PY = "dataio/transformers.py"
COLUMN_CONFIG_PY = "dataio/column_config.py"
MLFLOW_APP_PY = "app/app.py"
MLFLOW_PAGE_PY = "app/pages/mlflow_browser.py"

PRE_FILES = (DASHBOARD_PY, HELPERS_PY, DATAIO_INIT, TRANSFORMERS_PY, COLUMN_CONFIG_PY,
             MLFLOW_APP_PY, MLFLOW_PAGE_PY)

# Where each PRE file's statements may live afterwards. Scopes are deliberately narrow: two PRE files
# must never be able to satisfy each other's keys (`app = dash.Dash(...)` exists in both the dashboard
# and the MLflow app).
SEARCH_SCOPES = {
    DASHBOARD_PY: ("dashboard/*.py", "dashboard/callbacks/*.py"),
    HELPERS_PY: ("dashboard/*.py",),
    DATAIO_INIT: ("dataio/*.py", "dashboard/*.py"),
    TRANSFORMERS_PY: ("dataio/*.py", "dashboard/*.py", "analysis/*.py"),
    COLUMN_CONFIG_PY: ("dataio/*.py", "dashboard/*.py"),
    MLFLOW_APP_PY: ("app/*.py", "mlflow_app/*.py"),
    MLFLOW_PAGE_PY: ("app/pages/*.py", "mlflow_app/pages/*.py"),
}

# `dashboard/layout/` and `dashboard/components.py` are Stage-10 territory: never scanned, never moved.
EXCLUDED = ("dashboard/layout/",)

# B9 (plan §14), keyed by the callback's output id, which does not change.
RENAMES = {
    "..optimum.data...PID.data...opt_goal.data...fit_func_store.data..": "update_problem_stores",
    "print_STN_series_labels.children": "update_plotted_series_labels",
    "LON_data.data": "update_lon_data",
    "STN_data.data": "update_stn_data",
    "2DLinePlot.figure": "display_line_so",
    "2DBoxPlot.figure": "display_box_so",
}

MAIN_TEST = "__name__ == '__main__'"

# T4 (plan §9): in the moved MLflow page, `render_experiments` stops counting parents and uses
# `results.paths.MLRUNS_DIR`. Only this function, only at its final location, and only this exact form.
T4_MODULE = "mlflow_app/pages/mlflow_browser.py"
T4_KEY = ("def", "render_experiments")
T4_NEW = 'mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")'
T4_OLD = (
    "repo_root = Path(__file__).resolve().parents[4]",
    'mlruns_dir = repo_root / "data" / "mlruns"',
    'mlflow.set_tracking_uri(f"file:{mlruns_dir}")',
)


# --------------------------------------------------------------------------- digests

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def digest(obj) -> str:
    """The canonical form test_viz_package uses for a list of strings, so digests stay comparable."""
    return sha256(json.dumps({"list": list(obj)}, separators=(",", ":")))


def absolutise(node: ast.AST, package: str) -> ast.AST:
    """Rewrite relative ImportFrom nodes to absolute modules (same rule as test_viz_package)."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.level:
            parts = package.split(".")
            base = parts[: len(parts) - (child.level - 1)] if child.level > 1 else parts
            child.module = ".".join([*base, child.module] if child.module else base)
            child.level = 0
    return node


def relevel(node: ast.AST, from_package: str, to_package: str) -> ast.AST:
    """Express `node`'s relative imports as they would read inside `to_package` (T2)."""
    node = absolutise(node, from_package)
    prefix = to_package + "."
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
            if child.module == to_package:
                child.level, child.module = 1, None
            elif child.module.startswith(prefix):
                child.level, child.module = 1, child.module[len(prefix):]
    return node


def norm_ast_sha256(node: ast.AST, package: str) -> str:
    return sha256(ast.dump(absolutise(node, package)))


def literal_digest(node: ast.AST) -> str:
    """Multiset of every string constant, including f-string parts (I-10)."""
    return digest(sorted(child.value for child in ast.walk(node)
                         if isinstance(child, ast.Constant) and isinstance(child.value, str)))


# --------------------------------------------------------------------------- callback ids

def store_constants(pkg: Path) -> dict:
    """The `dashboard/layout/stores.py` id constants, read statically (e.g. LON_TABLE_SELECTED_PID_STORE)."""
    path = pkg / "dashboard" / "layout" / "stores.py"
    constants = {}
    if not path.is_file():
        return constants
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return constants


def _const(node, constants):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    raise ValueError(f"cannot resolve statically: {ast.unparse(node)}")


def callback_decorator(node):
    """The `@<something>.callback(...)` decorator of a function, or None."""
    for decorator in getattr(node, "decorator_list", []):
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) \
                and decorator.func.attr == "callback":
            return decorator
    return None


def callback_output_id(decorator, constants) -> str:
    """The callback id Dash builds from the decorator's Output specs (`dash._utils.create_callback_id`)."""
    flat = []
    for arg in decorator.args:
        flat.extend(arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg])
    outputs = [arg for arg in flat if isinstance(arg, ast.Call)
               and getattr(arg.func, "id", None) == "Output"]
    if not outputs:
        raise ValueError("callback decorator without an Output")
    parts = [f"{_const(o.args[0], constants)}.{_const(o.args[1], constants)}" for o in outputs]
    if len(parts) == 1:
        return parts[0]
    return ".." + "...".join(parts) + ".."


# --------------------------------------------------------------------------- statements

def statement_key(node, constants):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        decorator = callback_decorator(node)
        if decorator is not None:
            return ("cb", callback_output_id(decorator, constants))
        return ("def", node.name)
    if isinstance(node, ast.ClassDef):
        return ("def", node.name)
    if isinstance(node, ast.Assign):
        return ("assign", ast.unparse(node.targets[0]))
    if isinstance(node, ast.If):
        if ast.unparse(node.test) == MAIN_TEST:
            return ("main",)
        return ("if", ast.unparse(node.test))
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ("doc",)
        return ("expr", ast.unparse(node)[:80])
    return (type(node).__name__, ast.unparse(node)[:80])


def package_of(pkg: Path, path: Path) -> str:
    """noisyvis.dashboard.callbacks for src/noisyvis/dashboard/callbacks/visualization.py."""
    parts = path.relative_to(pkg.parent).parts[:-1]
    return ".".join(parts)


def module_key(pkg: Path, path: Path) -> str:
    return str(path.relative_to(pkg)).replace("\\", "/")


def statements(pkg: Path, path: Path, constants) -> list:
    """[(key, node)] for every non-import top-level statement, in file order."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [(statement_key(node, constants), node) for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))]


def scope_files(pkg: Path, pre_file: str) -> list:
    files = []
    for pattern in SEARCH_SCOPES[pre_file]:
        for path in sorted(pkg.glob(pattern)):
            key = module_key(pkg, path)
            if any(key.startswith(prefix) for prefix in EXCLUDED):
                continue
            if path.name == "__pycache__":
                continue
            files.append(path)
    return files


def index(pkg: Path, pre_file: str, constants) -> dict:
    """key -> [(module_key, package, node)] across every module the PRE file may have moved into.

    Module docstrings are excluded: every module has one, so they are matched through the PRE file's
    first real statement instead (`_docstring_of`).
    """
    found = {}
    for path in scope_files(pkg, pre_file):
        for key, node in statements(pkg, path, constants):
            if key == ("doc",):
                continue
            found.setdefault(key, []).append((module_key(pkg, path), package_of(pkg, path), node))
    return found


def _anchor_module(found, frozen_entries):
    """The module holding the PRE file's first non-docstring statement."""
    for entry in frozen_entries:
        key = tuple(entry["key"])
        if key == ("doc",):
            continue
        matches = found.get(key, [])
        if len(matches) == 1:
            return matches[0]
        return None
    return None


def _docstring_of(pkg: Path, module: str):
    """(module, package, node) for that module's docstring, or []."""
    path = pkg / module
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            return [(module, package_of(pkg, path), node)]
        break
    return []


def inventory(pkg: Path, pre_file: str, constants) -> list:
    """The PRE file's own statements: [{key, kind, name, ast, literals, module}]."""
    path = pkg / pre_file
    package = package_of(pkg, path)
    entries = []
    for key, node in statements(pkg, path, constants):
        entries.append({
            "key": list(key),
            "name": getattr(node, "name", None),
            "ast": norm_ast_sha256(node, package),
            "literals": literal_digest(node),
            "module": module_key(pkg, path),
        })
    return entries


def locate(pkg: Path, pre_file: str, frozen_entries: list, constants) -> list:
    """Find every frozen statement wherever it now lives, and hash it as it read in its PRE module.

    Returns [{key, found_in, count, name, ast, literals}]. `ast`/`literals` are computed after
    inverting T1 and T2, so an unmoved and a moved statement produce the same values.
    """
    pre_package = package_of(pkg, pkg / pre_file)
    found = index(pkg, pre_file, constants)
    anchor = _anchor_module(found, frozen_entries)
    report = []
    for entry in frozen_entries:
        key = tuple(entry["key"])
        matches = found.get(key, [])
        if key == ("doc",):
            matches = _docstring_of(pkg, anchor[0]) if anchor else []
        record = {"key": list(key), "count": len(matches)}
        if key == ("main",) and _main_matches(found, matches) is not matches:
            matches = _main_matches(found, matches)
            record["count"] = len(matches)
            record["reconstructed"] = True
        if len(matches) == 1:
            module, package, node = matches[0]
            node = relevel(node, package, pre_package)
            node = invert_mlruns_dir(key, module, node)
            record["found_in"] = module
            record["name"] = getattr(node, "name", None)
            if key[0] == "cb" and key[1] in RENAMES:
                node = copy.deepcopy(node)
                node.name = entry["name"]  # T1: hash under the PRE name
            record["ast"] = norm_ast_sha256(node, pre_package)
            record["literals"] = literal_digest(node)
        else:
            record["found_in"] = sorted(module for module, _, _ in matches)
        report.append(record)
    return report


def invert_mlruns_dir(key, module, node):
    """T4: restore the historical `parents[4]` derivation in the moved `render_experiments`.

    Applies only to that function at its final module, and only when its body holds exactly one
    `mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")` statement. That statement is replaced by the three
    historical ones; anything else about the function is left as it stands and so must hash equal.
    """
    if key != T4_KEY or module != T4_MODULE:
        return node
    new = ast.dump(ast.parse(T4_NEW).body[0])
    at = [i for i, stmt in enumerate(node.body) if ast.dump(stmt) == new]
    if len(at) != 1:
        return node
    node = copy.deepcopy(node)
    node.body[at[0]:at[0] + 1] = [ast.parse(stmt).body[0] for stmt in T4_OLD]
    return ast.fix_missing_locations(node)


def _guard_calls_main(guard, module) -> bool:
    """Is this the T3 guard: exactly `if __name__ == "__main__": main()`, beside `def main()`?"""
    if len(guard) != 1:
        return False
    guard_module, _, node = guard[0]
    return (guard_module == module and not node.orelse and len(node.body) == 1
            and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Call)
            and isinstance(node.body[0].value.func, ast.Name)
            and node.body[0].value.func.id == "main"
            and not node.body[0].value.args and not node.body[0].value.keywords)


def _reconstructed_main(found):
    """T3: rebuild `if __name__ == "__main__": <main body>` from a `def main()` entrypoint.

    The approved final form is `def main(): app.run(...)` plus a guard that does nothing but call
    `main()`; any other guard is compared as it stands and so fails against the frozen block.
    """
    matches = found.get(("def", "main"), [])
    if len(matches) != 1:
        return []
    module, package, node = matches[0]
    guard = found.get(("main",), [])
    rebuilt = ast.If(
        test=ast.parse(MAIN_TEST, mode="eval").body,
        body=copy.deepcopy(node.body),
        orelse=[],
    )
    ast.fix_missing_locations(rebuilt)
    if guard and not _guard_calls_main(guard, module):
        return []
    return [(module, package, rebuilt)]


def _main_matches(found, matches):
    """The ("main",) statement: as found, or rebuilt by T3 when absent or when it only calls main()."""
    defs = found.get(("def", "main"), [])
    if not matches or (len(defs) == 1 and _guard_calls_main(matches, defs[0][0])):
        return _reconstructed_main(found)
    return matches


def reassemble(pkg: Path, pre_file: str, frozen_entries: list, constants):
    """The PRE file's body, rebuilt in its frozen order from wherever the statements now live.

    Used to reproduce the Stage-10 BODIES hashes after the split (plan §15.1). Raises if a statement
    is missing or ambiguous, so a silent hole cannot pass as an equal hash.
    """
    pre_package = package_of(pkg, pkg / pre_file)
    found = index(pkg, pre_file, constants)
    anchor = _anchor_module(found, frozen_entries)
    body = []
    for entry in frozen_entries:
        key = tuple(entry["key"])
        matches = found.get(key, [])
        if key == ("doc",):
            matches = _docstring_of(pkg, anchor[0]) if anchor else []
        if key == ("main",):
            matches = _main_matches(found, matches)
        if len(matches) != 1:
            raise AssertionError(
                f"{pre_file}: statement {key} found {len(matches)} times "
                f"in {[m for m, _, _ in matches]}"
            )
        _, package, node = matches[0]
        node = relevel(node, package, pre_package)
        if key[0] == "cb" and key[1] in RENAMES:
            node = copy.deepcopy(node)
            node.name = entry["name"]
        body.append(node)
    # Exactly test_viz_package's `body_hash`: the raw dump of the import-free module, not the
    # absolutised one, so the frozen Stage-10 BODIES values are reproduced literally.
    module = ast.Module(body=body, type_ignores=[])
    return {"ast": sha256(ast.dump(module)),
            "literals": literal_digest(module),
            "statements": len(body)}


# --------------------------------------------------------------------------- import surface

def import_statements(pkg: Path, path: Path) -> list:
    """Every top-level import of a module, as (absolute module, name, asname) triples."""
    package = package_of(pkg, path)
    out = []
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(["import", alias.name, alias.asname])
        elif isinstance(node, ast.ImportFrom):
            resolved = absolutise(node, package)
            for alias in node.names:
                out.append(["from", resolved.module, alias.name, alias.asname])
    return out


def free_names(node: ast.AST) -> set:
    """Names a statement reads from its module namespace (same rule as test_viz_package)."""
    import builtins as _builtins

    loads, stores, args, nested = set(), set(), set(), set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            (loads if isinstance(child.ctx, ast.Load) else stores).add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            spec = child.args
            args.update(a.arg for a in spec.posonlyargs + spec.args + spec.kwonlyargs)
            if spec.vararg:
                args.add(spec.vararg.arg)
            if spec.kwarg:
                args.add(spec.kwarg.arg)
            if isinstance(child, ast.FunctionDef) and child is not node:
                nested.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            nested.update((a.asname or a.name).split(".")[0] for a in child.names)
        elif isinstance(child, ast.comprehension):
            for target in ast.walk(child.target):
                if isinstance(target, ast.Name):
                    nested.add(target.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            nested.add(child.name)
    return loads - stores - args - nested - set(dir(_builtins))
