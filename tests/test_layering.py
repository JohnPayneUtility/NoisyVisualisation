"""The two architectural rules of plan §5.1.

Both were encoded as expected failures in Stage 1, before the tree moved, so the debt was recorded
rather than hidden. Each is un-xfailed in the stage that genuinely fixes it.

    Rule 1 -- nothing in the science, common, analysis or visualisation packages imports the
              dashboard packages. This is problem P2. Enforced from Stage 5.
    Rule 2 -- nothing in the visualisation packages imports Dash. Xfailed until Stage 10, when
              lon_stats_plots.py is split.

Each rule is a separate test with its own xfail, so they can be un-xfailed independently in the
stage that fixes them.

The rules must keep holding while the tree is moving, so the directory sets match both the old and
the new names (visualization/plotting -> viz), and both absolute (`src.dashboard`,
`noisyvis.dashboard`) and relative (`..dashboard`) imports.

The scan is AST-only: nothing is imported, so a violation is detected structurally and the test
cannot be perturbed by import side effects.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

WORKSPACE = Path("/workspace")

# Old names and their post-migration equivalents, so the rules survive the moves.
RULE1_PACKAGES = (
    "src/algorithms", "src/problems", "src/common", "src/visualization", "src/plotting",
    "src/noisyvis/algorithms", "src/noisyvis/problems", "src/noisyvis/common",
    # Stage 3 moves these packages without renaming them; the viz/ rename is Stage 10, so the
    # intermediate names must be scanned too or the rules stop following their own violations.
    "src/noisyvis/visualization", "src/noisyvis/plotting",
    "src/noisyvis/networks", "src/noisyvis/tracking", "src/noisyvis/analysis", "src/noisyvis/viz",
)
RULE2_PACKAGES = (
    "src/visualization", "src/plotting",
    "src/noisyvis/visualization", "src/noisyvis/plotting",  # Stage 3 intermediate names
    "src/noisyvis/viz",
)

DASHBOARD_PACKAGES = {"dashboard", "app", "mlflow_app"}
PACKAGE_ROOTS = {"src", "noisyvis"}


class LayeringViolation(AssertionError):
    """An import crosses a layer boundary the architecture forbids."""


def _module_of(path: Path) -> str:
    """Dotted package of the module's parent, e.g. src/visualization/traces.py -> src.visualization.

    From Stage 3, `src/` is the src-layout root rather than the package, so the real package name
    drops that component: src/noisyvis/visualization/traces.py -> noisyvis.visualization. Keeping
    the literal path here would resolve `..dashboard` to `src.noisyvis.dashboard`, whose second
    component is `noisyvis`, and rule 1 would silently stop matching its own violations.
    """
    parts = path.relative_to(WORKSPACE).parts[:-1]
    if parts[:2] == ("src", "noisyvis"):
        parts = parts[1:]
    return ".".join(parts)


def _imported_modules(path: Path):
    """(lineno, absolute dotted module) for every import in the file, relative ones resolved."""
    tree = ast.parse(path.read_text(), filename=str(path))
    package = _module_of(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    yield node.lineno, node.module
                continue
            # Relative: walk up `level - 1` packages from the containing package.
            parts = package.split(".")
            base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
            absolute = ".".join([*base, node.module] if node.module else base)
            yield node.lineno, absolute


def _python_files(packages) -> list:
    files = []
    for relative in packages:
        directory = WORKSPACE / relative
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    return files


def _scan(packages, is_violation) -> list:
    """Collect violations. Raises (not xfail-covered) if there is nothing to scan."""
    files = _python_files(packages)
    if not files:
        raise RuntimeError(
            f"layering scan found no Python files under {list(packages)}; the directory set is "
            f"stale and the rule is not actually being enforced"
        )

    violations = []
    for path in files:
        for lineno, module in _imported_modules(path):
            if is_violation(module):
                violations.append(f"{path.relative_to(WORKSPACE)}:{lineno} imports {module}")
    return violations


def _imports_dashboard(module: str) -> bool:
    parts = module.split(".")
    if parts[0] in PACKAGE_ROOTS and len(parts) > 1:
        return parts[1] in DASHBOARD_PACKAGES
    return False


def _imports_dash(module: str) -> bool:
    top = module.split(".")[0]
    # `dash` itself, plus the legacy standalone component packages (dash_table, dash_core_components,
    # dash_html_components, dash_bootstrap_components).
    return top == "dash" or top.startswith("dash_")


def test_science_does_not_import_dashboard():
    violations = _scan(RULE1_PACKAGES, _imports_dashboard)
    if violations:
        raise LayeringViolation(
            "science/common/analysis/visualisation must not import the dashboard packages:\n  "
            + "\n  ".join(violations)
        )


@pytest.mark.xfail(
    strict=True,
    raises=LayeringViolation,
    reason="Rule 2: lon_stats_plots.py imports dash_table, pinning Dash onto the visualisation "
           "layer. Fixed in Stage 10, which splits that module and un-xfails this test.",
)
def test_visualisation_does_not_import_dash():
    violations = _scan(RULE2_PACKAGES, _imports_dash)
    if violations:
        raise LayeringViolation(
            "the visualisation packages must not import Dash:\n  " + "\n  ".join(violations)
        )
