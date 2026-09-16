"""The Stage 6 filesystem-path contract of `noisyvis.results.paths` (plan §5.9).

Each check runs in a fresh subprocess, for two reasons: the constants are computed at import time
from the environment, and the pytest process itself must never import the science packages. The
subprocess environment comes from the harness (`tests/.deps` stripped, no bytecode writes), with the
two path variables set or removed explicitly per test.

Deliberately narrow: this covers where the constants point, not what later stages do with them
(instances are Stage 9; the fast_storage configs are unchanged, R12).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.run_isolated import WORKSPACE, child_env

PATH_ENV = ("NOISYVIS_ROOT", "NOISYVIS_FAST_STORAGE")

CONSTANTS = (
    "PROJECT_ROOT", "DATA_DIR", "WAREHOUSE_DIR", "TEMP_DIR", "MLRUNS_DIR", "PLOTS_DIR",
    "INSTANCES_DIR", "FAST_STORAGE",
)

_PROBE = """
import json
import noisyvis.results.paths as paths
print(json.dumps({
    name: {"value": str(getattr(paths, name)), "type": type(getattr(paths, name)).__module__}
    for name in %r
}))
""" % (CONSTANTS,)


def _probe(cwd: str, root: str | None) -> dict:
    """Import the module in a clean subprocess and report every constant."""
    env = child_env(Path(root) if root is not None else WORKSPACE)
    for name in PATH_ENV:
        env.pop(name, None)
    if root is not None:
        env["NOISYVIS_ROOT"] = root
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=cwd, env=env, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, f"importing noisyvis.results.paths failed:\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _expected(root: Path) -> dict:
    return {
        "PROJECT_ROOT": root,
        "DATA_DIR": root / "data",
        "WAREHOUSE_DIR": root / "data" / "warehouse",
        "TEMP_DIR": root / "data" / "temp",
        "MLRUNS_DIR": root / "data" / "mlruns",
        "PLOTS_DIR": root / "plots",
        "INSTANCES_DIR": root / "instances",
        "FAST_STORAGE": root / "fast_storage",
    }


def test_default_root_is_the_repository_regardless_of_cwd():
    """With NOISYVIS_ROOT unset and an unrelated cwd, the root is the repository itself."""
    constants = _probe(cwd="/tmp", root=None)

    for name, reported in constants.items():
        assert reported["type"] == "pathlib", f"{name} is not a pathlib.Path: {reported}"

    project_root = Path(constants["PROJECT_ROOT"]["value"])
    assert project_root == WORKSPACE, f"PROJECT_ROOT resolved to {project_root}, not {WORKSPACE}"
    assert (project_root / "configs").is_dir(), f"{project_root} does not contain configs/"

    observed = {name: Path(reported["value"]) for name, reported in constants.items()}
    assert observed == _expected(WORKSPACE)


def test_noisyvis_root_overrides_every_derived_constant():
    """NOISYVIS_ROOT alone relocates every constant; nothing needs to exist to derive them."""
    root = Path("/nonexistent/noisyvis-stage6-root")
    constants = _probe(cwd="/tmp", root=str(root))

    observed = {name: Path(reported["value"]) for name, reported in constants.items()}
    assert observed == _expected(root)


def test_import_has_no_filesystem_write_side_effects():
    """Importing the module creates neither the root nor anything beneath it."""
    parent = Path(tempfile.mkdtemp(prefix="noisyvis-paths-"))
    try:
        root = parent / "root"
        constants = _probe(cwd=str(parent), root=str(root))

        assert Path(constants["PROJECT_ROOT"]["value"]) == root
        assert not root.exists(), f"importing paths created {root}"
        assert list(parent.iterdir()) == [], f"importing paths wrote {list(parent.iterdir())}"
    finally:
        shutil.rmtree(parent, ignore_errors=True)
