"""Run a real entry-point script in an isolated temp root (plan §5.5a).

The point of Stage 1 is to capture the behaviour of the *current* code, so the harness executes the
real `run.py` / `run_mo.py` / `run_lon_parallel.py` / `run_colon_parallel.py` rather than
reimplementing any of their science. It only changes where they write.

Parent side (`run_isolated`):
    * builds a fresh temp root outside the repository, with the runtime layout the scripts expect;
    * launches this same file as a child process with cwd set to that root, `NOISYVIS_ROOT`
      pointing at it (so every `noisyvis.results.paths` location resolves inside it), and
      `MLFLOW_TRACKING_URI` set to an unreachable sentinel (R24).

Child side (`--child`):
    1. puts /workspace first on sys.path, exactly as `python run.py` does;
    2. installs the write fence, so any write into /workspace fails loudly;
    3. sets sys.argv and runs the script through runpy with run_name="__main__";
    4. extracts the compared fields *in this process*, because MO results contain DEAP
       `creator.Individual` objects that cannot be unpickled anywhere else.

Usage from tests:

    result = run_isolated("run.py", "so_onemax", "so", overrides=["run.parallel=false"])
    result.extracted["cases"]["1"]["n_evals"]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

WORKSPACE = Path("/workspace")
TESTS_DIR = WORKSPACE / "tests"
TEST_CONFIG_DIR = TESTS_DIR / "configs"
DEPS_DIR = TESTS_DIR / ".deps"
HARNESS_DIR = Path(__file__).resolve().parent

INSTANCE_DIR_NAME = "instances"  # the loaders read INSTANCES_DIR / "knapsack" (Stage 9)

# Stage 6: every entry point sets its tracking URI explicitly, into the temp root through
# NOISYVIS_ROOT (SO/MO, and the LON module level) or relative to the cwd (LON configs). This URI
# has no transport, so a missed set_tracking_uri fails immediately instead of silently logging to
# the runner's configured MLflow server (R24).
MLFLOW_SENTINEL_URI = "noisyvis-harness-sentinel://unreachable-mlflow-tracking-uri-was-not-set"

DEFAULT_TIMEOUT = 1800


class HarnessError(RuntimeError):
    """The isolated run failed, or could not be set up safely."""


@dataclass
class IsolatedRun:
    script: str
    config_name: str
    overrides: list
    returncode: int
    stdout: str
    stderr: str
    extracted: dict
    root: str


# ---------------------------------------------------------------- parent side


def make_temp_root() -> Path:
    """A throwaway runtime root with the layout the entry-point scripts assume."""
    root = Path(tempfile.mkdtemp(prefix="noisyvis-stage1-")).resolve()

    # Belt and braces: the whole design depends on this not being inside the repo.
    if root == WORKSPACE or str(root).startswith(str(WORKSPACE) + os.sep):
        shutil.rmtree(root, ignore_errors=True)
        raise HarnessError(f"refusing to use a temp root inside {WORKSPACE}: {root}")

    for relative in ("data/outputs", "data/temp", "data/warehouse"):
        (root / relative).mkdir(parents=True, exist_ok=True)

    # Read-only reference to the instances; the loaders resolve them through NOISYVIS_ROOT (INSTANCES_DIR).
    (root / INSTANCE_DIR_NAME).symlink_to(WORKSPACE / INSTANCE_DIR_NAME)
    return root


def child_env(root: Path) -> dict:
    """Environment for the scientific subprocess.

    `tests/.deps` is removed from PYTHONPATH so experiment code runs against exactly the
    production environment, never against the test runner's dependencies.
    """
    env = os.environ.copy()

    entries = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            resolved = Path(entry)
        if resolved == DEPS_DIR:
            continue
        entries.append(entry)
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    else:
        env.pop("PYTHONPATH", None)

    # Every results.paths location (warehouse, temp, SO/MO and LON-module MLflow) lands in the root.
    env["NOISYVIS_ROOT"] = str(root)
    env["MLFLOW_TRACKING_URI"] = MLFLOW_SENTINEL_URI
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NOISYVIS_HARNESS_ROOT"] = str(root)
    return env


def run_isolated(
    script: str,
    config_name: str,
    kind: str,
    overrides=(),
    timeout: int = DEFAULT_TIMEOUT,
    keep_root: bool = False,
) -> IsolatedRun:
    """Execute `script` against `tests/configs/<config_name>.yaml` in a fresh temp root.

    `kind` selects the extractor: "so", "mo", "lon", "colon" or "none".
    """
    script_path = WORKSPACE / script
    if not script_path.is_file():
        raise HarnessError(f"entry-point script not found: {script_path}")
    config_file = TEST_CONFIG_DIR / f"{config_name}.yaml"
    if not config_file.is_file():
        raise HarnessError(f"test config not found: {config_file}")

    root = make_temp_root()
    overrides = list(overrides)
    argv = [
        sys.executable,
        str(HARNESS_DIR / "run_isolated.py"),
        "--child",
        "--root", str(root),
        "--script", str(script_path),
        "--config-name", config_name,
        "--kind", kind,
        "--",
        *overrides,
    ]

    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            env=child_env(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(
            f"{script} timed out after {timeout}s on config {config_name!r}; temp root kept at {root}"
        ) from exc

    if completed.returncode != 0:
        raise HarnessError(
            f"{script} failed (exit {completed.returncode}) on config {config_name!r}.\n"
            f"temp root kept at {root}\n"
            f"--- stdout tail ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr tail ---\n{completed.stderr[-4000:]}"
        )

    extract_path = root / "extract.json"
    if kind != "none":
        if not extract_path.is_file():
            raise HarnessError(f"{script} produced no extract.json; temp root kept at {root}")
        extracted = json.loads(extract_path.read_text())
    else:
        extracted = {}

    result = IsolatedRun(
        script=script,
        config_name=config_name,
        overrides=overrides,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        extracted=extracted,
        root=str(root),
    )

    if not keep_root:
        shutil.rmtree(root, ignore_errors=True)
    return result


# ----------------------------------------------------------------- child side


def _load_sibling(name: str):
    """Import a harness module by path, before sys.path is rewritten to mimic the runner."""
    spec = importlib.util.spec_from_file_location(f"_harness_{name}", HARNESS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _child(root: Path, script_path: Path, config_name: str, kind: str, overrides: list) -> int:
    import runpy

    os.chdir(root)

    # 1. Load harness helpers while their directory is still importable, then make sys.path
    #    look exactly like `python run.py`: the script directory, /workspace, comes first.
    fence = _load_sibling("fence")
    extract_module = _load_sibling("extract")
    sys.path[0] = str(WORKSPACE)

    # 2. Nothing may be written inside /workspace from here on.
    fence.install(str(WORKSPACE))

    # MLflow is not patched (Stage 6): the entry points' own set_tracking_uri calls resolve into
    # the temp root via NOISYVIS_ROOT and the cwd, and anything they miss hits the sentinel.

    # 3. Hydra reads the command line, so build it exactly as a terminal invocation would.
    #    Flat config names keep run.py's symlink hack from writing into configs/.
    sys.argv = [
        str(script_path),
        f"--config-path={TEST_CONFIG_DIR}",
        f"--config-name={config_name}",
        "hydra.run.dir=data/outputs",
        *overrides,
    ]

    # 4. Run the real entry point.
    runpy.run_path(str(script_path), run_name="__main__")

    # 5. Extract in-process: MO results hold DEAP creator.Individual objects that only
    #    unpickle where those classes exist.
    if kind != "none":
        payload = extract_module.extract(kind, root)
        (root / "extract.json").write_text(json.dumps(payload, indent=2, allow_nan=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--root", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--kind", default="none")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if not args.child:
        parser.error("this module is executed as a child process by run_isolated()")

    return _child(
        Path(args.root).resolve(),
        Path(args.script).resolve(),
        args.config_name,
        args.kind,
        list(args.overrides),
    )


if __name__ == "__main__":
    raise SystemExit(main())
