"""Session-scoped production-data guard and shared reporting (plan §5.5a).

The entry-point scripts write the production warehouse, `data/mlruns` and `data/temp`. A Stage 1
test session must leave all of it untouched. Isolation is enforced in three layers; this file is the
third, the detective one:

1. the harness runs every entry point in a temp root outside the repository;
2. `harness/fence.py` raises before any write into /workspace can happen;
3. this guard records production state before the session and asserts it is unchanged afterwards.

Both snapshots are printed in the terminal summary, so a run leaves evidence rather than a claim.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

WORKSPACE = Path("/workspace")
DEPS_DIR = WORKSPACE / "tests" / ".deps"

# The ACTIVE production paths. These four, and only these four, are what the guard watches.
#
# Renamed to data/warehouse in Stage 6; update this constant in that stage's commit.
WAREHOUSE_DIR = WORKSPACE / "data" / "dashboard_dw"
MLRUNS_DIR = WORKSPACE / "data" / "mlruns"
TEMP_DIR = WORKSPACE / "data" / "temp"
CONFIGS_DIR = WORKSPACE / "configs"

# data/old_mlruns is the archived pre-reset MLflow store. It is deliberately NOT part of the
# active guard and is never read, listed or walked here: it is archival, is not the store the
# services use, and walking its ~5.3M files cost ~14 minutes per session before the reset.
# If it is ever restored to active use, add it back to the watched set above.

_NOTES_KEY = "_noisyvis_notes"


# --------------------------------------------------------------- environment contract


def pytest_configure(config):
    """Fail early and explicitly if the documented test-runner contract is not met."""
    setattr(config, _NOTES_KEY, [])

    missing = []
    for name, minimum in (("packaging", (20,)), ("pygments", (2, 7, 2))):
        try:
            module = __import__(name)
        except ImportError:
            missing.append(f"{name} is not importable")
            continue
        raw = getattr(module, "__version__", "0")
        parts = []
        for chunk in raw.split(".")[:3]:
            digits = "".join(ch for ch in chunk if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        if tuple(parts) < minimum:
            missing.append(f"{name} {raw} is older than the required {'.'.join(map(str, minimum))}")

    if missing:
        raise pytest.UsageError(
            "test-runner dependency contract not met: "
            + "; ".join(missing)
            + ".\ntests/.deps deliberately contains only pytest, pluggy and iniconfig; "
            "packaging and pygments come from the locked `exp` environment on purpose, so that "
            "the locked versions are never shadowed. See tests/requirements-test.txt."
        )


def record_note(config, text: str) -> None:
    """Attach an observation to the terminal summary."""
    getattr(config, _NOTES_KEY).append(text)


@pytest.fixture
def note(request):
    """Fixture form of `record_note`, for tests that report rather than assert."""

    def _note(text: str) -> None:
        record_note(request.config, text)

    return _note


# ------------------------------------------------------------------- state snapshots


def _scan_tree(path: str) -> tuple:
    """(file count, newest file mtime_ns, dir count, newest dir mtime_ns) for one subtree."""
    files = dirs = 0
    newest_file = newest_dir = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs += 1
                            newest_dir = max(newest_dir, entry.stat(follow_symlinks=False).st_mtime_ns)
                            stack.append(entry.path)
                        else:
                            files += 1
                            newest_file = max(newest_file, entry.stat(follow_symlinks=False).st_mtime_ns)
                    except FileNotFoundError:
                        # A file vanishing mid-walk is itself a change; the counts will differ.
                        continue
        except (FileNotFoundError, PermissionError):
            continue
    return files, newest_file, dirs, newest_dir


def _walk_stats(root: Path) -> dict:
    """File/dir counts and newest mtimes for a whole tree.

    data/mlruns holds ~5.3M files, so the top-level subdirectories are walked in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not root.exists():
        return {"exists": False}

    tops = []
    loose_files = 0
    newest_loose = 0
    top_dirs = 0
    newest_top_dir = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                top_dirs += 1
                newest_top_dir = max(newest_top_dir, entry.stat(follow_symlinks=False).st_mtime_ns)
                tops.append(entry.path)
            else:
                loose_files += 1
                newest_loose = max(newest_loose, entry.stat(follow_symlinks=False).st_mtime_ns)

    files, newest_file, dirs, newest_dir = loose_files, newest_loose, top_dirs, newest_top_dir
    if tops:
        with ThreadPoolExecutor(max_workers=min(16, len(tops))) as pool:
            for sub_files, sub_newest_file, sub_dirs, sub_newest_dir in pool.map(_scan_tree, tops):
                files += sub_files
                dirs += sub_dirs
                newest_file = max(newest_file, sub_newest_file)
                newest_dir = max(newest_dir, sub_newest_dir)

    return {
        "exists": True,
        "files": files,
        "newest_file_mtime_ns": newest_file,
        "dirs": dirs,
        "newest_dir_mtime_ns": newest_dir,
    }


def production_snapshot() -> dict:
    warehouse = {}
    if WAREHOUSE_DIR.is_dir():
        for pickle_path in sorted(WAREHOUSE_DIR.glob("*.pkl")):
            stat = pickle_path.stat()
            warehouse[pickle_path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    return {
        "warehouse": warehouse,
        "mlruns": _walk_stats(MLRUNS_DIR),
        "temp": _walk_stats(TEMP_DIR),
        # run.py's __main__ symlink hack writes into configs/ for nested config names.
        # Flat test config names avoid it; this proves none was left behind.
        "configs_entries": sorted(os.listdir(CONFIGS_DIR)) if CONFIGS_DIR.is_dir() else [],
    }


def _describe_changes(before: dict, after: dict) -> list:
    changes = []
    for section in ("warehouse", "mlruns", "temp", "configs_entries"):
        if before[section] != after[section]:
            changes.append(f"{section}:\n  before: {before[section]}\n  after:  {after[section]}")
    return changes


@pytest.fixture(scope="session", autouse=True)
def production_data_guard(request):
    """Assert the real warehouse, MLflow store and temp data are untouched by the session."""
    before = production_snapshot()
    request.config._noisyvis_guard_before = before

    yield before

    after = production_snapshot()
    request.config._noisyvis_guard_after = after

    changes = _describe_changes(before, after)
    if changes:
        raise AssertionError(
            "PRODUCTION DATA CHANGED DURING THE TEST SESSION.\n"
            "A test run must never write the real warehouse, MLflow store or temp data.\n\n"
            + "\n\n".join(changes)
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    before = getattr(config, "_noisyvis_guard_before", None)
    after = getattr(config, "_noisyvis_guard_after", None)
    notes = getattr(config, _NOTES_KEY, [])

    if before is None:
        return

    write = terminalreporter.write_line
    write("")
    write("production-data guard (plan §5.5a)")
    for section in ("warehouse", "mlruns", "temp"):
        if after is None:
            write(f"  {section}: before={before[section]} (session did not complete)")
        else:
            state = "UNCHANGED" if before[section] == after[section] else "CHANGED"
            write(f"  {section}: {state} {before[section]}")
    if after is not None:
        state = "UNCHANGED" if before["configs_entries"] == after["configs_entries"] else "CHANGED"
        write(f"  configs/ entries: {state} ({len(before['configs_entries'])} entries)")

    if notes:
        write("")
        write("stage 1 observations")
        for note_text in notes:
            write(f"  - {note_text}")
