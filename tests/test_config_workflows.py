"""Config-resolution semantics per workflow, characterised before Stage 7 moved them (plan §5.4).

Stage 7 replaces the `resolve_config_dependencies` copies in the run scripts with
`noisyvis.experiments.config` workflow functions. The SO, MO, LON and CoLON copies differ in ways
the five reproducibility baselines never reach: None versus missing versus falsy defaults, the MO
mutation container, the CoLON `is not None` guards, the MO reference point, PID resolution, and
the exceptions raised on malformed input.

`config_workflow_cases.py` pins each of those branches with a synthetic config. The golden
`baselines/config_workflows.json` was recorded once, from the untouched pre-Stage-7 resolvers
(commit 777f46d). It is **frozen**: a mismatch means resolution behaviour changed and must be
investigated. It is never re-recorded to make a comparison pass, so this module refuses to
overwrite it in any record mode.

Resolution runs in a harness subprocess, so the pytest process never imports the science
packages.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from config_workflow_cases import build_cases
from harness import canonical
from harness.run_isolated import HARNESS_DIR, child_env, make_temp_root

GOLDEN_PATH = Path(__file__).resolve().parent / "baselines" / "config_workflows.json"
RECORD_ENV = "NOISYVIS_RECORD_BASELINES"

# Provenance written into the golden when it is recorded. Recording happens exactly once.
RECORDED_FROM_COMMIT = "777f46d"

# Every resolver listed for a workflow must reproduce that workflow's recorded outcomes.
#
# The golden was recorded from the run scripts' own `resolve_config_dependencies` copies
# (run.py, run_mo.py, run_lon.py + run_lon_parallel.py, run_colon_parallel.py; see
# `recorded_from` in the golden). Stage 7 Checkpoint B replaced those copies with these workflow
# functions, which both LON scripts now share.
RESOLVERS = {
    "so": ["noisyvis.experiments.config.workflows:resolve_so_config"],
    "mo": ["noisyvis.experiments.config.workflows:resolve_mo_config"],
    "lon": ["noisyvis.experiments.config.workflows:resolve_lon_config"],
    "colon": ["noisyvis.experiments.config.workflows:resolve_colon_config"],
}


def _record_requested() -> bool:
    return os.environ.get(RECORD_ENV, "").strip().lower() not in ("", "0", "false", "no")


@pytest.fixture(scope="session")
def workflow_outcomes():
    """Resolve every case through every listed resolver, in one subprocess."""
    root = make_temp_root()
    try:
        with tempfile.TemporaryDirectory() as workdir:
            resolvers_path = Path(workdir) / "resolvers.json"
            out_path = Path(workdir) / "outcomes.json"
            resolvers_path.write_text(json.dumps(RESOLVERS))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_DIR / "resolve_workflows.py"),
                    "--root", str(root),
                    "--resolvers", str(resolvers_path),
                    "--out", str(out_path),
                ],
                cwd=str(root),
                env=child_env(root),
                capture_output=True,
                text=True,
                timeout=900,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"config workflow resolution failed (exit {completed.returncode}):\n"
                    f"{completed.stderr[-4000:]}"
                )
            return json.loads(out_path.read_text())
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def golden(workflow_outcomes, request):
    """The frozen golden, recorded once from the first resolver of each workflow if absent."""
    if GOLDEN_PATH.is_file():
        if _record_requested():
            from conftest import record_note

            record_note(
                request.config,
                f"{GOLDEN_PATH.name} is frozen and was not re-recorded (Stage 7 characterisation)",
            )
        return json.loads(GOLDEN_PATH.read_text())

    if not _record_requested():
        pytest.fail(
            f"no golden at {GOLDEN_PATH}. It is recorded once, from the pre-Stage-7 resolvers, "
            f"with {RECORD_ENV}=1; a missing golden is never skipped."
        )

    cases = {}
    for workflow, specs in RESOLVERS.items():
        first = workflow_outcomes[workflow][specs[0]]
        for other in specs[1:]:
            difference = canonical.first_difference(first, workflow_outcomes[workflow][other])
            assert difference is None, (
                f"refusing to record: {specs[0]} and {other} disagree for {workflow}: {difference}"
            )
        cases[workflow] = first

    GOLDEN_PATH.write_text(
        canonical.dumps(
            {
                "schema": canonical.SCHEMA,
                "golden": "config_workflows",
                "recorded_from": {"commit": RECORDED_FROM_COMMIT, "resolvers": RESOLVERS},
                "cases": cases,
            }
        )
    )
    from conftest import record_note

    record_note(request.config, f"recorded frozen golden {GOLDEN_PATH.name}")
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("workflow", list(RESOLVERS))
def test_workflow_matches_golden(workflow, workflow_outcomes, golden):
    expected = golden["cases"][workflow]
    for spec in RESOLVERS[workflow]:
        difference = canonical.first_difference(expected, workflow_outcomes[workflow][spec])
        assert difference is None, (
            f"{workflow}: {spec} no longer resolves configs as the pre-Stage-7 resolver did.\n"
            f"  {difference}\n"
            f"The golden is frozen: investigate the change, never re-record it."
        )


def test_golden_covers_exactly_the_defined_cases(golden):
    """A case added, renamed or dropped on either side would otherwise go unchecked."""
    defined = {workflow: sorted(cases) for workflow, cases in build_cases().items()}
    recorded = {workflow: sorted(cases) for workflow, cases in golden["cases"].items()}
    assert recorded == defined
    assert golden["recorded_from"]["commit"] == RECORDED_FROM_COMMIT
