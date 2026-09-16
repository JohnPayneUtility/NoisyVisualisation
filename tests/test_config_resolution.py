"""Every config dotted path and dynamic name still resolves (plan §5.6).

The config corpus reaches the code through four channels, all of which break silently:

    _target_        Hydra instantiate()/call()          228 occurrences
    violation_fn    importlib, via run_colon_parallel    37 occurrences
    fitness_fn      getattr(sys.modules[...], name)     150 occurrences
    attr_function   getattr(sys.modules[...], name)     150 occurrences

The dynamic names are resolved **through each runner's own namespace mechanism**, inside a
subprocess that has loaded that runner with a non-"__main__" run name. See
`harness/resolve_in_runner.py` for why the test must not import those namespaces itself.

Configs are read, never written.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from harness.run_isolated import HARNESS_DIR, WORKSPACE, child_env, make_temp_root
from known_broken_configs import KNOWN_BROKEN

CONFIGS_DIR = WORKSPACE / "configs"

# run_backup.py is excluded: it is a Tier A deletion in Stage 2.
RUNNERS = {
    "so": "run.py",
    "mo": "run_mo.py",
    "lon": "run_lon.py",
    "lon_parallel": "run_lon_parallel.py",
    "colon": "run_colon_parallel.py",
}

KEY_KINDS = {
    "_target_": "target",
    "violation_fn": "violation",
    "fitness_fn": "fitness",
    "attr_function": "attr",
}


class ConfigResolutionError(AssertionError):
    """A config references something that cannot be resolved."""


def _walk(node, out: list) -> None:
    """Collect every dotted path / dynamic name at any depth of a raw config."""
    if isinstance(node, dict):
        for key, value in node.items():
            kind = KEY_KINDS.get(key)
            if kind and isinstance(value, str) and value:
                out.append((kind, value))
            else:
                _walk(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk(item, out)


def _classify(raw: dict, requests: list) -> list:
    """Which runner(s) would execute this config.

    Fragments under configs/defaults/ are not standalone configurations, so they are checked
    against every runner: each runner star-imports both dynamic namespaces, so a name that
    resolves in one must resolve in all.
    """
    targets = [value for kind, value in requests if kind == "target"]
    has_violation = any(kind == "violation" for kind, _ in requests)

    if has_violation:
        return ["colon"]
    if isinstance(raw, dict) and "lon" in raw:
        return ["lon", "lon_parallel"]
    if any("MOAlgorithms" in target for target in targets):
        return ["mo"]
    if any("algorithms.Algorithms" in target for target in targets):
        return ["so"]
    return list(RUNNERS)


def _config_files() -> list:
    return sorted(CONFIGS_DIR.rglob("*.yaml"))


def _collect() -> dict:
    """{config path relative to configs/: {"requests": [...], "runners": [...]}}"""
    collected = {}
    for path in _config_files():
        raw = yaml.safe_load(path.read_text())
        requests: list = []
        _walk(raw, requests)
        relative = str(path.relative_to(CONFIGS_DIR))
        collected[relative] = {
            "requests": requests,
            "runners": _classify(raw, requests) if requests else [],
        }
    return collected


COLLECTED = _collect()


@pytest.fixture(scope="session")
def resolution_results():
    """Resolve every request, one subprocess per runner, and index the answers."""
    by_runner: dict = {name: [] for name in RUNNERS}
    for config, entry in COLLECTED.items():
        for kind, value in entry["requests"]:
            for runner in entry["runners"]:
                by_runner[runner].append({"config": config, "kind": kind, "value": value})

    answers: dict = {}
    for runner, requests in by_runner.items():
        if not requests:
            continue
        root = make_temp_root()
        try:
            with tempfile.TemporaryDirectory() as workdir:
                requests_path = Path(workdir) / "requests.json"
                out_path = Path(workdir) / "results.json"
                requests_path.write_text(json.dumps(requests))

                completed = subprocess.run(
                    [
                        sys.executable,
                        str(HARNESS_DIR / "resolve_in_runner.py"),
                        "--root", str(root),
                        "--script", str(WORKSPACE / RUNNERS[runner]),
                        "--requests", str(requests_path),
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
                        f"could not load runner {RUNNERS[runner]} for config resolution "
                        f"(exit {completed.returncode}):\n{completed.stderr[-4000:]}"
                    )

                payload = json.loads(out_path.read_text())

            # The runner must expose both dynamic namespace keys, or the gate is not
            # actually testing the mechanism it claims to test.
            assert set(payload["keys"]) >= {"fitness", "attr"}, (
                f"{RUNNERS[runner]}: could not read both sys.modules keys from the runner source; "
                f"found {payload['keys']}"
            )

            for result in payload["results"]:
                answers.setdefault(result["config"], []).append({**result, "runner": runner})
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    return answers


def _params():
    for config in COLLECTED:
        reason = KNOWN_BROKEN.get(config)
        marks = (
            [pytest.mark.xfail(strict=True, raises=ConfigResolutionError, reason=reason)]
            if reason
            else []
        )
        yield pytest.param(config, marks=marks, id=config)


@pytest.mark.parametrize("config", list(_params()))
def test_config_resolves(config, resolution_results):
    failures = [
        f"{result['kind']} {result['value']!r} via {RUNNERS[result['runner']]}: {result['error']}"
        for result in resolution_results.get(config, [])
        if not result["ok"]
    ]
    if failures:
        raise ConfigResolutionError(f"configs/{config} does not resolve:\n  " + "\n  ".join(failures))


def test_known_broken_entries_exist():
    """A stale allowlist entry is itself a failure."""
    missing = [config for config in KNOWN_BROKEN if config not in COLLECTED]
    assert not missing, (
        f"known_broken_configs.py lists configs that no longer exist: {missing}. "
        f"Remove the entries rather than leaving the allowlist stale."
    )


def test_corpus_was_actually_walked():
    """Guard against a silent pass caused by collecting nothing."""
    assert len(COLLECTED) > 100, f"only {len(COLLECTED)} configs found under {CONFIGS_DIR}"
    total = sum(len(entry["requests"]) for entry in COLLECTED.values())
    assert total > 500, f"only {total} dotted paths/dynamic names collected; the walk looks broken"
