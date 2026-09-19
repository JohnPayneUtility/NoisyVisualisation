"""Every config dotted path and dynamic name still resolves (plan §5.6).

The config corpus reaches the code through four channels, all of which break silently:

    _target_        Hydra instantiate()/call()          228 occurrences
    violation_fn    importlib, via run_colon_parallel    37 occurrences
    fitness_fn      getattr(sys.modules[...], name)     150 occurrences
    attr_function   getattr(sys.modules[...], name)     150 occurrences

The dynamic names are resolved **through each runner's own namespace mechanism**, inside a
subprocess that has loaded that runner with a non-"__main__" run name. From Stage 7 a runner may be
a thin wrapper whose lookups live in the `noisyvis.experiments` modules it imports; the harness
reads the keys from those too. See `harness/resolve_in_runner.py` for why the test must not import
those namespaces itself.

Configs are read, never written.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from harness.run_isolated import HARNESS_DIR, WORKSPACE, child_env, make_temp_root
from known_broken_configs import KNOWN_BROKEN
from legacy_paths import LEGACY_TO_CANONICAL, canonicalise, canonicalise_tree

CONFIGS_DIR = WORKSPACE / "configs"

MO_ALGORITHMS_PREFIX = LEGACY_TO_CANONICAL["src.algorithms.MOAlgorithms."]
SO_ALGORITHMS_PREFIX = LEGACY_TO_CANONICAL["src.algorithms.Algorithms."]

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
    against every runner: each runner loads both dynamic namespaces (by star import, or through
    its noisyvis.experiments delegates), so a name that resolves in one must resolve in all.

    Algorithm targets are classified by their canonical spelling (Stage 12), so a config routes to
    the same runners whether it spells them `src.algorithms.*` or `noisyvis.algorithms.*`.
    """
    targets = [canonicalise(value) for kind, value in requests if kind == "target"]
    has_violation = any(kind == "violation" for kind, _ in requests)

    if has_violation:
        return ["colon"]
    if isinstance(raw, dict) and "lon" in raw:
        return ["lon", "lon_parallel"]
    if any(target.startswith(MO_ALGORITHMS_PREFIX) for target in targets):
        return ["mo"]
    if any(target.startswith(SO_ALGORITHMS_PREFIX) for target in targets):
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
def resolution_payloads():
    """Resolve every request, one subprocess per runner, and index the answers.

    Also keeps, per runner, the compatibility modules its child had loaded once everything resolved.
    """
    by_runner: dict = {name: [] for name in RUNNERS}
    for config, entry in COLLECTED.items():
        for kind, value in entry["requests"]:
            for runner in entry["runners"]:
                by_runner[runner].append({"config": config, "kind": kind, "value": value})

    answers: dict = {}
    compat: dict = {}
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
            compat[runner] = payload["compat_modules_loaded"]
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    return {"answers": answers, "compat_modules_loaded": compat}


@pytest.fixture(scope="session")
def resolution_results(resolution_payloads):
    return resolution_payloads["answers"]


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


def test_runner_routing_is_path_neutral():
    """Every config routes to the same runners under either dotted-path spelling (Stage 12).

    Each config is classified as written, fully canonicalised, and fully spelled the legacy way. If
    routing depended on the spelling, rewriting the configs would silently change which runners
    exercise them, and this gate would test less without failing.
    """
    to_legacy = {canonical: legacy for legacy, canonical in LEGACY_TO_CANONICAL.items()}

    def legacy_spelling(value):
        for canonical, legacy in to_legacy.items():
            if isinstance(value, str) and value.startswith(canonical):
                return legacy + value[len(canonical):]
        return value

    routed = {}
    changed = []
    for path in _config_files():
        raw = yaml.safe_load(path.read_text())
        requests: list = []
        _walk(raw, requests)
        if not requests:
            continue
        config = str(path.relative_to(CONFIGS_DIR))
        as_written = _classify(raw, requests)
        canonical = _classify(raw, canonicalise_tree(requests))
        legacy = _classify(raw, [(kind, legacy_spelling(value)) for kind, value in requests])
        if not (as_written == canonical == legacy):
            changed.append(f"{config}: as written {as_written}, canonical {canonical}, legacy {legacy}")
        routed[config] = as_written

    assert not changed, "runner routing depends on the dotted-path spelling:\n  " + "\n  ".join(changed)
    assert routed == {config: entry["runners"] for config, entry in COLLECTED.items() if entry["requests"]}
    # Both algorithm routes must actually occur, or the check above is vacuous.
    assert ["mo"] in routed.values() and ["so"] in routed.values(), "no config routes to SO or MO"


def test_config_resolution_loads_no_compatibility_module(resolution_payloads):
    """Every runner loaded and resolved its whole share of the corpus without loading the bridge.

    Stage 12 moved every config dotted path to the canonical `noisyvis.*` modules. While the
    compatibility modules still existed on disk, this is what proved the configs did not need them.
    """
    compat = resolution_payloads["compat_modules_loaded"]
    assert set(compat) == set(RUNNERS), f"not every runner was exercised: {sorted(compat)}"
    loaded = {RUNNERS[runner]: modules for runner, modules in compat.items() if modules}
    assert not loaded, f"config resolution loaded compatibility modules: {loaded}"


# The five legacy module paths, anywhere in a line, but not as the tail of a longer dotted name.
_LEGACY_PATH = re.compile(
    r"(?<![A-Za-z0-9_.])(?:" + "|".join(re.escape(prefix) for prefix in LEGACY_TO_CANONICAL) + ")"
)


def test_no_legacy_config_paths():
    """No live config input uses one of the five legacy dotted paths that Stage 12 migrated.

    Live inputs are every YAML under configs/ (gitignored ones included, because the gate walks
    them), the reproducibility configs under tests/configs/, and the synthetic workflow cases. The
    frozen golden and other historical records are not live inputs and are not scanned.
    """
    from config_workflow_cases import build_cases

    yaml_files = _config_files() + sorted((WORKSPACE / "tests" / "configs").rglob("*.yaml"))
    offenders = []
    for path in yaml_files:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _LEGACY_PATH.search(line):
                offenders.append(f"{path.relative_to(WORKSPACE)}:{number}: {line.strip()}")

    def leaves(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaves(value, f"{where}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                yield from leaves(item, f"{where}[{index}]")
        elif isinstance(node, str):
            yield where, node

    cases = build_cases()
    case_strings = list(leaves(cases, "build_cases()"))
    offenders += [f"{where}: {value}" for where, value in case_strings if _LEGACY_PATH.search(value)]

    assert len(yaml_files) > 100 and any("tests/configs" in str(path) for path in yaml_files)
    assert case_strings, "build_cases() yielded no strings; the walk looks broken"
    assert not offenders, "live config inputs still use legacy dotted paths:\n  " + "\n  ".join(offenders)
