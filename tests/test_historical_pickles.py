"""Existing persisted data must keep loading with the expected structure (plan §7.5).

This is the Stage 1 baseline for a check that reruns at Stage 8 (module locations moved) and
Stage 12 (shims removed). Stage 12 may not delete a shim until this passes without it.

Nothing here modifies, rewrites or copies an artefact: every file is opened read-only.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from historical_fixtures import FORBIDDEN_MODULE_PREFIXES, REGISTRY, Fixture


class _RecordingUnpickler(pickle.Unpickler):
    """Unpickler that records every class it resolves, substituting nothing.

    Recording `find_class` is how the artefacts are checked for embedded module paths. It costs
    nothing extra, unlike scanning 1.27 GB of opcodes, and it reflects what an actual load needs.
    """

    def __init__(self, file):
        super().__init__(file)
        self.referenced: set = set()

    def find_class(self, module, name):
        self.referenced.add((module, name))
        return super().find_class(module, name)


def _load(path: Path):
    with open(path, "rb") as handle:
        unpickler = _RecordingUnpickler(handle)
        return unpickler.load(), unpickler.referenced


def _forbidden(referenced) -> list:
    hits = []
    for module, name in sorted(referenced):
        for prefix in FORBIDDEN_MODULE_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                hits.append(f"{module}.{name}")
    return hits


@pytest.mark.parametrize("fixture", REGISTRY, ids=lambda f: f.id)
def test_historical_artefact_loads(fixture: Fixture, note):
    path = fixture.resolve()

    if path is None:
        if fixture.required:
            pytest.fail(
                f"required compatibility fixture {fixture.id!r} is missing "
                f"({fixture.path or fixture.glob}). {fixture.notes}"
            )
        note(f"compatibility fixture {fixture.id!r} is absent, so that coverage is not exercised")
        pytest.skip(f"optional fixture {fixture.id!r} not present")

    obj, referenced = _load(path)

    hits = _forbidden(referenced)
    assert not hits, (
        f"{fixture.id}: artefact embeds project module paths {hits}. The matching compatibility "
        f"shim must stay permanently (plan §7.5), and Stage 12 must not delete it."
    )

    if fixture.kind == "dataframe":
        import pandas as pd

        assert isinstance(obj, pd.DataFrame), f"{fixture.id}: expected a DataFrame, got {type(obj)}"
        assert tuple(obj.columns) == fixture.expected_columns, (
            f"{fixture.id}: column schema changed.\n"
            f"  missing: {sorted(set(fixture.expected_columns) - set(obj.columns))}\n"
            f"  added:   {sorted(set(obj.columns) - set(fixture.expected_columns))}"
        )
        if fixture.min_rows is not None:
            # >= rather than ==: real experiments legitimately append rows between stages.
            assert len(obj) >= fixture.min_rows, (
                f"{fixture.id}: {len(obj)} rows, fewer than the {fixture.min_rows} recorded at "
                f"Stage 1. Rows should never disappear."
            )
    elif fixture.kind == "payload":
        assert isinstance(obj, dict), f"{fixture.id}: expected a dict payload, got {type(obj)}"
        missing = sorted(set(fixture.expected_keys) - set(obj))
        assert not missing, f"{fixture.id}: payload is missing keys {missing}"
    else:
        pytest.fail(f"{fixture.id}: unknown fixture kind {fixture.kind!r}")


def test_registry_is_coherent():
    """Guard against a registry entry that can never resolve."""
    for fixture in REGISTRY:
        assert (fixture.path is None) != (fixture.glob is None), (
            f"{fixture.id}: set exactly one of path or glob"
        )
        if fixture.glob is not None:
            assert fixture.select in ("oldest", "newest"), f"{fixture.id}: glob needs a select"
        if fixture.kind == "dataframe":
            assert fixture.expected_columns, f"{fixture.id}: dataframe fixtures need columns"
        if fixture.kind == "payload":
            assert fixture.expected_keys, f"{fixture.id}: payload fixtures need keys"
