"""The nested `--config-name` compatibility helper behind `run.py` and `run_mo.py` (Stage 7).

`noisyvis.experiments.config.cli` symlinks a nested config into the config root under a flat name,
rewrites the command-line argument in place, runs `main`, and removes the symlink. These tests pin
that behaviour against a temporary config root: they never touch /workspace/configs and never start
Hydra. (The real `run.py` path is exercised separately with `--cfg job`.)

`noisyvis.experiments.config.cli` imports only the standard library, so it is imported in-process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from noisyvis.experiments.config.cli import run_with_nested_config_name

NESTED = "family/case"
FLAT = "family__case"


def _config_root(tmp_path: Path) -> Path:
    root = tmp_path / "configs"
    (root / "family").mkdir(parents=True)
    (root / "family" / "case.yaml").write_text("value: 1\n")
    return root


def _run_and_observe(root: Path, argv: list, monkeypatch, main_raises: Exception | None = None) -> dict:
    """Run the helper with `argv` as sys.argv; record what `main` sees while it runs."""
    monkeypatch.setattr(sys, "argv", argv)
    link = root / f"{FLAT}.yaml"
    seen: dict = {}

    def main():
        seen["argv_is_same_list"] = sys.argv is argv
        seen["argv"] = list(sys.argv)
        seen["is_symlink"] = link.is_symlink()
        seen["target"] = link.resolve() if link.is_symlink() else None
        seen["root_entries"] = sorted(p.name for p in root.iterdir())
        if main_raises is not None:
            raise main_raises

    if main_raises is None:
        run_with_nested_config_name(main, root)
    else:
        with pytest.raises(type(main_raises)) as caught:
            run_with_nested_config_name(main, root)
        seen["raised"] = caught.value
    seen["symlink_exists_after"] = link.exists() or link.is_symlink()
    return seen


def test_equals_form_is_flattened_symlinked_and_cleaned_up(tmp_path, monkeypatch):
    root = _config_root(tmp_path)
    argv = ["run.py", f"--config-name={NESTED}", "run.parallel=false"]

    seen = _run_and_observe(root, argv, monkeypatch)

    assert seen["argv_is_same_list"], "the helper must rewrite sys.argv in place, not replace it"
    assert seen["argv"] == ["run.py", f"--config-name={FLAT}", "run.parallel=false"]
    assert seen["is_symlink"]
    assert seen["target"] == (root / "family" / "case.yaml").resolve()
    assert not seen["symlink_exists_after"]
    assert (root / "family" / "case.yaml").read_text() == "value: 1\n"


def test_space_form_is_flattened_symlinked_and_cleaned_up(tmp_path, monkeypatch):
    root = _config_root(tmp_path)
    argv = ["run.py", "--config-name", NESTED, "run.parallel=false"]

    seen = _run_and_observe(root, argv, monkeypatch)

    assert seen["argv_is_same_list"]
    assert seen["argv"] == ["run.py", "--config-name", FLAT, "run.parallel=false"]
    assert seen["is_symlink"]
    assert seen["target"] == (root / "family" / "case.yaml").resolve()
    assert not seen["symlink_exists_after"]
    assert (root / "family" / "case.yaml").read_text() == "value: 1\n"


def test_flat_or_absent_config_name_is_left_untouched(tmp_path, monkeypatch):
    root = _config_root(tmp_path)
    before = sorted(p.name for p in root.iterdir())

    for argv in (["run.py", "--config-name=flat_name", "x=1"], ["run.py", "x=1"]):
        expected = list(argv)
        seen = _run_and_observe(root, argv, monkeypatch)
        assert seen["argv"] == expected, "a flat or absent config name must not be rewritten"
        assert not seen["is_symlink"]
        assert seen["root_entries"] == before, "no symlink may be created for a flat or absent name"
        assert sorted(p.name for p in root.iterdir()) == before


def test_symlink_is_removed_when_main_raises(tmp_path, monkeypatch):
    root = _config_root(tmp_path)
    argv = ["run.py", f"--config-name={NESTED}"]
    failure = RuntimeError("main failed")

    seen = _run_and_observe(root, argv, monkeypatch, main_raises=failure)

    assert seen["raised"] is failure, "the original exception must propagate unchanged"
    assert seen["is_symlink"], "the symlink must exist while main runs"
    assert not seen["symlink_exists_after"], "the symlink must be removed even when main raises"
