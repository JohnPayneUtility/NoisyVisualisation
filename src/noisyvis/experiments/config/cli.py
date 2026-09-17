"""The nested `--config-name` compatibility used by `run.py` and `run_mo.py` (plan Stage 7).

Hydra resolves `--config-name` only among the files directly in the config directory. So that a nested
config such as `SingleObjective/Continuous/Rastrigin2D/1p1ea` can be named on the command line, the
scripts temporarily symlink it into the config root under a flat name
(`SingleObjective__Continuous__Rastrigin2D__1p1ea.yaml`), rewrite the argument to that name, and remove
the symlink after `main` returns.

Moved verbatim from the scripts' `__main__` blocks, quirks included: every argument starting with
"--config-name" is treated as the flag, only the first is handled, the `--config-name NAME` form
reads the next argument, an existing symlink makes creation fail before `main` runs, and the symlink
is created before -- not inside -- the try/finally that removes it.

`configs_root` must be supplied by the caller: the scripts pass `Path(__file__).resolve().parent /
"configs"`. Nothing happens on import.
"""

import sys
from pathlib import Path


def flatten_nested_config_name(argv: list, configs_root: Path):
    """Rewrite a nested --config-name in `argv` (in place) to a flat symlink name; return the symlink or None."""
    symlink = None

    for i, arg in enumerate(argv):
        if arg.startswith("--config-name=") or arg.startswith("--config-name"):
            val = arg.split("=", 1)[1] if "=" in arg else argv[i + 1]
            if "/" in val:
                flat = val.replace("/", "__")
                symlink = configs_root / f"{flat}.yaml"
                symlink.symlink_to((configs_root / f"{val}.yaml").resolve())
                if "=" in arg:
                    argv[i] = f"--config-name={flat}"
                else:
                    argv[i + 1] = flat
            break

    return symlink


def run_with_nested_config_name(main, configs_root: Path) -> None:
    """Run a Hydra `main` with nested --config-name support, removing the temporary symlink afterwards."""
    symlink = flatten_nested_config_name(sys.argv, configs_root)

    try:
        main()
    finally:
        if symlink and symlink.exists():
            symlink.unlink()
