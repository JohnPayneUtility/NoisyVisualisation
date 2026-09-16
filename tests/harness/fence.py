"""Write fence for harness subprocesses.

The entry-point scripts write the 1.27 GB warehouse pickle, `lon_results.pkl` and the MLflow store.
A Stage 1 test run must never touch any of that. The temp root redirects those writes, and this
module makes a missed redirection fail *before* anything is written, rather than being detected
afterwards by the session guard.

It installs an audit hook that raises `FenceViolation` on any write-intent operation whose target
resolves under the protected root (`/workspace`). Reads are unaffected, and the hook is inherited by
forked worker processes, which is how the parallel SO baseline is covered.

A correct run never trips the fence.
"""

from __future__ import annotations

import os
import sys
import threading

DEFAULT_PROTECTED_ROOT = "/workspace"

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

# event name -> indices of arguments that are paths to check
_PATH_EVENTS: dict[str, tuple[int, ...]] = {
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.unlink": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.link": (1,),
    "os.symlink": (1,),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.utime": (0,),
    "shutil.rmtree": (0,),
    "shutil.copyfile": (1,),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
    "shutil.move": (1,),
}

_state = threading.local()


class FenceViolation(PermissionError):
    """A harness subprocess attempted to write inside the protected root."""


def _wants_write(mode: object, flags: object) -> bool:
    """True if an `open` audit event describes a write."""
    if isinstance(mode, str):
        return any(ch in mode for ch in "wxa+")
    if isinstance(flags, int):
        return bool(flags & _WRITE_FLAGS)
    # Unknown shape: treat as a write so the fence fails closed.
    return True


def _resolve(path: object) -> str | None:
    """Absolute, symlink-resolved path, or None when there is nothing to check."""
    if isinstance(path, int) or path is None:
        # An already-open file descriptor: it was checked when it was opened.
        return None
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, str):
        path = str(path)
    if not path:
        return None
    # realpath() uses readlink/lstat, neither of which raises audit events,
    # so this cannot recurse back into the hook.
    return os.path.realpath(path)


def install(protected_root: str = DEFAULT_PROTECTED_ROOT) -> None:
    """Install the audit hook. Cannot be uninstalled: that is a CPython guarantee."""
    root = os.path.realpath(protected_root)
    root_prefix = root.rstrip("/") + "/"

    def _is_protected(path: object) -> str | None:
        resolved = _resolve(path)
        if resolved is None:
            return None
        if resolved == root or resolved.startswith(root_prefix):
            return resolved
        return None

    def _hook(event: str, args: tuple) -> None:
        # Guard against re-entry: raising builds a message, which may itself
        # trigger audited operations in some interpreters.
        if getattr(_state, "busy", False):
            return
        if event == "open":
            path = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if not _wants_write(mode, flags):
                return
            indices: tuple[int, ...] = (0,)
        else:
            indices = _PATH_EVENTS.get(event, ())
            if not indices:
                return

        _state.busy = True
        try:
            for index in indices:
                if index >= len(args):
                    continue
                offending = _is_protected(args[index])
                if offending is not None:
                    raise FenceViolation(
                        f"harness subprocess tried to write inside the protected root: "
                        f"{event} -> {offending!r}. Production data must never be touched by "
                        f"a test run; the run should have written into its temp root instead."
                    )
        finally:
            _state.busy = False

    sys.addaudithook(_hook)
