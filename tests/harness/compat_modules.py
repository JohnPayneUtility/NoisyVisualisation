"""Which compatibility modules a harness child has loaded (plan §8 Stage 12).

Stage 12 moves every config dotted path off the temporary compatibility modules (`src/src/`'s
`src.algorithms.*` / `src.problems.*` forwarders and the root `run_helpers.py`) onto the canonical
`noisyvis.*` modules. The resolution children report this after resolving, so the gate proves the
canonical inputs never load the bridge, even while it still exists on disk.

Matched by namespace, not by a fixed list: `run_helpers` or anything under it, and any `src.*`
submodule. The bare top-level `src` is not reported: with `/workspace` on `sys.path` it is merely a
namespace package and forwards to nothing.

Pure: it only reads `sys.modules`.
"""

from __future__ import annotations

import sys


def compat_modules_loaded() -> list:
    return sorted(
        name
        for name in list(sys.modules)
        if name == "run_helpers" or name.startswith("run_helpers.") or name.startswith("src.")
    )
