"""Canonical encoding and comparison of scientific results.

Baselines are compared *exactly*: no tolerances. Two runs of the same seeded code must produce
identical numbers, so any difference is a real behavioural change and must be investigated rather
than absorbed by a tolerance.

What this module normalises is only representation, never value:

* numpy scalars and arrays become plain Python values;
* tuples become lists, because JSON has no tuple;
* dicts keyed by tuples (LON optima, edges) become sorted key/value pair lists, because JSON object
  keys must be strings and because the maps are compared as sets of entries;
* set-like sequences can be sorted explicitly by the caller via `sort_canonical`.

Ordering that carries meaning -- hypervolume time series, the per-generation Pareto history -- is
never sorted. Floats round-trip exactly through `repr`, which is what `json` uses.
"""

from __future__ import annotations

import json
import math
from typing import Any

SCHEMA = 1


def jsonable(obj: Any) -> Any:
    """Convert to JSON-representable form without changing any value."""
    # numpy is optional at import time so this module stays usable anywhere.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is always present in `exp`
        np = None

    if np is not None:
        if isinstance(obj, np.generic):
            return jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return [jsonable(x) for x in obj.tolist()]

    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, dict):
        # Tuple/other non-string keys cannot be JSON object keys; represent the
        # whole mapping as sorted pairs so comparison is order-insensitive.
        if all(isinstance(k, str) for k in obj):
            return {k: jsonable(v) for k, v in obj.items()}
        return map_to_pairs(obj)
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sort_canonical(obj)
    # pandas NA / NaT and anything else unexpected: fail loudly rather than
    # silently stringifying a value that is meant to be compared.
    raise TypeError(f"cannot canonicalise {type(obj)!r}: {obj!r}")


def _sort_key(value: Any) -> str:
    """Total, stable ordering over canonical values."""
    return json.dumps(value, sort_keys=True, allow_nan=True)


def map_to_pairs(mapping: dict) -> list:
    """A dict with non-string keys as a list of [key, value], sorted by key."""
    pairs = [[jsonable(k), jsonable(v)] for k, v in mapping.items()]
    pairs.sort(key=lambda kv: _sort_key(kv[0]))
    return pairs


def sort_canonical(seq) -> list:
    """Sort a set-like sequence into a canonical order."""
    items = [jsonable(x) for x in seq]
    items.sort(key=_sort_key)
    return items


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=True)


def first_difference(expected: Any, actual: Any, path: str = "") -> str | None:
    """Human-readable description of the first difference, or None if equal.

    Walks both structures in parallel so the report points at the exact field, seed or
    compression level that changed, instead of dumping two large blobs.
    """
    here = path or "<root>"

    if type(expected) is not type(actual):
        # int/float distinction matters (n_evals must not silently become a float),
        # but bool is a subclass of int and is compared by value below.
        if not (isinstance(expected, (int, float)) and isinstance(actual, (int, float))):
            return f"{here}: type changed, {type(expected).__name__} -> {type(actual).__name__}"

    if isinstance(expected, dict):
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        if missing:
            return f"{here}: key(s) missing from result: {missing}"
        if added:
            return f"{here}: unexpected key(s) in result: {added}"
        for key in expected:
            diff = first_difference(expected[key], actual[key], f"{path}.{key}" if path else str(key))
            if diff:
                return diff
        return None

    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{here}: length changed, {len(expected)} -> {len(actual)}"
        for index, (exp_item, act_item) in enumerate(zip(expected, actual)):
            diff = first_difference(exp_item, act_item, f"{path}[{index}]")
            if diff:
                return diff
        return None

    if isinstance(expected, float) and isinstance(actual, float):
        if math.isnan(expected) and math.isnan(actual):
            return None
        if expected != actual:
            return f"{here}: {expected!r} != {actual!r}"
        return None

    if expected != actual:
        return f"{here}: {expected!r} != {actual!r}"
    return None
