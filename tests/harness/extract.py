"""Pull the compared scientific fields out of a finished isolated run (plan §5.5).

This runs *inside* the harness child process, after the entry point has returned, for two reasons:

* MO results contain DEAP `creator.Individual` objects, which only unpickle in a process where
  `creator.create` has been called -- that is, the process that just ran the experiment;
* it keeps the pytest process free of any dependency on the science packages.

Output is plain JSON-safe data. Ordering that carries meaning (hypervolume time series, the
per-generation Pareto history) is preserved; set-like structures (LON optima and edge maps, Pareto
fronts within one snapshot) are sorted into a canonical order so that irrelevant ordering cannot
cause a false failure. The small sorting helpers are local rather than imported from
`canonical.py`, because this module is loaded by file path before sys.path is rewritten.

Fields deliberately excluded as non-scientific or environment-dependent:
`peak_ram_mb`, `run_id`, `parent_run_id`, `payload_path`.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

SCHEMA = 1

SO_SCALARS = [
    "n_evals",
    "n_gens",
    "stop_trigger",
    "seed_signature",
    "final_fit",
    "max_fit",
    "min_fit",
    "n_unique_sols",
]
SO_PAYLOAD_FIELDS = ["rep_sols", "rep_true_fits", "rep_noisy_fits"]

MO_SCALARS = [
    "n_gens",
    "n_evals",
    "stop_trigger",
    "seed_signature",
    "final_true_hv",
    "max_true_hv",
    "min_true_hv",
    "final_noisy_pf_hv",
    "max_noisy_pf_hv",
    "min_noisy_pf_hv",
]
MO_SERIES = [
    "noisy_pf_noisy_hypervolumes",
    "noisy_pf_true_hypervolumes",
    "true_pf_hypervolumes",
    "n_gens_pareto_best",
]
COLON_NODE_FIELDS = [
    "optima_feasibility",
    "neighbour_feasibility",
    "visit_counts",
    "visit_proportions",
]


def _py(obj):
    """Plain-Python form of a value, without changing it."""
    import numpy as np

    if isinstance(obj, np.generic):
        return _py(obj.item())
    if isinstance(obj, np.ndarray):
        return [_py(x) for x in obj.tolist()]
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {_key(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_py(x) for x in obj]
    # DEAP individuals are list subclasses and are handled above; anything else is
    # unexpected and should fail loudly rather than be stringified.
    raise TypeError(f"cannot serialise {type(obj)!r} for comparison")


def _key(value):
    return value if isinstance(value, str) else json.dumps(_py(value), sort_keys=True)


def _sort_key(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=True)


def _pairs(mapping: dict) -> list:
    """A mapping with non-string keys as a canonical, sorted list of [key, value]."""
    items = [[_py(k), _py(v)] for k, v in mapping.items()]
    items.sort(key=lambda kv: _sort_key(kv[0]))
    return items


def _sorted_items(seq) -> list:
    items = [_py(x) for x in seq]
    items.sort(key=_sort_key)
    return items


def _aligned_map(keys, values, label: str) -> dict:
    """Map aligned list-of-keys/list-of-values, refusing to hide a duplicate key."""
    if len(keys) != len(values):
        raise ValueError(f"{label}: length mismatch, {len(keys)} keys vs {len(values)} values")
    result = {}
    for key, value in zip(keys, values):
        canonical_key = tuple(_py(key))
        if canonical_key in result:
            raise ValueError(f"{label}: duplicate key {canonical_key!r}; aggregation is not unique")
        result[canonical_key] = value
    return result


def _read_pickle(path: Path):
    import pandas as pd

    return pd.read_pickle(path)


def _resolved_config(root: Path) -> dict:
    """The config Hydra actually used, which is how sequential execution is proved."""
    import yaml

    config_path = root / "data" / "outputs" / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"no resolved Hydra config at {config_path}")
    return yaml.safe_load(config_path.read_text())


def _meta(root: Path, warehouse_name: str) -> dict:
    """Isolation and mode evidence recorded alongside every baseline."""
    config = _resolved_config(root)
    warehouse = root / "data" / "dashboard_dw" / warehouse_name
    warehouse_rows = None
    if warehouse.is_file():
        warehouse_rows = int(len(_read_pickle(warehouse)))

    mlruns = root / "mlruns"
    mlruns_files = sum(1 for path in mlruns.rglob("*") if path.is_file()) if mlruns.is_dir() else 0

    return {
        "run_parallel": config.get("run", {}).get("parallel"),
        "run_num_runs": config.get("run", {}).get("num_runs"),
        "warehouse_rows": warehouse_rows,
        "mlruns_files": mlruns_files,
    }


def _extract_so(root: Path) -> dict:
    df = _read_pickle(root / "data" / "temp" / "results.pkl")
    cases = {}
    for row in df.to_dict(orient="records"):
        seed = str(int(row["seed"]))
        case = {field: _py(row[field]) for field in SO_SCALARS}

        payload_path = Path(row["payload_path"])
        if not payload_path.is_absolute():
            payload_path = root / payload_path
        with open(payload_path, "rb") as handle:
            payload = pickle.load(handle)
        for field in SO_PAYLOAD_FIELDS:
            case[field] = _py(payload[field])

        if seed in cases:
            raise ValueError(f"duplicate seed {seed} in results; cannot key by seed")
        cases[seed] = case

    return {"schema": SCHEMA, "meta": _meta(root, "algo_results.pkl"), "cases": cases}


def _front_snapshots(solutions, noisy_fits, true_fits) -> list:
    """Per-generation history, each snapshot canonically sorted.

    The outer list is generation order and is preserved. Within a snapshot the front is a set,
    so entries are sorted; solutions and their fitnesses are kept paired when aligned, which is
    strictly stronger than sorting each independently.
    """
    snapshots = []
    for index, front in enumerate(solutions):
        noisy = noisy_fits[index] if index < len(noisy_fits) else None
        true = true_fits[index] if index < len(true_fits) else None
        if noisy is not None and true is not None and len(front) == len(noisy) == len(true):
            entries = [
                [_py(sol), _py(nf), _py(tf)] for sol, nf, tf in zip(front, noisy, true)
            ]
            entries.sort(key=_sort_key)
            snapshots.append({"paired": True, "entries": entries})
        else:
            snapshots.append(
                {
                    "paired": False,
                    "solutions": _sorted_items(front),
                    "noisy_fitnesses": _sorted_items(noisy or []),
                    "true_fitnesses": _sorted_items(true or []),
                }
            )
    return snapshots


def _extract_mo(root: Path) -> dict:
    df = _read_pickle(root / "data" / "temp" / "results.pkl")
    cases = {}
    for row in df.to_dict(orient="records"):
        seed = str(int(row["seed"]))
        case = {field: _py(row[field]) for field in MO_SCALARS}
        for field in MO_SERIES:
            case[field] = _py(row[field])

        case["noisy_pareto_history"] = _front_snapshots(
            row["pareto_solutions"], row["pareto_fitnesses"], row["pareto_true_fitnesses"]
        )
        case["true_pareto_history"] = _front_snapshots(
            row["true_pareto_solutions"], row["true_pareto_fitnesses"], row["true_pareto_fitnesses"]
        )

        if seed in cases:
            raise ValueError(f"duplicate seed {seed} in MO results; cannot key by seed")
        cases[seed] = case

    return {"schema": SCHEMA, "meta": _meta(root, "algo_results.pkl"), "cases": cases}


def _lon_case(row, with_feasibility: bool) -> dict:
    optima = list(row["local_optima"])
    case = {
        "n_local_optima": _py(row["n_local_optima"]),
        "optimum_fitness": _pairs(_aligned_map(optima, row["fitness_values"], "fitness_values")),
        "edges": _pairs(dict(row["edges"])),
    }
    if with_feasibility:
        for field in COLON_NODE_FIELDS:
            case[field] = _pairs(_aligned_map(optima, row[field], field))
    return case


def _extract_lon(root: Path, with_feasibility: bool) -> dict:
    df = _read_pickle(root / "data" / "temp" / "lon_results.pkl")
    cases = {}
    for row in df.to_dict(orient="records"):
        level = str(row["compression_val"])
        if level in cases:
            raise ValueError(f"duplicate compression level {level!r} in LON results")
        cases[level] = _lon_case(row, with_feasibility)

    return {"schema": SCHEMA, "meta": _meta(root, "lon_results.pkl"), "cases": cases}


def extract(kind: str, root: Path) -> dict:
    root = Path(root)
    if kind == "so":
        return _extract_so(root)
    if kind == "mo":
        return _extract_mo(root)
    if kind == "lon":
        return _extract_lon(root, with_feasibility=False)
    if kind == "colon":
        return _extract_lon(root, with_feasibility=True)
    raise ValueError(f"unknown extractor kind: {kind!r}")
