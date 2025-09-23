# IMPORTS
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import importlib

import hydra
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import mlflow
import concurrent.futures
import os
import random as _rand

from src.problems import *    # fitness functions resolved by name
from src.algorithms import *  # BinaryCoLON, compress_lon_aggregated, attribute gens, etc.
from src.io.ExperimentsHelpers import save_or_append_results

# -------------------------------
# MLflow defaults (local file store under repo/data/mlruns)
# -------------------------------
base = Path(__file__).resolve().parents[1]  # project root
mlruns_dir = base / "data" / "mlruns"
mlflow.set_tracking_uri(f"file:{mlruns_dir}")
print("RUN(LON) tracking:", mlflow.get_tracking_uri())

# -------------------------------
# Helpers
# -------------------------------

def _import_from_dotted(dotted: str):
    """
    Import a callable from a fully qualified dotted path. Example:
    'src.problems.ViolationFunctions.knap_violation'
    """
    module_path, attr = dotted.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)

def resolve_config_dependencies(cfg: DictConfig) -> DictConfig:
    """Resolve loader-driven problem details and fitness/violation params (like run.py)."""
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Load problem if a loader is specified (e.g., knapsack instance)
    if "loader" in resolved.problem and resolved.problem.loader is not None:
        outputs = call(resolved.problem.loader)
        n_items, capacity, optimal, values, weights, items_dict, _ = outputs

        # normalise types
        items_dict = {int(k): (float(v[0]), float(v[1])) for k, v in items_dict.items()}

        # Attach to cfg
        resolved.problem.dimensions = int(n_items)
        resolved.problem.opt_global = float(optimal)
        resolved.problem.capacity = float(capacity)
        resolved.problem.mean_value = float(np.mean(values))
        resolved.problem.mean_weight = float(np.mean(weights))
        resolved.problem.items_dict = items_dict

        # Fitness params
        if hasattr(resolved.problem, "fitness_params") and resolved.problem.fitness_params is not None:
            resolved.problem.fitness_params.items_dict = items_dict
            resolved.problem.fitness_params.capacity = float(capacity)

        # Violation params (optional)
        if hasattr(resolved.problem, "violation_params") and resolved.problem.violation_params is not None:
            resolved.problem.violation_params.items_dict = items_dict
            resolved.problem.violation_params.capacity = float(capacity)
    else:
        # OneMax-style problems (no loader)
        if not getattr(resolved.problem, "capacity", None):
            resolved.problem.capacity = 0
        if not getattr(resolved.problem, "mean_value", None):
            resolved.problem.mean_value = 50.0
        if not getattr(resolved.problem, "mean_weight", None):
            resolved.problem.mean_weight = 0.5

    return resolved


# -------------------------------
# Worker wrapper (for parallel runs)
# -------------------------------

def _run_single_lon_worker(
    seed: int,
    len_sol: int,
    weights: Tuple[float, ...],
    attr_fn_name: str,
    n_flips_mut: int,
    n_flips_pert: int,
    pert_attempts: int,
    fitness_fn_name: str,
    fit_params: Dict[str, Any],
    target_stop: Optional[float],
    violation_fn_dotted: Optional[str],          # OPTIONAL fully-qualified dotted path
    viol_params: Optional[Dict[str, Any]],       # OPTIONAL params dict
) -> Tuple[
    List[Tuple[int, ...]],
    List[float],
    Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
    List[int],
    Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int],
    List[float]
]:
    """
    Run one BinaryCoLON build with a specific seed. Only pass simple/serializable args.

    Returns:
      local_optima,
      fitness_values,
      edges_dict,            # (src,dst) -> weight
      optima_feasibility,    # aligned with local_optima (1/0)
      edge_feas_map,         # (src,dst) -> 1/0 based on dst feasibility
      neighbour_feasibility  # aligned with local_optima (0..1)
    """
    # Seed per-process deterministically
    _rand.seed(seed)
    np.random.seed(seed)

    # Resolve the callables
    fitness_fn = getattr(sys.modules['src.problems'], fitness_fn_name)
    attr_fn = getattr(sys.modules['src.algorithms'], attr_fn_name)

    fitness_tuple = (fitness_fn, fit_params)

    # Build kwargs for BinaryCoLON call
    lon_kwargs: Dict[str, Any] = dict(
        pert_attempts=pert_attempts,
        len_sol=len_sol,
        weights=weights,
        attr_function=attr_fn,
        n_flips_mut=n_flips_mut,
        n_flips_pert=n_flips_pert,
        mutate_function=None,
        perturb_function=None,
        improv_method="best",
        fitness_function=fitness_tuple,
        starting_solution=None,
        true_fitness_function=None,
        target_stop=target_stop,
    )

    # If a violation function is provided (dotted path), import and attach it.
    if violation_fn_dotted:
        violation_fn = _import_from_dotted(violation_fn_dotted)
        violation_tuple = (violation_fn, viol_params or {})
        lon_kwargs["violation_function"] = violation_tuple  # CoLON uses Deb’s preorder

    # Call BinaryCoLON (returns 6 values; first 3 are list,list,list in your function,
    # here we convert edges_list->dict and build edge_feas_map right here for convenience)
    (local_optima,
     fitness_values,
     edges_list,
     optima_feasibility,
     edge_feasibility,
     neighbour_feasibility) = BinaryCoLON(**lon_kwargs)

    # Convert edges_list -> dict for consistent aggregation
    edges_dict: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    for (src, dst, w) in edges_list:
        edges_dict[(src, dst)] = edges_dict.get((src, dst), 0) + w

    # Build edge_feas_map aligned to the dict keys (dst feasibility from edge_feasibility list)
    # We rely on the same order used above; safe because we create the dict from edges_list here.
    edge_feas_map: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
    for (src, dst, _w), ef in zip(edges_list, edge_feasibility):
        edge_feas_map[(src, dst)] = int(ef)

    return (local_optima, fitness_values, edges_dict,
            optima_feasibility, edge_feas_map, neighbour_feasibility)


# -------------------------------
# Main
# -------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="test_lon_kp")
def main(cfg: DictConfig):
    start_time = time.perf_counter()

    # Resolve nested deps
    cfg = resolve_config_dependencies(cfg)

    # MLflow init
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Problem metadata
    prob_info = {
        "name": cfg.problem.prob_name,
        "type": cfg.problem.prob_type,
        "goal": cfg.problem.opt_goal,
        "dimensions": cfg.problem.dimensions,
        "opt_global": cfg.problem.opt_global,
        "mean_value": cfg.problem.mean_value,
        "mean_weight": cfg.problem.mean_weight,
        "PID": cfg.problem.PID,
    }

    # Fitness & attributes (names for worker resolution)
    fitness_fn_name = cfg.problem.fitness_fn
    fit_params = dict(cfg.problem.fitness_params)
    attr_fn_name = cfg.problem.attr_function
    weights = tuple(cfg.problem.weights)

    # OPTIONAL violation function (fully qualified dotted path)
    violation_fn_dotted = getattr(cfg.problem, "violation_fn", None)
    viol_params = dict(getattr(cfg.problem, "violation_params", {}) or {})

    # Parallel controls
    parallel = bool(getattr(cfg.run, "parallel", False))
    num_workers = int(getattr(cfg.run, "num_workers", os.cpu_count() or 1))

    with mlflow.start_run(run_name=cfg.lon.name):
        print("RUN(LON) artifact root:", mlflow.get_artifact_uri())

        # Log params
        mlflow.log_params({
            "dimensions": cfg.problem.dimensions,
            "seed": cfg.run.seed,
            "num_runs": cfg.run.num_runs,
            "parallel": parallel,
            "num_workers": num_workers,
            "pert_attempts": cfg.lon.pert_attempts,
            "n_flips_mut": cfg.lon.n_flips_mut,
            "n_flips_pert": cfg.lon.n_flips_pert,
            "lon_constrained": bool(violation_fn_dotted),
            "violation_fn": violation_fn_dotted or "None",
            **{f"fit_{k}": v for k, v in fit_params.items()},
            **({f"viol_{k}": v for k, v in viol_params.items()} if violation_fn_dotted else {}),
        })

        # -------------------------------
        # Build aggregated LON (parallel or sequential)
        # -------------------------------
        aggregated = {
            "local_optima": [],        # insertion-ordered unique optima
            "fitness_values": [],      # aligned with local_optima
            "edges": {},               # (src,dst) -> weight

            # NEW lookup caches
            "opt_index": {},           # opt -> index
            "opt_feas_map": {},        # opt -> 1/0
            "neigh_feas_map": {},      # opt -> float (0..1)
        }

        def _merge(local_optima, fitness_values, edges_dict,
                   optima_feasibility, edge_feas_map, neighbour_feasibility):
            # Merge optima/fitness + per-optimum feasibility aligned by index
            for opt, fit, of, nf in zip(local_optima, fitness_values,
                                        optima_feasibility, neighbour_feasibility):
                if opt not in aggregated["opt_index"]:
                    idx = len(aggregated["local_optima"])
                    aggregated["opt_index"][opt] = idx
                    aggregated["local_optima"].append(opt)
                    aggregated["fitness_values"].append(fit)
                    aggregated["opt_feas_map"][opt] = int(of)
                    aggregated["neigh_feas_map"][opt] = float(nf)
                else:
                    # Deterministic problems -> should match; keep first-seen.
                    pass

            # Merge edges: accumulate weights
            for key, w in edges_dict.items():
                aggregated["edges"][key] = aggregated["edges"].get(key, 0) + w
            # We do NOT need to merge per-edge feasibility here; we will derive it
            # for each row from the (possibly compressed) target node feasibility.

        base_seed = cfg.run.seed
        total = cfg.run.num_runs

        if parallel and total > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(total):
                    seed = base_seed + i
                    futures.append(
                        executor.submit(
                            _run_single_lon_worker,
                            seed,
                            cfg.problem.dimensions,
                            weights,
                            attr_fn_name,
                            cfg.lon.n_flips_mut,
                            cfg.lon.n_flips_pert,
                            cfg.lon.pert_attempts,
                            fitness_fn_name,
                            fit_params,
                            cfg.problem.opt_global,
                            violation_fn_dotted,  # OPTIONAL
                            viol_params,          # OPTIONAL
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    (loc, fitv, edges_dict,
                     optf, edgef_map, neighf) = fut.result()
                    # edgef_map is not needed during aggregation; we ignore it here
                    _merge(loc, fitv, edges_dict, optf, edgef_map, neighf)
        else:
            for i in range(total):
                seed = base_seed + i
                (loc, fitv, edges_dict,
                 optf, edgef_map, neighf) = _run_single_lon_worker(
                    seed,
                    cfg.problem.dimensions,
                    weights,
                    attr_fn_name,
                    cfg.lon.n_flips_mut,
                    cfg.lon.n_flips_pert,
                    cfg.lon.pert_attempts,
                    fitness_fn_name,
                    fit_params,
                    cfg.problem.opt_global,
                    violation_fn_dotted,  # OPTIONAL
                    viol_params,          # OPTIONAL
                )
                _merge(loc, fitv, edges_dict, optf, edgef_map, neighf)

        # Helper: build edge_feas_map for any edges dict using feasibility of TARGET node
        def _build_edge_feas_map(edges_dict: Dict[Tuple[Tuple[int,...], Tuple[int,...]], float],
                                 opt_feas_lookup: Dict[Tuple[int,...], int]) -> Dict[Tuple[Tuple[int,...], Tuple[int,...]], int]:
            return { (src, dst): int(opt_feas_lookup.get(dst, 0))
                     for (src, dst) in edges_dict.keys() }

        # -------------------------------
        # Rows per compression setting
        # -------------------------------
        rows: List[Dict[str, Any]] = []
        for comp in cfg.lon.compression_accs:
            if comp == 'None':
                # Use aggregated directly
                L_local_optima = aggregated["local_optima"]
                L_fitness_values = aggregated["fitness_values"]
                L_edges = aggregated["edges"]
                # lookups
                opt_feas_lookup = aggregated["opt_feas_map"]
                neigh_feas_lookup = aggregated["neigh_feas_map"]
            else:
                # Compressed LON
                L = compress_lon_aggregated(aggregated, accuracy=float(comp))
                L_local_optima = L["local_optima"]
                L_fitness_values = L["fitness_values"]
                L_edges = L["edges"]
                # lookups from aggregated (exact tuple keys)
                opt_feas_lookup = {opt: aggregated["opt_feas_map"].get(opt, 0) for opt in L_local_optima}
                neigh_feas_lookup = {opt: aggregated["neigh_feas_map"].get(opt, 0.0) for opt in L_local_optima}

            # per-optimum lists aligned with local_optima
            optima_feasibility = [int(opt_feas_lookup[opt]) for opt in L_local_optima]
            neighbour_feasibility = [float(neigh_feas_lookup[opt]) for opt in L_local_optima]

            # per-edge feasibility dict (same keys as L_edges)
            edge_feas_map = _build_edge_feas_map(L_edges, opt_feas_lookup)

            rows.append({
                "problem_name": prob_info["name"],
                "problem_type": prob_info["type"],
                "problem_goal": prob_info["goal"],
                "dimensions": prob_info["dimensions"],
                "opt_global": prob_info["opt_global"],
                "PID": prob_info["PID"],
                "LON_Algo": cfg.lon.name,
                "n_flips_mut": cfg.lon.n_flips_mut,
                "n_flips_pert": cfg.lon.n_flips_pert,
                "compression_val": comp,
                "n_local_optima": len(L_local_optima),
                "local_optima": L_local_optima,
                "fitness_values": L_fitness_values,
                "edges": L_edges,                          # (src,dst) -> weight
                "optima_feasibility": optima_feasibility,  # per-node (aligned)
                "neighbour_feasibility": neighbour_feasibility,  # per-node (aligned)
                # "edge_feas_map": edge_feas_map,            # (src,dst) -> 1/0 by target feasibility
            })

        df = pd.DataFrame(rows)

        # Log metric per compression row
        for i, r in enumerate(df.itertuples()):
            mlflow.log_metric("n_local_optima", int(r.n_local_optima), step=i)

        # Artifacts
        out_dir = Path("data/temp")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out_dir / "lon_results.pkl")
        save_or_append_results(df, 'data/dashboard_dw/lon_results.pkl')
        df.to_csv(out_dir / "lon_results.csv", index=False)
        mlflow.log_artifact(str(out_dir / "lon_results.pkl"))
        mlflow.log_artifact(str(out_dir / "lon_results.csv"))
        mlflow.log_artifact("data/outputs/.hydra/config.yaml")

    mlflow.end_run(status="FINISHED")

    # Summary
    try:
        print(df[["compression_val", "n_local_optima"]])
    except Exception:
        print(df.head())
    print("Compute time (s):", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
