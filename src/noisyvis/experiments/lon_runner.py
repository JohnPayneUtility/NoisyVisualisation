"""Local optima network runs behind `run_lon.py`, `run_lon_parallel.py` and `run_colon_parallel.py`.

Moved from the three run scripts in Stage 7 with their bodies unchanged apart from the renames below
and the helpers extracted from code that was identical between workflows:

    run_lon.py              main (body)             -> run_lon_experiment
    run_lon_parallel.py     main (body)             -> run_lon_parallel_experiment
                            _run_single_lon_worker  -> _run_single_lon_worker
                            _merge (closure)        -> _merge_lon (also run_lon.py's inline merge)
    run_colon_parallel.py   main (body)             -> run_colon_experiment
                            _import_from_dotted     -> _import_from_dotted
                            _run_single_lon_worker  -> _run_single_colon_worker
                            _merge (closure)        -> _merge_colon
                            _build_edge_feas_map    -> _build_edge_feas_map
    LON and LON-parallel    compression rows        -> _lon_compression_rows
    all three               metric/persist/artifact -> _log_and_persist_lon_df

`run_lon_experiment` is genuinely sequential: it calls BinaryLON inline and never uses a worker.
LON and CoLON merging stay separate: CoLON keeps first-seen feasibility and accumulates visit counts.

Parallel runs merge results in completion order (`as_completed`), so parallel LON/CoLON output is
nondeterministic by design (risk R27); only the sequential paths are reproducible baselines.

The MLflow tracking URI is set twice, exactly as before: the wrappers call
`tracking.set_lon_module_tracking_uri()` at import, and each workflow sets the config's
`mlflow.tracking_uri` right after resolving the config, before `set_experiment` (R24).
"""

# IMPORTS
import sys
import time
from typing import Any, Dict, List, Tuple, Optional
import importlib

from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import mlflow
import concurrent.futures
import os
import random as _rand

# The fitness-function and attribute-generator names in configs are looked up with
# getattr(sys.modules['noisyvis.problems' | 'noisyvis.algorithms'], name) (risk R5). These imports
# register both keys; the LON scripts used to do it with star imports, in this order.
import noisyvis.problems
import noisyvis.algorithms
from noisyvis.networks import BinaryLON, BinaryCoLON, compress_lon_aggregated

from noisyvis.results.store import save_or_append_results
from noisyvis.results.paths import TEMP_DIR, WAREHOUSE_DIR
from noisyvis.experiments.config.workflows import resolve_colon_config, resolve_lon_config


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
    target_stop: float | int | None,
) -> Tuple[List[Tuple[int, ...]], List[float], List[Tuple[Tuple[int, ...], Tuple[int, ...], float]]]:
    """
    Run one BinaryLON build with a specific seed. Only pass simple/serializable args.
    Returns: (local_optima, fitness_values, edges_list)
    """
    # Seed per-process deterministically
    _rand.seed(seed)
    np.random.seed(seed)

    # Resolve the callables by name (avoid sending function objects)
    fitness_fn = getattr(sys.modules['noisyvis.problems'], fitness_fn_name)
    attr_fn = getattr(sys.modules['noisyvis.algorithms'], attr_fn_name)
    fitness_tuple = (fitness_fn, fit_params)

    local_optima, fitness_values, edges_list = BinaryLON(
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
    return local_optima, fitness_values, edges_list


def _run_single_colon_worker(
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
      neighbour_feasibility, # aligned with local_optima (0..1)
      visit_count            # opt -> total times landed here this run
    """
    # Seed per-process deterministically
    _rand.seed(seed)
    np.random.seed(seed)

    # Resolve the callables
    fitness_fn = getattr(sys.modules['noisyvis.problems'], fitness_fn_name)
    attr_fn = getattr(sys.modules['noisyvis.algorithms'], attr_fn_name)

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
     neighbour_feasibility,
     visit_count) = BinaryCoLON(**lon_kwargs)

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
            optima_feasibility, edge_feas_map, neighbour_feasibility,
            visit_count)


def _merge_lon(aggregated, local_optima, fitness_values, edges_list):
    # Merge optima/fitness
    for opt, fit in zip(local_optima, fitness_values):
        if opt not in aggregated["local_optima"]:
            aggregated["local_optima"].append(opt)
            aggregated["fitness_values"].append(fit)
    # Merge edges
    for (src, dst, w) in edges_list:
        key = (src, dst)
        aggregated["edges"][key] = aggregated["edges"].get(key, 0) + w


def _merge_colon(aggregated, local_optima, fitness_values, edges_dict,
                 optima_feasibility, edge_feas_map, neighbour_feasibility,
                 visit_count):
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

    # Accumulate visit counts across runs (counts repeats within a run)
    for opt, count in visit_count.items():
        aggregated["total_visit_count"][opt] = (
            aggregated["total_visit_count"].get(opt, 0) + count
        )

    # Merge edges: accumulate weights
    for key, w in edges_dict.items():
        aggregated["edges"][key] = aggregated["edges"].get(key, 0) + w
    # We do NOT need to merge per-edge feasibility here; we will derive it
    # for each row from the (possibly compressed) target node feasibility.


# Helper: build edge_feas_map for any edges dict using feasibility of TARGET node
def _build_edge_feas_map(edges_dict: Dict[Tuple[Tuple[int,...], Tuple[int,...]], float],
                         opt_feas_lookup: Dict[Tuple[int,...], int]) -> Dict[Tuple[Tuple[int,...], Tuple[int,...]], int]:
    return { (src, dst): int(opt_feas_lookup.get(dst, 0))
             for (src, dst) in edges_dict.keys() }


def _lon_compression_rows(cfg: DictConfig, prob_info: Dict[str, Any], aggregated: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One results row per compression setting (LON and LON-parallel; CoLON builds its own rows)."""
    # For each compression setting, create a row
    rows: List[Dict[str, Any]] = []
    for comp in cfg.lon.compression_accs:
        if comp == 'None':
            L = aggregated
        else:
            L = compress_lon_aggregated(aggregated, accuracy=float(comp))

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
            "n_local_optima": len(L["local_optima"]),
            "local_optima": L["local_optima"],
            "fitness_values": L["fitness_values"],
            "edges": L["edges"],
        })
    return rows


def _log_and_persist_lon_df(df: pd.DataFrame) -> None:
    """Per-row metric, temp and warehouse pickles, CSV and artifacts (LON, LON-parallel, CoLON)."""
    # Log metric per compression row
    for i, r in enumerate(df.itertuples()):
        mlflow.log_metric("n_local_optima", int(r.n_local_optima), step=i)

    # Artifacts
    out_dir = TEMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_dir / "lon_results.pkl")
    save_or_append_results(df, WAREHOUSE_DIR / 'lon_results.pkl')
    df.to_csv(out_dir / "lon_results.csv", index=False)
    mlflow.log_artifact(str(out_dir / "lon_results.pkl"))
    mlflow.log_artifact(str(out_dir / "lon_results.csv"))
    mlflow.log_artifact("data/outputs/.hydra/config.yaml")


# -------------------------------
# Workflows
# -------------------------------

def run_lon_experiment(cfg: DictConfig):
    start_time = time.perf_counter()

    # Resolve nested deps
    cfg = resolve_lon_config(cfg)

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

    # Fitness & attributes
    fitness_fn = getattr(sys.modules['noisyvis.problems'], cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)
    fitness_tuple = (fitness_fn, fit_params)
    attr_fn = getattr(sys.modules['noisyvis.algorithms'], cfg.problem.attr_function)
    weights = tuple(cfg.problem.weights)

    with mlflow.start_run(run_name=cfg.lon.name):
        print("RUN(LON) artifact root:", mlflow.get_artifact_uri())

        # Log params
        mlflow.log_params({
            "dimensions": cfg.problem.dimensions,
            "seed": cfg.run.seed,
            "num_runs": cfg.run.num_runs,
            "pert_attempts": cfg.lon.pert_attempts,
            "n_flips_mut": cfg.lon.n_flips_mut,
            "n_flips_pert": cfg.lon.n_flips_pert,
            **{f"fit_{k}": v for k, v in fit_params.items()},
        })

        # -------------------------------
        # Build aggregated LON inline (no helper)
        # -------------------------------
        aggregated = {"local_optima": [], "fitness_values": [], "edges": {}}
        for i in range(cfg.run.num_runs):
            seed = cfg.run.seed + i
            import random as _rand
            _rand.seed(seed)
            np.random.seed(seed)

            local_optima, fitness_values, edges_list = BinaryLON(
                pert_attempts=cfg.lon.pert_attempts,
                len_sol=cfg.problem.dimensions,
                weights=weights,
                attr_function=attr_fn,
                n_flips_mut=cfg.lon.n_flips_mut,
                n_flips_pert=cfg.lon.n_flips_pert,
                mutate_function=None,
                perturb_function=None,
                improv_method="best",
                fitness_function=fitness_tuple,
                starting_solution=None,
                true_fitness_function=None,
                target_stop=cfg.problem.opt_global,
            )

            # Merge optima/fitness and edges
            _merge_lon(aggregated, local_optima, fitness_values, edges_list)

        # For each compression setting, create a row
        rows = _lon_compression_rows(cfg, prob_info, aggregated)

        df = pd.DataFrame(rows)

        _log_and_persist_lon_df(df)

    mlflow.end_run(status="FINISHED")

    # Summary
    try:
        print(df[["compression_val", "n_local_optima"]])
    except Exception:
        print(df.head())
    print("Compute time (s):", time.perf_counter() - start_time)


def run_lon_parallel_experiment(cfg: DictConfig):
    start_time = time.perf_counter()

    # Resolve nested deps
    cfg = resolve_lon_config(cfg)

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

    # Parallel controls (new): cfg.run.parallel (bool) and cfg.run.num_workers (int)
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
            **{f"fit_{k}": v for k, v in fit_params.items()},
        })

        # -------------------------------
        # Build aggregated LON (parallel or sequential)
        # -------------------------------
        aggregated = {"local_optima": [], "fitness_values": [], "edges": {}}

        base_seed = cfg.run.seed
        total = cfg.run.num_runs

        if parallel and total > 1:
            # Use processes for CPU-bound work
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
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    loc, fitv, edges = fut.result()
                    _merge_lon(aggregated, loc, fitv, edges)
        else:
            # Sequential fallback
            for i in range(total):
                seed = base_seed + i
                loc, fitv, edges = _run_single_lon_worker(
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
                )
                _merge_lon(aggregated, loc, fitv, edges)

        # For each compression setting, create a row
        rows = _lon_compression_rows(cfg, prob_info, aggregated)

        df = pd.DataFrame(rows)

        _log_and_persist_lon_df(df)

    mlflow.end_run(status="FINISHED")

    # Summary
    try:
        print(df[["compression_val", "n_local_optima"]])
    except Exception:
        print(df.head())
    print("Compute time (s):", time.perf_counter() - start_time)


def run_colon_experiment(cfg: DictConfig):
    start_time = time.perf_counter()
    print(OmegaConf.to_yaml(cfg))  # DEBUG

    # Resolve nested deps
    cfg = resolve_colon_config(cfg)

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
            "total_visit_count": {},   # opt -> total landings across all runs
        }

        base_seed = cfg.run.seed
        total = cfg.run.num_runs
        done = 0

        if parallel and total > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for i in range(total):
                        seed = base_seed + i
                        futures.append(
                            executor.submit(
                                _run_single_colon_worker,
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
                        optf, edgef_map, neighf, vc) = fut.result()
                        _merge_colon(aggregated, loc, fitv, edges_dict, optf, edgef_map, neighf, vc)

                        done += 1
                        print(f"[{done}/{total}] Run completed")
            except KeyboardInterrupt:
                print("Interrupted! Cancelling workers")
                executor.shutdown(cancel_futures=True)
                raise
        else:
            for i in range(total):
                seed = base_seed + i
                (loc, fitv, edges_dict,
                 optf, edgef_map, neighf, vc) = _run_single_colon_worker(
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
                _merge_colon(aggregated, loc, fitv, edges_dict, optf, edgef_map, neighf, vc)

                done += 1
                print(f"[{done}/{total}] Run completed")

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
            visit_counts = [aggregated["total_visit_count"].get(opt, 0) for opt in L_local_optima]
            visit_proportions = [aggregated["total_visit_count"].get(opt, 0) / total for opt in L_local_optima]

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
                "visit_counts": visit_counts,            # per-node total landings across all runs
                "visit_proportions": visit_proportions,  # per-node visit_count / num_runs
                # "edge_feas_map": edge_feas_map,            # (src,dst) -> 1/0 by target feasibility
            })

        df = pd.DataFrame(rows)

        _log_and_persist_lon_df(df)

    mlflow.end_run(status="FINISHED")

    # Summary
    print(df[["compression_val", "n_local_optima"]])
    print("Compute time (s):", time.perf_counter() - start_time)

