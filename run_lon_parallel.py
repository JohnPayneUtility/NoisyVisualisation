# `run_lon.py`

# IMPORTS
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import mlflow
import concurrent.futures
import os
import random as _rand

from noisyvis.problems import *
from noisyvis.algorithms import *
from noisyvis.results.store import save_or_append_results
from noisyvis.results.paths import MLRUNS_DIR, TEMP_DIR, WAREHOUSE_DIR
from noisyvis.experiments.config.workflows import resolve_lon_config

# -------------------------------
# MLflow defaults (local file store under repo/data/mlruns)
# -------------------------------
mlruns_dir = MLRUNS_DIR  # <project root>/data/mlruns (results.paths)
mlflow.set_tracking_uri(f"file:{mlruns_dir}")
print("RUN(LON) tracking:", mlflow.get_tracking_uri())

# -------------------------------
# Helpers
# -------------------------------

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


# -------------------------------
# Main
# -------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="test_lon_kp")
def main(cfg: DictConfig):
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

        def _merge(local_optima, fitness_values, edges_list):
            # Merge optima/fitness
            for opt, fit in zip(local_optima, fitness_values):
                if opt not in aggregated["local_optima"]:
                    aggregated["local_optima"].append(opt)
                    aggregated["fitness_values"].append(fit)
            # Merge edges
            for (src, dst, w) in edges_list:
                key = (src, dst)
                aggregated["edges"][key] = aggregated["edges"].get(key, 0) + w

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
                    _merge(loc, fitv, edges)
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
                _merge(loc, fitv, edges)

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

        df = pd.DataFrame(rows)

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

    mlflow.end_run(status="FINISHED")

    # Summary
    try:
        print(df[["compression_val", "n_local_optima"]])
    except Exception:
        print(df.head())
    print("Compute time (s):", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
