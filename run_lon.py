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

# Your modules
from src.problems import *  # fitness fns & loaders
from src.algorithms import *  # attribute generators (e.g., binary_attribute)
# from src.LONs import BinaryLON, compress_lon_aggregated

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

def resolve_config_dependencies(cfg: DictConfig) -> DictConfig:
    """Resolve loader-driven problem details and fitness params (like run.py)."""
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Load problem if a loader is specified (e.g., knapsack instance)
    if "loader" in resolved.problem and resolved.problem.loader is not None:
        outputs = call(resolved.problem.loader)
        n_items, capacity, optimal, values, weights, items_dict, _ = outputs

        items_dict = {int(k): (float(v[0]), float(v[1])) for k, v in items_dict.items()}

        # Attach to cfg
        resolved.problem.dimensions = int(n_items)
        resolved.problem.opt_global = float(optimal)
        resolved.problem.capacity = float(capacity)
        resolved.problem.mean_value = float(np.mean(values))
        resolved.problem.mean_weight = float(np.mean(weights))
        resolved.problem.items_dict = items_dict

        # Fitness params
        if hasattr(resolved.problem, "fitness_params"):
            resolved.problem.fitness_params.items_dict = items_dict
            resolved.problem.fitness_params.capacity = float(capacity)
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
        "PID": f"{cfg.problem.prob_name}_{cfg.problem.dimensions}",
    }

    # Fitness & attributes
    fitness_fn = getattr(sys.modules['src.problems'], cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)
    fitness_tuple = (fitness_fn, fit_params)
    attr_fn = getattr(sys.modules['src.algorithms'], cfg.problem.attr_function)
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

            # Merge optima/fitness
            for opt, fit in zip(local_optima, fitness_values):
                if opt not in aggregated["local_optima"]:
                    aggregated["local_optima"].append(opt)
                    aggregated["fitness_values"].append(fit)

            # Merge edges
            for (src, dst, w) in edges_list:
                key = (src, dst)
                aggregated["edges"][key] = aggregated["edges"].get(key, 0) + w

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
        out_dir = Path("data/temp")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out_dir / "lon_results.pkl")
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