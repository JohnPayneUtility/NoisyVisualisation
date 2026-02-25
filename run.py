
# IMPORTS
import sys
import time
import random
import pickle
import pandas as pd
import hydra
from hydra.utils import instantiate, call
from omegaconf import OmegaConf, DictConfig
import mlflow
from src.algorithms import *
from src.problems import *
import random
import numpy as np
import pandas as pd
import concurrent.futures
import itertools
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Type
from deap import tools

from src.io.ExperimentsHelpers import save_or_append_results
from src.algorithms.Logger import clear_active_logger
from run_helpers import *


# explicit mlflow path
from pathlib import Path
# base = Path(__file__).resolve().parents[1]  # project root
# mlruns_dir = base / "data" / "mlruns"
# mlflow.set_tracking_uri(f"file:{mlruns_dir}")

base = Path(__file__).resolve().parents[0]          # folder containing run.py
project_root = base                                 # if run.py is in root
mlruns_dir = project_root / "data" / "mlruns"
mlruns_dir.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")

print("RUN tracking:", mlflow.get_tracking_uri())
# -------------------------------
# Helper Functions for Dependency Resolution
# -------------------------------

def resolve_config_dependencies(cfg: DictConfig) -> DictConfig:
    """
    Resolve dependencies in a nested config structure.
    This keeps the nested structure but resolves dynamic values.
    """
    
    # Create a copy to avoid modifying the original
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Step 1: Load problem data if needed
    if "loader" in resolved_cfg.problem and resolved_cfg.problem.loader is not None:
        # Load problem-specific data (e.g., for Knapsack)
        outputs = call(resolved_cfg.problem.loader)
        n_items, capacity, optimal, values, weights, items_dict, _ = outputs
        
        # Convert items_dict to proper format
        items_dict = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in items_dict.items()
        }
        
        # Set problem metadata
        resolved_cfg.problem.dimensions = int(n_items)
        resolved_cfg.problem.opt_global = float(optimal)
        resolved_cfg.problem.capacity = float(capacity)
        resolved_cfg.problem.mean_value = float(np.mean(values))
        resolved_cfg.problem.mean_weight = float(np.mean(weights))
        resolved_cfg.problem.items_dict = items_dict
        
        # Set fitness parameters that depend on problem loading
        if hasattr(resolved_cfg.problem, 'fitness_params'):
            resolved_cfg.problem.fitness_params.items_dict = items_dict
            resolved_cfg.problem.fitness_params.capacity = float(capacity)
    else:
        # For problems like OneMax that don't need a loader
        # Ensure dimensions and opt_global are set if they exist in config
        if hasattr(resolved_cfg.problem, 'dimensions') and resolved_cfg.problem.dimensions is not None:
            # Dimensions are already set in config, no need to load
            pass
        if hasattr(resolved_cfg.problem, 'opt_global') and resolved_cfg.problem.opt_global is not None:
            # Optimum is already set in config, no need to load
            pass
        
        # Set default values for OneMax-specific parameters if not already set
        if resolved_cfg.problem.prob_name == 'onemax':
            if not hasattr(resolved_cfg.problem, 'capacity') or resolved_cfg.problem.capacity is None:
                resolved_cfg.problem.capacity = 0
            if not hasattr(resolved_cfg.problem, 'mean_value') or resolved_cfg.problem.mean_value is None:
                resolved_cfg.problem.mean_value = 50.0
            if not hasattr(resolved_cfg.problem, 'mean_weight') or resolved_cfg.problem.mean_weight is None:
                resolved_cfg.problem.mean_weight = 0.5
    
    # Step 2: Resolve algorithm dependencies
    if hasattr(resolved_cfg.algo, 'indpb_fn') and resolved_cfg.algo.indpb_fn is not None:
        # Set n_items for dynamic mutation rate
        if hasattr(resolved_cfg.problem, 'dimensions'):
            resolved_cfg.algo.indpb_fn.n_items = resolved_cfg.problem.dimensions
    
    # Step 3: Resolve mutation parameters
    if hasattr(resolved_cfg.algo, 'use_dynamic_mutation'):
        if resolved_cfg.algo.use_dynamic_mutation:
            # Calculate dynamic mutation rate
            if hasattr(resolved_cfg.algo, 'indpb_fn'):
                resolved_cfg.algo.init_args.mutate_params.indpb = call(resolved_cfg.algo.indpb_fn)
        else:
            # Use static mutation rate
            if hasattr(resolved_cfg.algo, 'static_indpb'):
                resolved_cfg.algo.init_args.mutate_params.indpb = resolved_cfg.algo.static_indpb
    
    # Step 4: Resolve noise-dependent eval limits
    if hasattr(resolved_cfg.run, 'use_noise_dependent_eval_limit'):
        if resolved_cfg.run.use_noise_dependent_eval_limit:
            noise_val = resolved_cfg.problem.fitness_params.noise_intensity
            if hasattr(resolved_cfg.run, 'eval_limit_for_noise'):
                mapping = {k: int(v) for k, v in resolved_cfg.run.eval_limit_for_noise.items()}
                resolved_cfg.run.eval_limit = mapping.get(f"{noise_val}", resolved_cfg.run.eval_limit)
    
    # Resolve problem ID
    resolved_cfg.problem.PID = determine_pid_from_cfg(resolved_cfg)

    return resolved_cfg

# -------------------------------
# Run Functions
# -------------------------------

def hydra_algo_data_single(prob_info: Dict[str, Any],
                          algo_config: Dict[str, Any],
                          algo_params: Dict[str, Any],
                          seed: int,
                          payload_dir: str = "data/temp/payloads") -> Tuple[Dict[str, Any], str]:
    # Seeds
    random.seed(seed)
    np.random.seed(seed)

    # Run algorithm
    import resource
    algo_config = OmegaConf.create(algo_config)
    # algo_params = OmegaConf.create(algo_params)
    algo_instance = instantiate(algo_config, **algo_params)
    algo_instance.run()
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    logger = algo_instance.logger

    # -------- BIG DATA: payload --------
    payload = {
        "rep_sols": logger.representative_solutions,
        "rep_true_fits": logger.representative_true_fitnesses,
        "rep_noisy_fits": logger.representative_noisy_fitnesses,
        "rep_noisy_sols": logger.representative_noisy_solutions,
        "rep_estimated_true_fits_whenadopted": logger.representative_estimated_true_fits_whenadopted,
        "rep_estimated_true_fits_whendiscarded": logger.representative_estimated_true_fits_whendiscarded,
        "count_estimated_fits_whenadopted": logger.count_estimated_fits_whenadopted,
        "count_estimated_fits_whendiscarded": logger.count_estimated_fits_whendiscarded,
        "sol_iterations": logger.solution_iterations,
        "sol_iterations_evals": logger.solution_evals,
        "sol_transitions": logger.solution_transitions,
        "noisy_sol_variants": logger.representative_noisy_sols,
        "noisy_variant_fitnesses": logger.noisy_variant_fitnesses,
    }

    seed_signature = algo_instance.seed_signature

    # Write payload to disk (unique name, safe for parallel)
    payload_dir = Path(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"stn_payload_seed{seed}_sig{seed_signature}.pkl"
    with open(payload_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -------- SMALL DATA: row (scalars only) --------
    rep_true_fits = payload["rep_true_fits"]
    row = {
        "problem_name": prob_info["name"],
        "problem_type": prob_info["type"],
        "problem_goal": prob_info["goal"],
        "dimensions": prob_info["dimensions"],
        "opt_global": prob_info["opt_global"],
        "mean_value": prob_info["mean_value"],
        "mean_weight": prob_info["mean_weight"],
        "PID": prob_info["PID"],

        "fit_func": algo_params["fitness_function"][0].__name__,
        "noise": algo_params["fitness_function"][1]["noise_intensity"],
        "algo_type": algo_instance.type,
        "algo_name": algo_instance.name,

        "n_gens": algo_instance.gens,
        "n_evals": algo_instance.evals,
        "stop_trigger": algo_instance.stop_trigger,
        "n_unique_sols": len(payload["rep_sols"]),

        "final_fit": rep_true_fits[-1] if rep_true_fits else None,
        "max_fit": max(rep_true_fits) if rep_true_fits else None,
        "min_fit": min(rep_true_fits) if rep_true_fits else None,

        "seed": seed,
        "seed_signature": seed_signature,

        "peak_ram_mb": round(peak_ram_mb, 1),

        # Keep track of where payload is (for main process logging)
        "payload_path": str(payload_path),
    }

    # Cleanup (important for sequential + parallel)
    clear_active_logger()
    algo_instance.logger.clear()
    if hasattr(algo_instance, "population"):
        del algo_instance.population

    return row, str(payload_path)


def hydra_algo_data_multi(prob_info: Dict[str, Any],
                          algo_config: Dict[str, Any],
                          algo_params: Dict[str, Any],
                          num_runs: int,
                          base_seed: int = 0,
                          parallel: bool = False,
                          override_max_workers: int = None) -> pd.DataFrame:

    results_list = []

    if parallel:
        import os
        max_workers = override_max_workers if override_max_workers is not None else min(num_runs, os.cpu_count() or 1)
        print(f"Running {num_runs} runs in PARALLEL with up to {max_workers} workers")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(num_runs):
                seed = base_seed + i
                futures.append(
                    executor.submit(hydra_algo_data_single, prob_info, algo_config, algo_params, seed)
                )

            for future in concurrent.futures.as_completed(futures):
                row, payload_path = future.result()
                results_list.append(row)

        print("Parallel execution complete.")
    else:
        print(f"Running {num_runs} runs SEQUENTIALLY")
        for i in range(num_runs):
            seed = base_seed + i
            row, payload_path = hydra_algo_data_single(prob_info, algo_config, algo_params, seed)
            results_list.append(row)

    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values(by="seed")
    return df_sorted

def mlflow_log_child_from_row(row: Dict[str, Any], algo_params: Dict[str, Any]) -> str:
    """
    Create one child run for this seed and log params/metrics/artifacts.
    Returns run_id.
    """
    # Descriptive child name
    child_name = (
        f"PID={row['PID']} | {row['algo_name']} | noise={row['noise']} | seed={row['seed']}"
    )

    with mlflow.start_run(run_name=child_name, nested=True) as child:
        run_id = child.info.run_id

        # ---- Params (config-ish) ----
        mlflow.log_params({
            "PID": row["PID"],
            "problem_name": row["problem_name"],
            "problem_type": row["problem_type"],
            "problem_goal": row["problem_goal"],
            "dimensions": row["dimensions"],
            "algo_type": row["algo_type"],
            "algo_name": row["algo_name"],
            "fit_func": row["fit_func"],
            "noise": row["noise"],
            "seed": row["seed"],
            "seed_signature": row["seed_signature"],
            "eval_limit": algo_params.get("eval_limit"),
        })

        # ---- Metrics (numbers) ----
        metrics = {
            "n_evals": int(row["n_evals"]),
            "n_gens": int(row["n_gens"]),
            "n_unique_sols": int(row["n_unique_sols"]),
        }
        if row["final_fit"] is not None: metrics["final_fit"] = float(row["final_fit"])
        if row["max_fit"] is not None:   metrics["max_fit"] = float(row["max_fit"])
        if row["min_fit"] is not None:   metrics["min_fit"] = float(row["min_fit"])
        mlflow.log_metrics(metrics)

        # ---- Artifacts (payload pickle) ----
        # Log with a FIXED name inside each run for easy dashboard retrieval later
        # We copy/rename into temp so MLflow sees "stn_payload.pkl" consistently.
        payload_src = Path(row["payload_path"])
        payload_tmp = payload_src.parent / "stn_payload.pkl"
        if payload_src.name != "stn_payload.pkl":
            # copy bytes (avoid shutil import if you want)
            payload_tmp.write_bytes(payload_src.read_bytes())
            mlflow.log_artifact(str(payload_tmp), artifact_path="payloads")
            # optional: remove tmp copy afterwards
            try:
                payload_tmp.unlink()
            except Exception:
                pass
        else:
            mlflow.log_artifact(str(payload_src), artifact_path="payloads")

        return run_id
    
# Temp function for dashboard transition
def enrich_df_with_payloads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rehydrate heavy trajectory columns from payload pickles so that
    algo_results.pkl remains dashboard-compatible.
    """
    df = df.copy()

    # Initialise columns (important so Pandas knows they exist)
    df["rep_sols"] = None
    df["rep_fits"] = None
    df["rep_noisy_fits"] = None
    df["rep_estimated_fits_whenadopted"] = None
    df["rep_estimated_fits_whendiscarded"] = None
    df["count_estimated_fits_whenadopted"] = None
    df["count_estimated_fits_whendiscarded"] = None
    df["sol_iterations"] = None
    df["sol_iterations_evals"] = None
    df["sol_transitions"] = None
    df["rep_noisy_sols"] = None
    df["noisy_sol_variants"] = None
    df["noisy_variant_fitnesses"] = None

    for idx, row in df.iterrows():
        payload_path = row.get("payload_path")
        if not payload_path:
            continue

        try:
            with open(payload_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load payload {payload_path}: {e}")
            continue

        # Map payload keys → dashboard column names
        df.at[idx, "rep_sols"] = payload.get("rep_sols", [])
        df.at[idx, "rep_fits"] = payload.get("rep_true_fits", [])
        df.at[idx, "rep_noisy_fits"] = payload.get("rep_noisy_fits", [])
        df.at[idx, "rep_estimated_fits_whenadopted"] = payload.get("rep_estimated_true_fits_whenadopted", [])
        df.at[idx, "rep_estimated_fits_whendiscarded"] = payload.get("rep_estimated_true_fits_whendiscarded", [])
        df.at[idx, "count_estimated_fits_whenadopted"] = payload.get("count_estimated_fits_whenadopted", [])
        df.at[idx, "count_estimated_fits_whendiscarded"] = payload.get("count_estimated_fits_whendiscarded", [])
        df.at[idx, "sol_iterations"] = payload.get("sol_iterations", [])
        df.at[idx, "sol_iterations_evals"] = payload.get("sol_iterations_evals", [])
        df.at[idx, "sol_transitions"] = payload.get("sol_transitions", [])
        df.at[idx, "rep_noisy_sols"] = payload.get("rep_noisy_sols", [])
        df.at[idx, "noisy_sol_variants"] = payload.get("noisy_sol_variants", [])
        df.at[idx, "noisy_variant_fitnesses"] = payload.get("noisy_variant_fitnesses", [])

    return df


@hydra.main(version_base=None, config_path="configs", config_name="test1_kp_1p1")
def main(cfg: DictConfig):
    start_time = time.perf_counter() # Record start time

    # Resolve dependencies in nested config structure
    cfg = resolve_config_dependencies(cfg)
    
    # Initialise MLflow
    # mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Problem metadata
    # Problem info is passed to algorithm class via run function
    prob_info = {
        'name':       cfg.problem.prob_name,
        'type':       cfg.problem.prob_type,
        'goal':       cfg.problem.opt_goal,
        'dimensions': cfg.problem.dimensions,
        'opt_global': cfg.problem.opt_global,
        'mean_value': cfg.problem.mean_value,
        'mean_weight': cfg.problem.mean_weight,
        'PID':        cfg.problem.PID,
    }

    # Instantiate fitness
    fitness_fn = getattr(sys.modules['src.problems'], cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)

    # Algorithm class and params
    algo_params = {
        'sol_length':                cfg.problem.dimensions,
        'opt_weights':               tuple(cfg.problem.weights),
        'eval_limit':                cfg.run.eval_limit,
        'attr_function':             getattr(sys.modules['src.algorithms'], cfg.problem.attr_function),
        'starting_solution':         None,
        'target_stop':               cfg.problem.opt_global if getattr(cfg.run, 'target_stop', False) else None,
        'gen_limit':                 None,
        'fitness_function':          (fitness_fn, fit_params),
        'progress_print_interval':   getattr(cfg.run, 'progress_print_interval', None),
        'record_population':         getattr(cfg.run, 'record_population', False),
    }

    # Run and log via MLflow
    with mlflow.start_run(run_name=f"SWEEP | PID={cfg.problem.PID} | {cfg.algo.name}") as parent:
        parent_run_id = parent.info.run_id
        print("PARENT artifact root:", mlflow.get_artifact_uri())

        # ---- Parent-level params ----
        mlflow.log_params({
            "PID": cfg.problem.PID,
            "problem_name": cfg.problem.prob_name,
            "dimensions": cfg.problem.dimensions,
            "base_seed": cfg.run.seed,
            "num_runs": cfg.run.num_runs,
            "eval_limit": cfg.run.eval_limit,
            "max_gens": cfg.run.max_gens,
            **{f"fit_{k}": v for k, v in fit_params.items()}
        })

        # Convert configs to plain python before passing to workers
        algo_config_plain = OmegaConf.to_container(cfg.algo.init_args, resolve=True)
        # algo_params_plain = OmegaConf.to_container(OmegaConf.create(algo_params), resolve=True)
        prob_info_plain   = dict(prob_info)

        # ---- Parallel compute (no MLflow inside workers) ----
        df = hydra_algo_data_multi(
            prob_info_plain,
            algo_config_plain,
            algo_params,
            num_runs=cfg.run.num_runs,
            base_seed=cfg.run.seed,
            parallel=cfg.run.parallel,
            override_max_workers=getattr(cfg.run, 'override_max_workers', None)
        )

        compute_time = time.perf_counter() - start_time
        print(f"Computation time: {compute_time:.2f}s")

        # ---- Child runs: log results + artifacts ----
        seed_to_runid = {}
        for row in df.to_dict(orient="records"):
            seed_to_runid[row["seed"]] = mlflow_log_child_from_row(row, algo_params)

        df["run_id"] = df["seed"].map(seed_to_runid)
        df["parent_run_id"] = parent_run_id

        # ---- Keep CSV unchanged (backup) ----
        df.to_csv('data/temp/results.csv', index=False)
        mlflow.log_artifact('data/temp/results.csv')

        # Keep your pickle + dashboard append unchanged for now
        df.to_pickle("data/temp/results.pkl")
        mlflow.log_artifact("data/temp/results.pkl")
        df_dashboard = enrich_df_with_payloads(df)
        save_or_append_results(df_dashboard, 'data/dashboard_dw/algo_results.pkl')

        mlflow.log_artifact("data/outputs/.hydra/config.yaml")

    process_time = time.perf_counter() - start_time - compute_time
    print(f"Processing/saving time: {process_time:.2f}s")

    # Print summary
    print(df[['seed', 'final_fit', 'n_evals', 'peak_ram_mb']])

if __name__ == '__main__':
    main()