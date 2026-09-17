"""Single- and multi-objective experiment runs behind `run.py` and `run_mo.py` (plan Stage 7).

Moved from the two run scripts with their bodies unchanged apart from the function renames below
and the STN payload helpers now in `payloads.py`. The SO and MO paths are deliberately separate:
they differ in results, executor sizing, payloads and MLflow logging.

    run.py     hydra_algo_data_single -> so_algo_data_single
               hydra_algo_data_multi  -> so_algo_data_multi
               main (body)            -> run_so_experiment
    run_mo.py  hydra_algo_data_single -> mo_algo_data_single
               hydra_algo_data_multi  -> mo_algo_data_multi
               main (body)            -> run_mo_experiment

Reproducibility (plan §7.1, R1, R3): each seed calls `random.seed` then `np.random.seed` before
anything else, and statement order inside these functions is part of the recorded baselines. Do not
reorder. Worker functions run in `ProcessPoolExecutor` processes; results are sorted by seed.

The MLflow tracking URI is NOT set here: the wrappers call `tracking.set_so_mo_tracking_uri()` at
import time, before Hydra runs `main` (R24).
"""

# IMPORTS
import shutil
import sys
import time
import random
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import mlflow

# The fitness-function and attribute-generator names in configs are looked up with
# getattr(sys.modules['noisyvis.problems' | 'noisyvis.algorithms'], name) (risk R5). These imports
# register both keys; the run scripts used to do it with star imports, in this order.
import noisyvis.algorithms
import noisyvis.problems

import numpy as np
import concurrent.futures
from pathlib import Path
from typing import Tuple, Any, Dict

from noisyvis.results.store import save_or_append_results
from noisyvis.results.paths import TEMP_DIR, WAREHOUSE_DIR
from noisyvis.tracking.logger import clear_active_logger
from noisyvis.experiments.config.workflows import resolve_mo_config, resolve_so_config
from noisyvis.experiments.payloads import (
    build_stn_payload,
    enrich_df_with_payloads,
    persisted_payload_path,
    write_stn_payload,
)
from noisyvis.experiments.tracking import mlflow_log_child_from_row


# -------------------------------
# Single-objective (run.py)
# -------------------------------

def so_algo_data_single(prob_info: Dict[str, Any],
                          algo_config: Dict[str, Any],
                          algo_params: Dict[str, Any],
                          seed: int,
                          payload_dir: str = str(TEMP_DIR / "payloads"),
                          nvme_base_path: str = None) -> Tuple[Dict[str, Any], str]:
    # Seeds
    random.seed(seed)
    np.random.seed(seed)

    # Construct a per-seed NVMe LMDB path if a base path is configured.
    # Each seed gets its own subdirectory so parallel runs don't share an environment.
    nvme_run_path = None
    if nvme_base_path is not None:
        nvme_run_path = Path(nvme_base_path) / f"lmdb_seed{seed}"
        # Directory creation is handled by LMDBFitHistory.__init__ via os.makedirs,
        # which follows symlinks correctly. Avoid pathlib.mkdir here — it does not
        # handle the case where a parent component is a symlink to a directory.

    # Inject nvme_path into a local copy of algo_params
    algo_params = dict(algo_params)
    algo_params['nvme_path'] = str(nvme_run_path) if nvme_run_path else None

    # Run algorithm
    import resource
    algo_config = OmegaConf.create(algo_config)
    algo_instance = instantiate(algo_config, **algo_params)
    algo_instance.run()
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    logger = algo_instance.logger

    # -------- BIG DATA: payload --------
    payload = build_stn_payload(logger)

    seed_signature = algo_instance.seed_signature

    # Write payload to disk (unique name, safe for parallel)
    payload_path = write_stn_payload(payload, payload_dir, seed, seed_signature)

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
        "experiment_name":        prob_info["experiment_name"],
        "experiment_description": prob_info["experiment_description"],

        "fit_func": algo_params["fitness_function"][0].__name__,
        "noise": algo_params["fitness_function"][1]["noise_intensity"],
        "penalty": algo_params["fitness_function"][1].get("penalty"),
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
        "payload_path": persisted_payload_path(payload_path),
    }

    # Cleanup (important for sequential + parallel)
    clear_active_logger()
    algo_instance.logger.clear()  # closes LMDB env if active (must happen before rmtree)
    if hasattr(algo_instance, "population"):
        del algo_instance.population

    # Delete the per-seed LMDB directory now that all data has been extracted
    if nvme_run_path is not None and nvme_run_path.exists():
        shutil.rmtree(nvme_run_path)

    return row, str(payload_path)


def so_algo_data_multi(prob_info: Dict[str, Any],
                          algo_config: Dict[str, Any],
                          algo_params: Dict[str, Any],
                          num_runs: int,
                          base_seed: int = 0,
                          parallel: bool = False,
                          override_max_workers: int = None,
                          nvme_base_path: str = None) -> pd.DataFrame:

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
                    executor.submit(so_algo_data_single, prob_info, algo_config, algo_params, seed,
                                    nvme_base_path=nvme_base_path)
                )

            for future in concurrent.futures.as_completed(futures):
                row, payload_path = future.result()
                results_list.append(row)

        print("Parallel execution complete.")
    else:
        print(f"Running {num_runs} runs SEQUENTIALLY")
        for i in range(num_runs):
            seed = base_seed + i
            row, payload_path = so_algo_data_single(prob_info, algo_config, algo_params, seed,
                                                        nvme_base_path=nvme_base_path)
            results_list.append(row)

    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values(by="seed")
    return df_sorted


def run_so_experiment(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    start_time = time.perf_counter() # Record start time

    # Resolve dependencies in nested config structure
    cfg = resolve_so_config(cfg)
    
    # Initialise MLflow
    # mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Problem metadata
    # Problem info is passed to algorithm class via run function
    prob_info = {
        'name':            cfg.problem.prob_name,
        'type':            cfg.problem.prob_type,
        'goal':            cfg.problem.opt_goal,
        'dimensions':      cfg.problem.dimensions,
        'opt_global':      cfg.problem.opt_global,
        'mean_value':      cfg.problem.mean_value,
        'mean_weight':     cfg.problem.mean_weight,
        'PID':             cfg.problem.PID,
        'experiment_name':        cfg.experiment_name,
        'experiment_description': cfg.get('experiment_description', ''),
    }

    # Instantiate fitness
    fitness_fn = getattr(sys.modules['noisyvis.problems'], cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)

    # Algorithm class and params
    algo_params = {
        'sol_length':                cfg.problem.dimensions,
        'opt_weights':               tuple(cfg.problem.weights),
        'eval_limit':                cfg.run.eval_limit,
        'attr_function':             getattr(sys.modules['noisyvis.algorithms'], cfg.problem.attr_function),
        'starting_solution':         None,
        'target_stop':               cfg.problem.opt_global if getattr(cfg.run, 'target_stop', False) else None,
        'no_improve_limit':          getattr(cfg.run, 'no_improve_limit', None),
        'noisy_no_improve_limit':    getattr(cfg.run, 'noisy_no_improve_limit', None),
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
        df = so_algo_data_multi(
            prob_info_plain,
            algo_config_plain,
            algo_params,
            num_runs=cfg.run.num_runs,
            base_seed=cfg.run.seed,
            parallel=cfg.run.parallel,
            override_max_workers=getattr(cfg.run, 'override_max_workers', None),
            nvme_base_path=getattr(cfg.run, 'nvme_path', None),
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
        df.to_csv(TEMP_DIR / 'results.csv', index=False)
        mlflow.log_artifact(str(TEMP_DIR / 'results.csv'))

        # Keep your pickle + dashboard append unchanged for now
        df.to_pickle(TEMP_DIR / "results.pkl")
        mlflow.log_artifact(str(TEMP_DIR / "results.pkl"))
        df_dashboard = enrich_df_with_payloads(df)
        save_or_append_results(df_dashboard, WAREHOUSE_DIR / 'algo_results.pkl')

        mlflow.log_artifact("data/outputs/.hydra/config.yaml")

    process_time = time.perf_counter() - start_time - compute_time
    print(f"Processing/saving time: {process_time:.2f}s")

    # Print summary
    print(df[['seed', 'final_fit', 'n_evals', 'peak_ram_mb']])



# -------------------------------
# Multi-objective (run_mo.py)
# -------------------------------

def mo_algo_data_single(prob_info: Dict[str, Any], 
                          algo_config: DictConfig, 
                          algo_params: Dict[str, Any], 
                          seed: int) -> Dict[str, Any]:
    """
    """
    # Set the random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create and run the algorithm instance.
    algo_instance = instantiate(algo_config, **algo_params)
    algo_instance.run()  # This updates the instance's internal data.
    
    # Retrieve derived data from the run.
    # unique_sols, unique_fits, noisy_fits, sol_iterations, sol_transitions = algo_instance.get_trajectory_data()
    # seed_signature = algo_instance.seed_signature
    
    return {
        "problem_name":           prob_info['name'],
        "problem_type":           prob_info['type'],
        "problem_goal":           prob_info['goal'],
        "dimensions":             prob_info['dimensions'],
        "opt_global":             prob_info['opt_global'],
        "mean_value":             prob_info['mean_value'],
        "mean_weight":            prob_info['mean_weight'],
        'PID':                    prob_info['PID'],
        "experiment_name":        prob_info['experiment_name'],
        "experiment_description": prob_info['experiment_description'],
        "fit_func": algo_params['fitness_function'][0].__name__,
        "noise": algo_params['fitness_function'][1]['noise_intensity'],
        # "algo_class": algorithm_class.__name__,
        "algo_type": algo_instance.type,
        "algo_name": algo_instance.name,
        "n_gens": algo_instance.gens,
        "n_evals": algo_instance.evals,
        "stop_trigger": algo_instance.stop_trigger,
        # "n_unique_sols": len(unique_sols),
        # "unique_sols": unique_sols,
        # "unique_fits": unique_fits,
        # "noisy_fits": noisy_fits,
        # "final_fit": unique_fits[-1],
        # "max_fit": max(unique_fits),
        # "min_fit": min(unique_fits),
        # "sol_iterations": sol_iterations,
        # "sol_transitions": sol_transitions,
        "seed": seed,
        "seed_signature": algo_instance.seed_signature,
        # PARETO DATA
        # noisy PF data (as the algorithm optimises)
        "pareto_solutions": algo_instance.pareto_solutions,
        "pareto_fitnesses": algo_instance.pareto_fitnesses,
        "pareto_true_fitnesses": algo_instance.pareto_true_fitnesses,
        # true PF (approx) built from full-pop true evals
        "true_pareto_solutions": algo_instance.true_pareto_solutions,
        "true_pareto_fitnesses": algo_instance.true_pareto_fitnesses,
        # hypervolumes (full lists)
        "noisy_pf_noisy_hypervolumes": algo_instance.noisy_pf_noisy_hypervolumes,
        "noisy_pf_true_hypervolumes": algo_instance.noisy_pf_true_hypervolumes,
        "true_pf_hypervolumes": algo_instance.true_pf_hypervolumes,
        # hypervolume scalar values (for dashboard plotting/tables)
        "final_true_hv": algo_instance.true_pf_hypervolumes[-1] if algo_instance.true_pf_hypervolumes else None,
        "max_true_hv": max(algo_instance.true_pf_hypervolumes) if algo_instance.true_pf_hypervolumes else None,
        "min_true_hv": min(algo_instance.true_pf_hypervolumes) if algo_instance.true_pf_hypervolumes else None,
        "final_noisy_pf_hv": algo_instance.noisy_pf_true_hypervolumes[-1] if algo_instance.noisy_pf_true_hypervolumes else None,
        "max_noisy_pf_hv": max(algo_instance.noisy_pf_true_hypervolumes) if algo_instance.noisy_pf_true_hypervolumes else None,
        "min_noisy_pf_hv": min(algo_instance.noisy_pf_true_hypervolumes) if algo_instance.noisy_pf_true_hypervolumes else None,
        # iterations
        "n_gens_pareto_best": algo_instance.n_gens_pareto_best
    }


def mo_algo_data_multi(prob_info: Dict[str, Any],
                    algo_config: DictConfig, 
                    algo_params: Dict[str, Any], 
                    num_runs: int, 
                    base_seed: int = 0, 
                    parallel: bool = False) -> pd.DataFrame:

    results_list = []
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_runs):
                seed = base_seed + i
                futures.append(executor.submit(mo_algo_data_single, prob_info, algo_config, algo_params, seed))
            for future in concurrent.futures.as_completed(futures):
                results_list.append(future.result())
    else:
        for i in range(num_runs):
            seed = base_seed + i
            results_list.append(mo_algo_data_single(prob_info, algo_config, algo_params, seed))
    
    # Create a DataFrame from the list of dictionaries.
    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values(by='seed')
    return df_sorted


def run_mo_experiment(cfg: DictConfig):
    start_time = time.perf_counter() # Record start time

    # Resolve dependencies in nested config structure
    cfg = resolve_mo_config(cfg)
    
    # Initialise MLflow (tracking URI set at module level)
    # mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Problem metadata
    # Problem info is passed to algorithm class via run function
    prob_info = {
        'name':                   cfg.problem.prob_name,
        'type':                   cfg.problem.prob_type,
        'goal':                   cfg.problem.opt_goal,
        'dimensions':             cfg.problem.dimensions,
        'opt_global':             cfg.problem.opt_global,
        'mean_value':             cfg.problem.mean_value,
        'mean_weight':            cfg.problem.mean_weight,
        'PID':                    cfg.problem.PID,
        'experiment_name':        cfg.experiment_name,
        'experiment_description': cfg.get('experiment_description', ''),
    }

    # Instantiate fitness
    fitness_fn = getattr(sys.modules['noisyvis.problems'], cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)

    # Algorithm class and params
    true_fit_params = fit_params.copy()
    true_fit_params['noise_intensity'] = 0
    ref_point = cfg.problem.get("ref_point", None)

    # check for starting solution
    # ! possibly move into dependency resolution
    cfg_start_sol = getattr(cfg.algo, "starting_solution", None)
    if cfg_start_sol is not None:
        start_sol = list(cfg_start_sol)
    else:
        start_sol = None

    # create algo parameter dict
    algo_params = {
        'sol_length':            cfg.problem.dimensions,
        'opt_weights':           tuple(cfg.problem.weights),
        'eval_limit':            cfg.run.eval_limit,
        'attr_function':         getattr(sys.modules['noisyvis.algorithms'], cfg.problem.attr_function),
        'starting_solution':     start_sol,
        # 'target_stop':           cfg.problem.opt_global,
        'target_stop':           None,
        'gen_limit':             cfg.run.max_gens,
        'stop_without_improvement_in_gens': cfg.run.get("stop_without_improvement_in_gens", None),
        'fitness_function':      (fitness_fn, fit_params),
        'true_fitness_function': (fitness_fn, true_fit_params),
        'ref_point': ref_point,
        'verbose_rate': cfg.run.get("verbose_rate", 0),
    }

    # Run and log via MLflow
    with mlflow.start_run(run_name=cfg.algo.name):
        # Log parameters
        print("RUN artifact root:", mlflow.get_artifact_uri())
        mlflow.log_params({
            'dimensions':  cfg.problem.dimensions,
            'seed':        cfg.run.seed,
            'max_gens':    cfg.run.max_gens,
            **{f"fit_{k}": v for k, v in fit_params.items()}
        })

        # Execute experiment (single or multirun seed)
        df = mo_algo_data_multi(
            prob_info,
            cfg.algo.init_args,
            algo_params,
            num_runs=cfg.run.num_runs,
            base_seed=cfg.run.seed,
            parallel=cfg.run.parallel
        )

        # Log metrics and artifacts
        for row in df.itertuples():
            # Log final hypervolume instead of single fitness
            if row.true_pf_hypervolumes:
                mlflow.log_metric('final_true_hypervolume', row.true_pf_hypervolumes[-1], step=row.seed)
            if row.noisy_pf_true_hypervolumes:
                mlflow.log_metric('final_noisy_pf_hypervolume', row.noisy_pf_true_hypervolumes[-1], step=row.seed)
        df.to_csv(TEMP_DIR / 'results.csv', index=False) # save csv
        mlflow.log_artifact(str(TEMP_DIR / 'results.csv'))
        df.to_pickle(TEMP_DIR / "results.pkl") # save pickle
        save_or_append_results(df, WAREHOUSE_DIR / 'algo_results.pkl')
        mlflow.log_artifact(str(TEMP_DIR / "results.pkl"))
        mlflow.log_artifact("data/outputs/.hydra/config.yaml")
    
    mlflow.end_run(status="FINISHED")

    # Print summary - show final hypervolume instead of final_fit
    summary_df = df[['seed', 'n_gens', 'n_evals']].copy()
    summary_df['final_hv'] = df['true_pf_hypervolumes'].apply(lambda x: x[-1] if x else None)
    print(summary_df)

    compute_time = time.perf_counter() - start_time
    print(compute_time)
