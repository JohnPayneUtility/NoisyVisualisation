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
from noisyvis.algorithms import *
from noisyvis.problems import *
import random
import numpy as np
import pandas as pd
import concurrent.futures
import itertools
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Type
from deap import tools

from noisyvis.results.store import save_or_append_results
from noisyvis.results.paths import MLRUNS_DIR, TEMP_DIR, WAREHOUSE_DIR
from noisyvis.experiments.hyperparams import determine_pid_from_cfg
from noisyvis.experiments.config.workflows import resolve_mo_config

# explicit mlflow path
from pathlib import Path
mlruns_dir = MLRUNS_DIR                             # <project root>/data/mlruns (results.paths)
mlruns_dir.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")

print("RUN tracking:", mlflow.get_tracking_uri())
# -------------------------------
# Helper Functions for Dependency Resolution
# -------------------------------

# -------------------------------
# Run Functions
# -------------------------------

def hydra_algo_data_single(prob_info: Dict[str, Any], 
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

def hydra_algo_data_multi(prob_info: Dict[str, Any],
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
                futures.append(executor.submit(hydra_algo_data_single, prob_info, algo_config, algo_params, seed))
            for future in concurrent.futures.as_completed(futures):
                results_list.append(future.result())
    else:
        for i in range(num_runs):
            seed = base_seed + i
            results_list.append(hydra_algo_data_single(prob_info, algo_config, algo_params, seed))
    
    # Create a DataFrame from the list of dictionaries.
    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values(by='seed')
    return df_sorted

# -------------------------------
# Hydra config management
# -------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="test1_kp_1p1")
def main(cfg: DictConfig):
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
        df = hydra_algo_data_multi(
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

if __name__ == '__main__':
    import sys
    from pathlib import Path

    configs_root = Path(__file__).resolve().parent / "configs"
    symlink = None

    for i, arg in enumerate(sys.argv):
        if arg.startswith("--config-name=") or arg.startswith("--config-name"):
            val = arg.split("=", 1)[1] if "=" in arg else sys.argv[i + 1]
            if "/" in val:
                flat = val.replace("/", "__")
                symlink = configs_root / f"{flat}.yaml"
                symlink.symlink_to((configs_root / f"{val}.yaml").resolve())
                if "=" in arg:
                    sys.argv[i] = f"--config-name={flat}"
                else:
                    sys.argv[i + 1] = flat
            break

    try:
        main()
    finally:
        if symlink and symlink.exists():
            symlink.unlink()