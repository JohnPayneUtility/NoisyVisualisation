
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
# Algorithms imported via src.algorithms above

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
    
    return resolved_cfg

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
    unique_sols, unique_fits, noisy_fits, sol_iterations, sol_transitions = algo_instance.get_trajectory_data()
    seed_signature = algo_instance.seed_signature
    
    return {
        "problem_name": prob_info['name'],
        "problem_type": prob_info['type'],
        "problem_goal": prob_info['goal'],
        "dimensions": prob_info['dimensions'],
        "opt_global": prob_info['opt_global'],
        "mean_value": prob_info['mean_value'],
        "mean_weight": prob_info['mean_weight'],
        'PID': prob_info['PID'],
        "fit_func": algo_params['fitness_function'][0].__name__,
        "noise": algo_params['fitness_function'][1]['noise_intensity'],
        # "algo_class": algorithm_class.__name__,
        "algo_type": algo_instance.type,
        "algo_name": algo_instance.name,
        "n_gens": algo_instance.gens,
        "n_evals": algo_instance.evals,
        "stop_trigger": algo_instance.stop_trigger,
        "n_unique_sols": len(unique_sols),
        "unique_sols": unique_sols,
        "unique_fits": unique_fits,
        "noisy_fits": noisy_fits,
        "final_fit": unique_fits[-1],
        "max_fit": max(unique_fits),
        "min_fit": min(unique_fits),
        "sol_iterations": sol_iterations,
        "sol_transitions": sol_transitions,
        "seed": seed,
        "seed_signature": seed_signature,
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
    # Record start time
    start_time = time.perf_counter()

    # Resolve dependencies in nested config structure
    cfg = resolve_config_dependencies(cfg)
    
    # Initialise MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Problem metadata
    prob_info = {
        'name':       cfg.problem.prob_name,
        'type':       cfg.problem.prob_type,
        'goal':       cfg.problem.opt_goal,
        'dimensions': cfg.problem.dimensions,
        'opt_global': cfg.problem.opt_global,
        'mean_value': cfg.problem.mean_value,
        'mean_weight': cfg.problem.mean_weight,
        'PID':        f"{cfg.problem.prob_name}_{cfg.problem.dimensions}"
    }

    # Instantiate fitness
    fitness_fn = getattr(sys.modules['src.problems'], cfg.problem.fitness_fn)
    fit_params = dict(cfg.problem.fitness_params)

    # Algorithm class and params
    true_fit_params = fit_params.copy()
    true_fit_params['noise_intensity'] = 0

    algo_params = {
        'sol_length':            cfg.problem.dimensions,
        'opt_weights':           tuple(cfg.problem.weights),
        'eval_limit':            cfg.run.eval_limit,
        'attr_function':         getattr(sys.modules['src.algorithms'], cfg.problem.attr_function),
        'starting_solution':     None,
        'target_stop':           cfg.problem.opt_global,
        'gen_limit':             None,
        'fitness_function':      (fitness_fn, fit_params),
        'true_fitness_function': (fitness_fn, true_fit_params)
    }

    # Run and log via MLflow
    with mlflow.start_run(run_name=cfg.algo.name):
        # Log parameters
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
            mlflow.log_metric('final_fitness', row.final_fit, step=row.seed)
        df.to_csv('data/results.csv', index=False)
        mlflow.log_artifact('data/results.csv')
        mlflow.log_artifact("data/outputs/.hydra/config.yaml")
    
    mlflow.end_run(status="FINISHED")

    # Print summary
    print(df[['seed', 'final_fit']])

    compute_time = time.perf_counter() - start_time
    print(compute_time)

if __name__ == '__main__':
    main()