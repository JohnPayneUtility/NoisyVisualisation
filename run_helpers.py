import numpy as np
from omegaconf import DictConfig

# -------------------------------
# Helpers for population size
# -------------------------------

def dynamic_pop_size_UMDA(n_items, noise):
    return int(20 * np.sqrt(n_items) * np.log(n_items))

def dynamic_pop_size_PCEA(n_items, noise):
    return int(10 * np.sqrt(n_items) * np.log(n_items))

def dynamic_pop_size_mu(n_items, noise):
    return int(max(noise * noise, 1) * np.log(n_items))

def inverse_n_mut_rate(n_items, noise):
    return 1/n_items

# -------------------------------
# Helpers for config dependencies
# -------------------------------

def determine_pid_from_cfg(cfg: DictConfig) -> str:
    """ Returns PID string based on provided config """

    # If PID provided in config opt for this first
    override_PID = getattr(cfg.problem, 'PID', None)
    if override_PID is not None:
        return override_PID
    
    # If problem loader with filename provided use this next
    loader = getattr(cfg.problem, 'loader', None)
    if loader is not None:
        filename = getattr(loader, 'filename', None)
        return filename
    
    # Else use combination of problem name and dimension
    prob_name = getattr(cfg.problem, "prob_name", "problem")
    dimensions = getattr(cfg.problem, "dimensions", None)
    pid = f'{prob_name}_{dimensions}'
    return pid