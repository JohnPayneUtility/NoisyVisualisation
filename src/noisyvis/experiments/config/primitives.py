"""Config-resolution steps that are identical in every workflow that uses them (plan §5.4).

A step lives here only if the pre-Stage-7 copies of `resolve_config_dependencies` in the run
scripts implemented it identically -- guards, ordering, missing/None/falsy handling and exceptions
included -- in every workflow that calls it. Anything that differed between workflows stays in its
own function in `workflows.py`. Bodies are moved verbatim; do not "tidy" them.

    copy_resolved_config, has_problem_loader, load_problem_instance    SO, MO, LON, CoLON
    set_indpb_n_items, resolve_dynamic_pop_size,
    resolve_noise_dependent_eval_limit, resolve_pid                     SO, MO
"""

import numpy as np
from hydra.utils import call
from omegaconf import OmegaConf, DictConfig

from noisyvis.experiments.hyperparams import determine_pid_from_cfg


def copy_resolved_config(cfg: DictConfig) -> DictConfig:
    """A fully interpolated copy, so resolution never modifies the caller's config."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def has_problem_loader(resolved: DictConfig) -> bool:
    """Whether the problem is loaded from an instance (e.g. knapsack) rather than configured inline."""
    return "loader" in resolved.problem and resolved.problem.loader is not None


def load_problem_instance(resolved: DictConfig):
    """Call the loader and set the problem metadata on `resolved`, in the original order.

    Returns `(items_dict, capacity)`: the normalised items dict and the loader's raw capacity, which
    the workflows then inject into `fitness_params` (and, for CoLON, `violation_params`).
    """
    outputs = call(resolved.problem.loader)
    n_items, capacity, optimal, values, weights, items_dict, _ = outputs

    # Convert items_dict to proper format
    items_dict = {
        int(k): (float(v[0]), float(v[1]))
        for k, v in items_dict.items()
    }

    # Set problem metadata
    resolved.problem.dimensions = int(n_items)
    resolved.problem.opt_global = float(optimal)
    resolved.problem.capacity = float(capacity)
    resolved.problem.mean_value = float(np.mean(values))
    resolved.problem.mean_weight = float(np.mean(weights))
    resolved.problem.items_dict = items_dict
    return items_dict, capacity


def set_indpb_n_items(resolved: DictConfig) -> None:
    """Resolve algorithm dependencies: n_items for the dynamic mutation rate."""
    if hasattr(resolved.algo, 'indpb_fn') and resolved.algo.indpb_fn is not None:
        # Set n_items for dynamic mutation rate
        if hasattr(resolved.problem, 'dimensions'):
            resolved.algo.indpb_fn.n_items = resolved.problem.dimensions


def resolve_dynamic_pop_size(resolved: DictConfig) -> None:
    """Resolve dynamic population size."""
    if getattr(resolved.algo, 'use_dynamic_pop_size', False):
        fn_cfg = resolved.algo.pop_size_fn
        fn_cfg.n_items = resolved.problem.dimensions
        fn_cfg.noise = resolved.problem.fitness_params.noise_intensity
        resolved.algo.init_args.pop_size = call(fn_cfg)


def resolve_noise_dependent_eval_limit(resolved: DictConfig) -> None:
    """Resolve noise-dependent eval limits."""
    if hasattr(resolved.run, 'use_noise_dependent_eval_limit'):
        if resolved.run.use_noise_dependent_eval_limit:
            noise_val = resolved.problem.fitness_params.noise_intensity
            if hasattr(resolved.run, 'eval_limit_for_noise'):
                mapping = {k: int(v) for k, v in resolved.run.eval_limit_for_noise.items()}
                resolved.run.eval_limit = mapping.get(f"{noise_val}", resolved.run.eval_limit)


def resolve_pid(resolved: DictConfig) -> None:
    """Resolve problem ID."""
    resolved.problem.PID = determine_pid_from_cfg(resolved)
