"""Per-workflow config resolution: SO, MO, LON and CoLON (plan §5.4).

Each function replaces the `resolve_config_dependencies` copy of its run script(s), statement for
statement: steps identical across workflows are calls into `primitives.py`, and every step whose
semantics differ stays written out here. There are deliberately no mode flags. Differences that
must survive, several of them latent in today's configs:

    fitness_params     SO, MO, LON: `hasattr` guard.  CoLON: `hasattr` and `is not None`.
    unloaded defaults  SO, MO: only for `onemax`, when missing or None (an explicit 0 is kept).
                       LON, CoLON: any problem, when falsy (an explicit 0 is overwritten).
    mutation           SO: never creates `mutate_params`; dynamic mode needs `indpb_fn`.
                       MO: `mutate_kwargs` naming, may create `mutate_params`, no `indpb_fn` guard.
    MO only            knapsack reference point.
    SO only            fraction-based no-improve limits.
    CoLON only         `violation_params` injection.
    LON, CoLON         no indpb/mutation/population/eval-limit steps and no PID resolution.

The recorded behaviour of all of these is pinned by `tests/test_config_workflows.py`.
"""

from hydra.utils import call
from omegaconf import OmegaConf, DictConfig

from noisyvis.experiments.config.primitives import (
    copy_resolved_config,
    has_problem_loader,
    load_problem_instance,
    resolve_dynamic_pop_size,
    resolve_noise_dependent_eval_limit,
    resolve_pid,
    set_indpb_n_items,
)


# ----------------------------------------------------------------------- unloaded problems

def _apply_onemax_unloaded_defaults(resolved: DictConfig) -> None:
    """SO and MO: defaults for `onemax` only, applied when a value is missing or None.

    The two `pass` probes are not no-ops: reading a `???` value raises MissingMandatoryValue.
    """
    # For problems like OneMax that don't need a loader
    # Ensure dimensions and opt_global are set if they exist in config
    if hasattr(resolved.problem, 'dimensions') and resolved.problem.dimensions is not None:
        # Dimensions are already set in config, no need to load
        pass
    if hasattr(resolved.problem, 'opt_global') and resolved.problem.opt_global is not None:
        # Optimum is already set in config, no need to load
        pass

    # Set default values for OneMax-specific parameters if not already set
    if resolved.problem.prob_name == 'onemax':
        if not hasattr(resolved.problem, 'capacity') or resolved.problem.capacity is None:
            resolved.problem.capacity = 0
        if not hasattr(resolved.problem, 'mean_value') or resolved.problem.mean_value is None:
            resolved.problem.mean_value = 50.0
        if not hasattr(resolved.problem, 'mean_weight') or resolved.problem.mean_weight is None:
            resolved.problem.mean_weight = 0.5


def _apply_falsy_unloaded_defaults(resolved: DictConfig) -> None:
    """LON and CoLON: defaults for any unloaded problem, applied when a value is falsy."""
    # OneMax-style problems (no loader)
    if not getattr(resolved.problem, "capacity", None):
        resolved.problem.capacity = 0
    if not getattr(resolved.problem, "mean_value", None):
        resolved.problem.mean_value = 50.0
    if not getattr(resolved.problem, "mean_weight", None):
        resolved.problem.mean_weight = 0.5


# ------------------------------------------------------------------------------ workflows

def resolve_so_config(cfg: DictConfig) -> DictConfig:
    """Single-objective (run.py)."""
    resolved_cfg = copy_resolved_config(cfg)

    # Step 1: Load problem data if needed
    if has_problem_loader(resolved_cfg):
        items_dict, capacity = load_problem_instance(resolved_cfg)

        # Set fitness parameters that depend on problem loading
        if hasattr(resolved_cfg.problem, 'fitness_params'):
            resolved_cfg.problem.fitness_params.items_dict = items_dict
            resolved_cfg.problem.fitness_params.capacity = float(capacity)
    else:
        _apply_onemax_unloaded_defaults(resolved_cfg)

    # Step 2: Resolve algorithm dependencies
    set_indpb_n_items(resolved_cfg)

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

    # Step 4: Resolve dynamic population size
    resolve_dynamic_pop_size(resolved_cfg)

    # Step 5: Resolve noise-dependent eval limits
    resolve_noise_dependent_eval_limit(resolved_cfg)

    # Step 6: Resolve fraction-based no-improve limits from eval_limit
    no_improve_fraction = getattr(resolved_cfg.run, 'no_improve_limit_fraction', None)
    if no_improve_fraction is not None:
        resolved_cfg.run.no_improve_limit = int(resolved_cfg.run.eval_limit * no_improve_fraction)
    noisy_fraction = getattr(resolved_cfg.run, 'noisy_no_improve_limit_fraction', None)
    if noisy_fraction is not None:
        resolved_cfg.run.noisy_no_improve_limit = int(resolved_cfg.run.eval_limit * noisy_fraction)

    # Resolve problem ID
    resolve_pid(resolved_cfg)

    return resolved_cfg


def resolve_mo_config(cfg: DictConfig) -> DictConfig:
    """Multi-objective (run_mo.py)."""
    resolved_cfg = copy_resolved_config(cfg)

    # Step 1: Load problem data if needed
    if has_problem_loader(resolved_cfg):
        items_dict, capacity = load_problem_instance(resolved_cfg)

        # Set fitness parameters that depend on problem loading
        if hasattr(resolved_cfg.problem, 'fitness_params'):
            resolved_cfg.problem.fitness_params.items_dict = items_dict
            resolved_cfg.problem.fitness_params.capacity = float(capacity)

        # Set reference point for multiobjective knapsack
        if getattr(resolved_cfg.problem, "prob_name", "").lower() == "knapsack":
            W = float(sum(float(v[1]) for v in resolved_cfg.problem.items_dict.values()))
            # code will fail if fitness function produces weight higher than ref point for knapsack
            resolved_cfg.problem.ref_point = [0.0, 2*W] # double max weight to account for noise
    else:
        _apply_onemax_unloaded_defaults(resolved_cfg)

    # Step 2: Resolve algorithm dependencies
    set_indpb_n_items(resolved_cfg)

    # Step 3: Resolve mutation parameters
    if hasattr(resolved_cfg.algo, 'use_dynamic_mutation'):
        # Ensure mutation dict exists in either naming scheme
        if not hasattr(resolved_cfg.algo.init_args, "mutate_params") and hasattr(resolved_cfg.algo.init_args, "mutate_kwargs"):
            # NSGA2-style naming present
            mut_container = resolved_cfg.algo.init_args.mutate_kwargs
            target_path = "mutate_kwargs"
        else:
            # legacy naming
            if not hasattr(resolved_cfg.algo.init_args, "mutate_params"):
                resolved_cfg.algo.init_args.mutate_params = OmegaConf.create({})
            mut_container = resolved_cfg.algo.init_args.mutate_params
            target_path = "mutate_params"

        if resolved_cfg.algo.use_dynamic_mutation:
            resolved_cfg.algo.init_args[target_path].indpb = call(resolved_cfg.algo.indpb_fn)
        else:
            if hasattr(resolved_cfg.algo, 'static_indpb'):
                resolved_cfg.algo.init_args[target_path].indpb = resolved_cfg.algo.static_indpb

    # Step 4: Resolve dynamic population size
    resolve_dynamic_pop_size(resolved_cfg)

    # Step 5: Resolve noise-dependent eval limits
    resolve_noise_dependent_eval_limit(resolved_cfg)

    # Resolve problem ID
    resolve_pid(resolved_cfg)

    return resolved_cfg


def resolve_lon_config(cfg: DictConfig) -> DictConfig:
    """Local optima networks (run_lon.py and run_lon_parallel.py)."""
    resolved = copy_resolved_config(cfg)

    # Load problem if a loader is specified (e.g., knapsack instance)
    if has_problem_loader(resolved):
        items_dict, capacity = load_problem_instance(resolved)

        # Fitness params
        if hasattr(resolved.problem, "fitness_params"):
            resolved.problem.fitness_params.items_dict = items_dict
            resolved.problem.fitness_params.capacity = float(capacity)
    else:
        _apply_falsy_unloaded_defaults(resolved)

    return resolved


def resolve_colon_config(cfg: DictConfig) -> DictConfig:
    """Constrained local optima networks (run_colon_parallel.py)."""
    resolved = copy_resolved_config(cfg)

    # Load problem if a loader is specified (e.g., knapsack instance)
    if has_problem_loader(resolved):
        items_dict, capacity = load_problem_instance(resolved)

        # Fitness params
        if hasattr(resolved.problem, "fitness_params") and resolved.problem.fitness_params is not None:
            resolved.problem.fitness_params.items_dict = items_dict
            resolved.problem.fitness_params.capacity = float(capacity)

        # Violation params (optional)
        if hasattr(resolved.problem, "violation_params") and resolved.problem.violation_params is not None:
            resolved.problem.violation_params.items_dict = items_dict
            resolved.problem.violation_params.capacity = float(capacity)
    else:
        _apply_falsy_unloaded_defaults(resolved)

    return resolved
