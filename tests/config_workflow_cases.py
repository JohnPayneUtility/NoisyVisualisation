"""Synthetic configs that characterise the four config-resolution workflows (plan §5.4, Stage 7).

The five reproducibility baselines exercise only the config-resolution branches their tiny configs
happen to reach. Stage 7 moves config resolution into `noisyvis.experiments.config`, and §5.4 records
differences between the SO, MO, LON and CoLON resolvers that are latent today: None versus missing
versus falsy defaults, the MO mutation container, the CoLON `is not None` guards, and so on. Each case
below pins one of those branches.

The recorded outcomes (`baselines/config_workflows.json`) came from the untouched pre-Stage-7
`resolve_config_dependencies` copies in the run scripts. They are frozen: a mismatch means resolution
behaviour changed, and is never fixed by re-recording.

Pure data: nothing here imports the science packages. Every case is built fresh, so no case can
mutate another's input.
"""

from __future__ import annotations

import copy

DELETE = object()

KP10 = "f1_l-d_kp_10_269"


def _patch(base: dict, patch: dict) -> dict:
    """Deep-merge `patch` into a copy of `base`; a DELETE value removes the key."""
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if value is DELETE:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _patch(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# ------------------------------------------------------------------------ building blocks

LOADED_KP_PROBLEM = {
    "prob_name": "knapsack",
    "prob_type": "discrete",
    "opt_goal": "maximise",
    "dimensions": None,
    "opt_global": None,
    "capacity": None,
    "mean_value": None,
    "mean_weight": None,
    "items_dict": None,
    "loader": {"_target_": "src.problems.ProblemScripts.load_problem_KP", "filename": KP10},
    "fitness_fn": "eval_noisy_kp_v1",
    "fitness_params": {"items_dict": None, "capacity": None, "noise_intensity": 1},
    "attr_function": "binary_attribute",
    "weights": [1.0],
}

UNLOADED_ONEMAX_PROBLEM = {
    "prob_name": "onemax",
    "prob_type": "discrete",
    "opt_goal": "maximise",
    "dimensions": 20,
    "opt_global": 20,
    "capacity": None,
    "mean_value": None,
    "mean_weight": None,
    "loader": None,
    "fitness_fn": "OneMax_fitness",
    "fitness_params": {"noise_intensity": 1},
    "attr_function": "binary_attribute",
    "weights": [1.0],
}

SO_ALGO = {
    "name": "OnePlusOneEA",
    "type": "MuPlusLamdaEA",
    "use_dynamic_pop_size": False,
    "static_indpb": 0.01,
    "use_dynamic_mutation": True,
    "indpb_fn": {"_target_": "run_helpers.inverse_n_mut_rate", "n_items": None, "noise": 1},
    "init_args": {
        "_target_": "src.algorithms.Algorithms.MuPlusLamdaEA",
        "mu": 1,
        "lam": 1,
        "mutate_function": "probFlipBit",
        "mutate_params": {"indpb": None},
    },
}

MO_ALGO = {
    "name": "SEMO",
    "type": "SEMO",
    "init_args": {"_target_": "src.algorithms.MOAlgorithms.SEMO"},
}

RUN = {
    "max_gens": 10,
    "eval_limit": 100,
    "use_noise_dependent_eval_limit": False,
    "eval_limit_for_noise": None,
    "seed": 1,
    "num_runs": 1,
    "parallel": False,
}

DYNAMIC_EXTRAS = {
    "algo": {
        "use_dynamic_pop_size": True,
        "pop_size_fn": {"_target_": "run_helpers.dynamic_pop_size_UMDA", "n_items": None, "noise": None},
        "init_args": {"pop_size": None},
    },
    "run": {
        "use_noise_dependent_eval_limit": True,
        "eval_limit_for_noise": {"0": 500, "1": 700, "2": 900},
        "no_improve_limit_fraction": 0.5,
        "noisy_no_improve_limit_fraction": 0.25,
    },
}

LON_BLOCK = {"name": "BinaryLON_case", "pert_attempts": 10, "n_flips_mut": 1, "n_flips_pert": 2}


def _so(problem: dict, algo_patch: dict | None = None, run_patch: dict | None = None) -> dict:
    return {
        "experiment_name": "config_workflow_case",
        "run": _patch(RUN, run_patch or {}),
        "algo": _patch(SO_ALGO, algo_patch or {}),
        "problem": problem,
    }


def _mo(problem: dict, algo_patch: dict | None = None, run_patch: dict | None = None) -> dict:
    return {
        "experiment_name": "config_workflow_case",
        "run": _patch(RUN, run_patch or {}),
        "algo": _patch(MO_ALGO, algo_patch or {}),
        "problem": problem,
    }


def _lon(problem: dict, extra: dict | None = None) -> dict:
    cfg = {
        "experiment_name": "config_workflow_case",
        "run": _patch(RUN, {}),
        "mlflow": {"tracking_uri": "data/mlruns"},
        "lon": dict(LON_BLOCK),
        "problem": problem,
    }
    return _patch(cfg, extra or {})


def _colon_problem(patch: dict) -> dict:
    base = _patch(
        LOADED_KP_PROBLEM,
        {
            "violation_fn": "src.problems.ViolationFunctions.knap_violation",
            "violation_params": {"items_dict": None, "capacity": None},
        },
    )
    return _patch(base, patch)


# --------------------------------------------------------------------------------- cases

def build_cases() -> dict:
    """{workflow: {case name: plain config dict}} -- rebuilt on every call."""
    so = {
        "so_loaded_knapsack_dynamic_mutation": _so(copy.deepcopy(LOADED_KP_PROBLEM)),
        "so_loaded_no_fitness_params_static_mutation": _so(
            _patch(LOADED_KP_PROBLEM, {"fitness_params": DELETE}),
            algo_patch={"use_dynamic_mutation": False, "static_indpb": 0.05},
        ),
        "so_unloaded_onemax_null_defaults": _so(copy.deepcopy(UNLOADED_ONEMAX_PROBLEM)),
        "so_unloaded_onemax_missing_keys": _so(
            _patch(
                UNLOADED_ONEMAX_PROBLEM,
                {"capacity": DELETE, "mean_value": DELETE, "mean_weight": DELETE, "loader": DELETE},
            )
        ),
        "so_unloaded_onemax_explicit_zero_kept": _so(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"capacity": 0, "mean_value": 0.0, "mean_weight": 0.0})
        ),
        "so_unloaded_non_onemax_untouched": _so(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"prob_name": "jump", "fitness_fn": "jump_fitness"})
        ),
        "so_dynamic_pop_noise_limit_fractions": _so(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"dimensions": 30, "opt_global": 30}),
            algo_patch=DYNAMIC_EXTRAS["algo"],
            run_patch=DYNAMIC_EXTRAS["run"],
        ),
        "so_noise_limit_mapping_miss": _so(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"fitness_params": {"noise_intensity": 3}}),
            algo_patch={"use_dynamic_mutation": DELETE},
            run_patch={
                "use_noise_dependent_eval_limit": True,
                "eval_limit_for_noise": {"0": 500, "1": 700},
            },
        ),
        "so_missing_mandatory_value_raises": _so(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"dimensions": "???"})
        ),
        "so_pid_override_kept_without_indpb_fn": _so(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"PID": "custom_pid"}),
            algo_patch={"indpb_fn": DELETE},
        ),
    }

    mo = {
        "mo_loaded_knapsack_ref_point_dynamic_mutation": _mo(
            copy.deepcopy(LOADED_KP_PROBLEM),
            algo_patch={
                "use_dynamic_mutation": True,
                "indpb_fn": copy.deepcopy(SO_ALGO["indpb_fn"]),
                "init_args": {"mutate_params": {"indpb": None}},
            },
        ),
        "mo_loaded_mixed_case_knapsack_ref_point": _mo(
            _patch(LOADED_KP_PROBLEM, {"prob_name": "KnapSack"})
        ),
        "mo_loaded_non_knapsack_no_ref_point": _mo(
            _patch(LOADED_KP_PROBLEM, {"prob_name": "kp_other"})
        ),
        "mo_mutate_kwargs_dynamic": _mo(
            copy.deepcopy(LOADED_KP_PROBLEM),
            algo_patch={
                "use_dynamic_mutation": True,
                "indpb_fn": copy.deepcopy(SO_ALGO["indpb_fn"]),
                "init_args": {"mutate_kwargs": {"indpb": None, "other": 3}},
            },
        ),
        "mo_mutation_container_created_static_without_indpb": _mo(
            copy.deepcopy(UNLOADED_ONEMAX_PROBLEM),
            algo_patch={"use_dynamic_mutation": False},
        ),
        "mo_mutate_params_preferred_over_kwargs": _mo(
            copy.deepcopy(UNLOADED_ONEMAX_PROBLEM),
            algo_patch={
                "use_dynamic_mutation": False,
                "static_indpb": 0.2,
                "init_args": {"mutate_params": {"indpb": None}, "mutate_kwargs": {"indpb": None}},
            },
        ),
        "mo_dynamic_mutation_without_indpb_fn_raises": _mo(
            copy.deepcopy(UNLOADED_ONEMAX_PROBLEM),
            algo_patch={"use_dynamic_mutation": True, "init_args": {"mutate_params": {"indpb": None}}},
        ),
        "mo_dynamic_pop_noise_limit_fractions_ignored": _mo(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"dimensions": 30, "opt_global": 30}),
            algo_patch={
                "use_dynamic_pop_size": True,
                "pop_size_fn": {
                    "_target_": "run_helpers.dynamic_pop_size_PCEA",
                    "n_items": None,
                    "noise": None,
                },
                "init_args": {"pop_size": None},
            },
            run_patch=DYNAMIC_EXTRAS["run"],
        ),
        "mo_unloaded_onemax_null_defaults": _mo(copy.deepcopy(UNLOADED_ONEMAX_PROBLEM)),
    }

    lon = {
        "lon_loaded_knapsack_no_pid_no_algo_steps": _lon(
            copy.deepcopy(LOADED_KP_PROBLEM),
            extra={
                "algo": copy.deepcopy(SO_ALGO),
                "run": {"no_improve_limit_fraction": 0.5},
            },
        ),
        "lon_unloaded_falsy_defaults_overwrite_zero": _lon(
            _patch(
                UNLOADED_ONEMAX_PROBLEM,
                {"prob_name": "jump", "capacity": 0, "mean_value": 0.0, "mean_weight": DELETE},
            )
        ),
        "lon_loaded_fitness_params_null_raises": _lon(
            _patch(LOADED_KP_PROBLEM, {"fitness_params": None})
        ),
    }

    colon = {
        "colon_loaded_violation_params": _lon(_colon_problem({})),
        "colon_fitness_params_null_skipped": _lon(_colon_problem({"fitness_params": None})),
        "colon_violation_params_null_skipped": _lon(_colon_problem({"violation_params": None})),
        "colon_unloaded_falsy_defaults": _lon(
            _patch(UNLOADED_ONEMAX_PROBLEM, {"mean_value": 0, "mean_weight": 0.5})
        ),
    }

    return {"so": so, "mo": mo, "lon": lon, "colon": colon}
