"""MLflow helpers shared by the experiment workflows (plan Stage 7, risk R24).

Nothing here runs on import. Each entry point sets its tracking URI *explicitly*, at the moment the
old scripts did: the SO and MO wrappers call `set_so_mo_tracking_uri()` and the LON-family wrappers
call `set_lon_module_tracking_uri()` at module level, before Hydra runs `main`. The LON workflows
then set the config's `mlflow.tracking_uri` themselves, right after resolving the config. A workflow
must never inherit another workflow's URI, and a missed call fails loudly against the test harness's
unreachable sentinel instead of logging to the runner's configured server.

Parent-run lifecycles (`set_experiment`, `start_run`, parameters, artefacts, `end_run`) stay inside
the workflow functions in `runner.py` and `lon_runner.py`, because they interleave with computation
and persistence.
"""

from typing import Any, Dict

import mlflow

from noisyvis.results.paths import MLRUNS_DIR, PROJECT_ROOT


def set_so_mo_tracking_uri() -> None:
    """The SO/MO file store: <project root>/data/mlruns (results.paths), created if missing."""
    # explicit mlflow path
    mlruns_dir = MLRUNS_DIR                             # <project root>/data/mlruns (results.paths)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")

    print("RUN tracking:", mlflow.get_tracking_uri())


def set_lon_module_tracking_uri() -> None:
    """The LON-family import-time default: <project root>/data/mlruns (results.paths), not created.

    Each LON workflow replaces it with the config's `mlflow.tracking_uri` before logging anything.
    """
    # MLflow defaults (local file store under repo/data/mlruns)
    mlruns_dir = MLRUNS_DIR  # <project root>/data/mlruns (results.paths)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    print("RUN(LON) tracking:", mlflow.get_tracking_uri())


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
        payload_src = PROJECT_ROOT / row["payload_path"]
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
