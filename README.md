# NoisyVisualisation

Evolutionary search on deterministic and noisy optimisation problems, plus tools for looking at how
that search behaves: search trajectory networks (STNs), local optima networks (LONs), constrained
LONs (CoLONs), a Dash dashboard and an MLflow browser.

- [Overview](#overview)
- [Repository architecture](#repository-architecture)
- [Runtime / container architecture](#runtime--container-architecture)
- [Installation and the editable-install model](#installation-and-the-editable-install-model)
- [Starting and managing the services](#starting-and-managing-the-services)
- [Running experiments](#running-experiments)
- [Configuration system](#configuration-system)
- [Creating a new experiment config](#creating-a-new-experiment-config)
- [Results, MLflow and data locations](#results-mlflow-and-data-locations)
- [Fast storage (NVMe) and CWD-sensitive paths](#fast-storage-nvme-and-cwd-sensitive-paths)
- [Extending the codebase](#extending-the-codebase)
- [Architecture / layering rules](#architecture--layering-rules)
- [Tests and reproducibility contracts](#tests-and-reproducibility-contracts)
- [Known and intentionally deferred issues](#known-and-intentionally-deferred-issues)
- [Environment lock and recovery](#environment-lock-and-recovery)
- [Development workflow](#development-workflow)

---

## Overview

The project does four things:

1. **Runs search algorithms.** These are DEAP-based evolutionary and estimation-of-distribution
   algorithms: (μ+λ) EAs, PCEA, UMDA, compact GA, SEMO, MO-UMDA and NSGA-II. They run on
   combinatorial problems (OneMax, Jump, 0/1 knapsack, including penalty and multi-objective
   variants) and on continuous problems (Rastrigin, bi-Rastrigin, Ackley). Each problem comes with
   several models of *prior* and *posterior* noise.
2. **Executes experiments.** Hydra configs drive the experiments, many seeds run in parallel
   processes, and results go to a pickle "warehouse" and to MLflow.
3. **Builds landscape networks.** It builds LONs by basin hopping, and CoLONs under Deb's
   constraint-handling order with feasibility annotations. Optional compression merges optima of
   similar fitness.
4. **Visualises and analyses the results:**
   - `noisyvis-dashboard` (port 8050): STN/LON graphs, Pareto fronts, performance plots and
     misjudgement analysis.
   - `noisyvis-app` (port 8051): browses MLflow experiments and runs.
   - `mlflow-ui` (port 5000): the stock MLflow UI.

**Quick start** (host, repository root; the containers are normally already running):

```sh
docker ps --filter name=evovis-runner-1 --filter name=noisyvis --filter name=mlflow-ui

# a single-objective sweep (see "Running experiments" for what this runs)
docker exec -it evovis-runner-1 python run.py --config-name SingleObjective/Continuous/Rastrigin2D/1p1ea

# a constrained LON
docker exec -it evovis-runner-1 python run_colon_parallel.py \
    --config-path configs/LONs/test01_SO_KP_LON --config-name kp10_colon
```

Then open `http://<host>:8050` (dashboard), `http://<host>:8051/experiments` (MLflow browser) or
`http://<host>:5000` (MLflow UI).

---

## Repository architecture

```
NoisyVisualisation/
├── run.py                  single-objective runner         (Hydra; nested --config-name supported)
├── run_mo.py               multi-objective runner          (Hydra; nested --config-name supported)
├── run_lon.py              LON runner, strictly sequential (Hydra; needs --config-path)
├── run_lon_parallel.py     LON runner, process-parallel    (Hydra; needs --config-path)
├── run_colon_parallel.py   CoLON runner, process-parallel  (Hydra; needs --config-path)
├── run_batch.py            run every SO config in a directory tree through run.py
├── run_colon_batch.py      run every CoLON config in one directory through run_colon_parallel.py
├── pyproject.toml          package metadata only: no dependencies, two console scripts
├── src/noisyvis/           the Python package (editable-installed; see below)
├── configs/                Hydra experiment configs
├── instances/knapsack/     knapsack problem instances and their optima
├── tests/                  reproducibility harness and architecture gates (see tests/README.md)
├── docker/                 Dockerfile, entrypoint, four Compose files, env-lock/
├── env/environment.yml     the conda specification the image was built from
├── assets/                 Dash CSS/JS; never actually served (see "Known issues")
└── data/, fast_storage/, outputs/, plots/   generated at runtime; gitignored
```

The five root scripts are thin. Each one sets the MLflow tracking URI at import, declares
`@hydra.main` and calls one workflow function in `noisyvis.experiments`.

### The `noisyvis` package

```
src/noisyvis/
├── algorithms/     single_objective.py, multi_objective.py, operators.py
├── problems/       onemax.py, jump.py, knapsack.py, knapsack_mo.py, continuous.py,
│                   instances.py (loaders), constraints.py (violation functions)
├── networks/       lon.py (BinaryLON), colon.py (BinaryCoLON), compression.py
├── tracking/       logger.py (ExperimentLogger and the active-logger singleton)
├── experiments/    runner.py (SO/MO), lon_runner.py (LON/CoLON), payloads.py, tracking.py,
│                   hyperparams.py, config/{cli,primitives,workflows}.py
├── results/        paths.py, store.py (warehouse read/write), mlflow_query.py
├── common/         distance.py, embedding.py, geometry.py
├── analysis/       graph_stats.py, misjudgements.py
├── viz/            config.py, layout.py, styling.py, traces.py, graph/{stn,lon}.py,
│                   plots/{registry,base,lon_stats}.py, plots/pareto/*, plots/performance/*
├── dashboard/      app.py (entry), instance.py, data.py, columns.py, components.py, tables.py,
│                   helpers.py, layout/*, callbacks/*
└── mlflow_app/     app.py (entry), pages/mlflow_browser.py
```

| Package | Responsibility |
|---|---|
| `noisyvis.algorithms` | Search algorithms. `single_objective.py` holds the SO base class `OptimisationAlgorithm` and `MuPlusLamdaEA` (+ `_forgetful`, `_estimated`), `PCEA`, `UMDA` (+ `_estimated`) and `CompactGA`. `multi_objective.py` holds its own `OptimisationAlgorithm` plus `SEMO`, `MoUMDA`, `MoUMDA_noDuplicates`, `MoUMDA_ParetoArchive` and `NSGA2`. `operators.py` holds attribute generators (`binary_attribute`, `Rastrigin_attribute`) and bit-level operators. The package namespace is built by star imports and is the lookup table for `problem.attr_function`. |
| `noisyvis.problems` | Fitness functions grouped by family, the knapsack instance loader (`load_problem_KP`), instance statistics and `knap_violation`. `__init__.py` re-exports every evaluator explicitly, because runners look up `problem.fitness_fn` by name in this namespace. |
| `noisyvis.networks` | Landscape-network builders. `BinaryLON` builds iterated-local-search LONs. `BinaryCoLON` builds constrained LONs with feasibility, neighbour-feasibility and visit counts. `compress_lon_aggregated` merges optima whose fitness lies within an accuracy threshold. |
| `noisyvis.tracking` | `ExperimentLogger`, with an in-memory or LMDB fit-history backend. It records every noisy evaluation and every generation so that STN trajectories can be rebuilt. The module-level active-logger singleton is how fitness functions reach it. |
| `noisyvis.experiments` | Orchestration behind the root scripts. It resolves configs (`config/workflows.py`), runs seeds in process pools, builds per-seed STN payloads, logs parent and child MLflow runs, and appends to the warehouse. It also holds `hyperparams.py`, the helpers configs call through `_target_`, and `config/cli.py`, the nested `--config-name` support. |
| `noisyvis.results` | Where data lives (`paths.py`: every location anchored to the repository root, overridable with `NOISYVIS_ROOT`). Reads and writes the warehouse pickles (`store.py`) and lists MLflow experiments and runs (`mlflow_query.py`). |
| `noisyvis.common` | Pure helpers shared by `viz`, `analysis` and `dashboard`: Hamming/Euclidean distances and solution keys, MDS/landmark embeddings, Bézier edge geometry. |
| `noisyvis.analysis` | Numbers derived from results and display graphs. `graph_stats.py` covers LON statistics and correlations; `misjudgements.py` covers noise-induced wrong-direction steps. It computes and never renders. |
| `noisyvis.viz` | Dash-free visualisation. `config.py` defines `PlotConfig`. `graph/stn.py` and `graph/lon.py` fill one shared `networkx.MultiDiGraph`. `layout.py` positions nodes (MDS, t-SNE, Kamada–Kawai, …), `styling.py` sizes and colours them, and `traces.py` turns them into Plotly traces. `plots/` holds pure 2-D Plotly figures: `pareto/` (12 plot types, reached through `plots/registry.py`), `performance/` (line and box plots) and `lon_stats.py`. |
| `noisyvis.dashboard` | The Dash STN/LON dashboard. `instance.py` owns the single `app`. `data.py` loads the warehouse once at import. `layout/` builds the page (`main_layout.py`, `components.py`, `stores.py` for `dcc.Store` ids, `styles.py`). `callbacks/` holds one module per callback group (`schematic`, `selection`, `performance`, `graph_data`, `visualization`, `pareto`); `app.py` imports them in that registration order. `components.py` and `tables.py` build Dash tables. `app.py` is the entry point and must never be imported by anything else. |
| `noisyvis.mlflow_app` | A small multi-page Dash app (`use_pages=True`) whose `/experiments` page lists the experiments and runs in `data/mlruns`. |

**Data flow.**

- **SO:** `run.py` → `experiments.runner` → algorithm + `ExperimentLogger` → per-seed payload
  pickle (`data/temp/payloads/`) plus a results row. The row goes to an MLflow child run and is
  appended to `data/warehouse/algo_results.pkl`, which the dashboard loads.
- **LON and CoLON:** the runner → `experiments.lon_runner` → `networks` builder per seed → merged
  network → one row per compression setting → `data/warehouse/lon_results.pkl` and MLflow.

---

## Runtime / container architecture

Four independent Compose projects, one per file, all defined in `docker/`. The three
`noisyvis:latest` services bind-mount the repository at `/workspace`, use
`/workspace/docker/entrypoint.sh` as their entrypoint and run with `working_dir: /workspace`.

| Container | Compose project | File | Service | Port | Command | Extra mounts / env |
|---|---|---|---|---|---|---|
| `evovis-runner-1` | `evovis` | `docker/compose.evovis.yaml` | `runner` | none | `sleep infinity`: you `docker exec` into it | `/mnt/nvme` → `/workspace/fast_storage`; `MLFLOW_TRACKING_URI`, `GIT_PYTHON_REFRESH=quiet` |
| `noisyvis-dashboard` | `dashboard` | `docker/compose.dashboard.yaml` | `dashboard` | 8050 | `python -m noisyvis.dashboard.app` | `PYTHONUNBUFFERED=1` |
| `noisyvis-app` | `app` | `docker/compose.app.yaml` | **`dashboard`** (sic) | 8051 | `python -m noisyvis.mlflow_app.app` | — |
| `mlflow-ui` | `mlflow` | `docker/compose.mlflow.yaml` | `mlflow-ui` | 5000 | `mlflow ui --backend-store-uri /workspace/data/mlruns` | host `data/mlruns` only; image `bitnami/mlflow:3.2.0` |

All four use `restart: unless-stopped`. The runner is not an HTTP service. The other three serve
HTTP on the ports shown.

**The separation is intentional.** The runner, the dashboard and the MLflow browser have
independent lifecycles. You can restart or recreate the dashboard to pick up new results or code
without touching a long experiment running in `evovis-runner-1`, and the other way round.
Consolidating the Compose files was considered and is not planned: merging projects would change
container names and networking for no functional gain.

Every data path is a host bind mount, so the containers share `data/` and `fast_storage/` with the
host and with each other.

---

## Installation and the editable-install model

There is no separate installation step. The image `noisyvis:latest` contains a micromamba
environment called **`exp`**, which owns all scientific dependencies (`env/environment.yml`,
pinned in `docker/env-lock/`). The repository is not baked into the image; it is bind-mounted at
`/workspace`, and the package source lives at `/workspace/src/noisyvis`.

At every container start, `docker/entrypoint.sh` runs:

```sh
micromamba run -n exp python -m pip install \
    -e /workspace --no-deps --no-build-isolation --quiet --disable-pip-version-check
exec micromamba run -n exp "$@"
```

- **`-e` (editable).** Python imports `noisyvis` from the live `/workspace/src` tree through a
  `.pth` file. Editing source on the host is visible in every container without rebuilding or
  reinstalling. Long-running processes such as the dashboard must still be restarted to import
  changed code.
- **`--no-deps`.** `pyproject.toml` deliberately declares `dependencies = []`, and this flag also
  stops pip from resolving, upgrading or replacing anything conda installed.
- **`--no-build-isolation`.** Uses the `setuptools` already in `exp` instead of creating a fresh
  build environment, which would need network access.
- The install runs on every start, so a `pyproject.toml` change takes effect on the next restart.
  `set -eu` makes a failed install fatal, so a service never starts with a broken import state.

`pyproject.toml` also defines two console scripts: `noisyvis-dashboard`
(`noisyvis.dashboard.app:main`) and `noisyvis-mlflow` (`noisyvis.mlflow_app.app:main`). The Compose
services use `python -m …` instead.

**An image rebuild is not needed for ordinary development.** It is only relevant for dependency,
`environment.yml`/lock, system-package or `Dockerfile` changes; see
[Environment lock and recovery](#environment-lock-and-recovery).

---

## Starting and managing the services

Run these from the repository root on the host. Always pass the project name (`-p`) and file
(`-f`) together, so each command targets exactly one project. Compose prints a harmless warning
that the `version` attribute is obsolete.

```sh
# status
docker compose ls
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
docker compose -p evovis    -f docker/compose.evovis.yaml    ps
docker compose -p dashboard -f docker/compose.dashboard.yaml ps
docker compose -p app       -f docker/compose.app.yaml       ps
docker compose -p mlflow    -f docker/compose.mlflow.yaml    ps

# runner: start / restart / recreate
docker compose -p evovis -f docker/compose.evovis.yaml up -d runner
docker compose -p evovis -f docker/compose.evovis.yaml restart runner
docker compose -p evovis -f docker/compose.evovis.yaml up -d --force-recreate runner

# dashboard (8050)
docker compose -p dashboard -f docker/compose.dashboard.yaml up -d dashboard
docker compose -p dashboard -f docker/compose.dashboard.yaml restart dashboard
docker compose -p dashboard -f docker/compose.dashboard.yaml up -d --force-recreate dashboard

# MLflow browser app (8051); note that its service is also called "dashboard"
docker compose -p app -f docker/compose.app.yaml up -d dashboard
docker compose -p app -f docker/compose.app.yaml restart dashboard

# MLflow UI (5000)
docker compose -p mlflow -f docker/compose.mlflow.yaml up -d mlflow-ui
docker compose -p mlflow -f docker/compose.mlflow.yaml restart mlflow-ui

# logs
docker logs -f noisyvis-dashboard
docker logs -f noisyvis-app
```

> **Restarting or recreating `evovis-runner-1` kills every experiment running inside it.** Check
> with `docker top evovis-runner-1` first. The dashboard and the MLflow apps can be restarted at
> any time.

Notes:

- The dashboard reads the warehouse once, at start-up. **Restart the dashboard to see new results.**
  Start-up takes a while because the warehouse is large.
- `mlflow-ui` shows `(unhealthy)` in `docker ps` because its healthcheck calls `curl`, which the
  bitnami image does not contain. The UI itself serves normally.
- The `mlflow` project was originally created by Portainer, so `docker compose ls` may still list a
  Portainer path as its config file. `docker/compose.mlflow.yaml` matches the running container and
  is the file to manage it with.
- Both Dash apps run with `debug=True` (Werkzeug debugger on error pages).

---

## Running experiments

All experiments run **inside `evovis-runner-1`, from `/workspace`**. `docker exec` starts in
`/workspace` by default, which matters: see
[CWD-sensitive paths](#fast-storage-nvme-and-cwd-sensitive-paths). `python` there is the `exp`
interpreter.

```sh
docker exec -it evovis-runner-1 bash                            # interactive shell in /workspace
docker exec -it evovis-runner-1 python run.py --config-name …   # one-shot

# long runs: detach, so closing the terminal does not kill the job
docker exec -d evovis-runner-1 sh -c \
  'mkdir -p data/run_logs && python run.py --config-name SingleObjective/Continuous/Rastrigin2D/1p1ea \
     > data/run_logs/rastrigin2d_1p1ea.log 2>&1'
docker exec evovis-runner-1 tail -f data/run_logs/rastrigin2d_1p1ea.log
docker top evovis-runner-1            # what is running
```

### `--config-name` vs `--config-path`, and the stale defaults (B4)

Every runner declares `@hydra.main(config_path="configs", config_name=…)`. The default names are
stale: `test1_kp_1p1` (in `run.py` and `run_mo.py`) and `test_lon_kp` (in the LON runners) do not
exist at the root of `configs/`.
**Never run a runner without `--config-name`: it fails with `MissingConfigException`.**
This is known issue B4, deliberately left unfixed. Always name the config explicitly:

- **`--config-name`** selects the *primary* config file, without `.yaml`. Hydra looks for it in the
  config directory, which is `configs/` unless `--config-path` changes it.
- **`--config-path`** replaces that config directory. A relative path is resolved against the
  directory containing the script, i.e. the repository root, not the shell's working directory.

The two runner families take configs differently:

| Runners | How to name a config | Why |
|---|---|---|
| `run.py`, `run_mo.py` | `--config-name <path under configs/ without .yaml>`, e.g. `SingleObjective/Continuous/Rastrigin2D/1p1ea`. Do **not** pass `--config-path`. | These scripts rewrite a nested name to a flat one. They temporarily symlink `configs/SingleObjective__Continuous__Rastrigin2D__1p1ea.yaml` → the real file, run, then delete the symlink. The config is then loaded as if it sat at the root of `configs/`. Keys stay at the top level, and composed configs can find `defaults/...` components. |
| `run_lon.py`, `run_lon_parallel.py`, `run_colon_parallel.py` | `--config-path configs/<dir> --config-name <file stem>` | These scripts have no flattening. Passing a nested `--config-name` loads the file but nests every key under the directory name (`LONs.…`), and the runner then fails. |

If a nested `run.py` run is killed hard, the temporary symlink can be left behind in `configs/`.
The next run with the same name then fails with `FileExistsError`. Delete the stray
`configs/*__*.yaml` symlink and rerun.

### Single-objective experiments (`run.py`)

```sh
# a standalone config: (1+1)EA on 2-D Rastrigin, noise 0..5 sweep, 30 seeds per point, parallel
python run.py --config-name SingleObjective/Continuous/Rastrigin2D/1p1ea

# a composed sweep experiment: {mu+1, 1+1, UMDA, PCEA} x {onemax100, onemax100_1qbitwise} x 14 noise levels
python run.py --config-name experiments/SO_performance/onemax_performance

# the same, with one-off overrides
python run.py --config-name SingleObjective/Continuous/Rastrigin2D/1p1ea run.num_runs=5 run.parallel=false
```

Most SO configs set `hydra.mode: MULTIRUN`, so a single invocation launches one Hydra job per
combination of sweep values, sequentially. Each job runs `run.num_runs` seeds (`run.seed`,
`run.seed+1`, …), in a process pool when `run.parallel: true`.

### Multi-objective experiments (`run_mo.py`)

```sh
python run_mo.py --config-name Multiobjective/MO_knapsack/kp100_semo
```

This takes the same nested `--config-name` form as `run.py`. MO configs use
`noisyvis.algorithms.multi_objective.*` targets and two-element `problem.weights`. For knapsack, the
runner computes `problem.ref_point` automatically.
`Multiobjective/MO_knapsack_test/mo_1p1ea.yaml` is known to be broken (B1).

### LON experiments (`run_lon.py`, `run_lon_parallel.py`)

```sh
# parallel (run.parallel / run.num_workers are read from the config)
python run_lon_parallel.py --config-path configs/LONs/LON --config-name kp10_lon
python run_lon_parallel.py --config-path configs/LONs/test01_SO_KP_LON --config-name test_lon_kp_parallel

# strictly sequential (ignores run.parallel); reproducible for a given seed
python run_lon.py --config-path configs/LONs/test01_SO_KP_LON --config-name test_lon_kp

# a quick check with fewer basin-hopping runs
python run_lon_parallel.py --config-path configs/LONs/LON --config-name kp10_lon run.num_runs=20
```

Each of the `run.num_runs` seeds builds one basin-hopping trajectory with `BinaryLON`. The runs are
merged into one network, and the runner writes one results row per entry in
`lon.compression_accs`.

### CoLON / constrained-LON experiments (`run_colon_parallel.py`)

```sh
python run_colon_parallel.py --config-path configs/LONs/test01_SO_KP_LON --config-name kp10_colon
python run_colon_parallel.py --config-path configs/LONs/penalty_colons --config-name kp5_v1_1
```

A CoLON config must set `problem.violation_fn`, the full dotted path of a violation function such
as `noisyvis.problems.constraints.knap_violation`. It must also set `problem.violation_params`.
`BinaryCoLON` raises `ValueError` without a violation function, so **plain LONs must use the LON
runners**. CoLON rows add `optima_feasibility`, `neighbour_feasibility`, `visit_counts` and
`visit_proportions`.

### Parallelism

| Runner | Switch | Worker count |
|---|---|---|
| `run.py` | `run.parallel` | `min(num_runs, cpu_count)`, or `run.override_max_workers` |
| `run_mo.py` | `run.parallel` | `ProcessPoolExecutor()` default (CPU count) |
| `run_lon_parallel.py`, `run_colon_parallel.py` | `run.parallel` (and `num_runs > 1`) | `run.num_workers`, default CPU count |
| `run_lon.py` | none; always sequential | — |

SO/MO results are sorted by seed, so parallel and sequential SO runs give the same rows. **Parallel
LON/CoLON output depends on worker completion order and is not reproducible**; use
`run.parallel=false` (or `run_lon.py`) when you need a reproducible network.

### Running a directory of configs (batch runs)

There are two batch drivers. Both launch the normal runner once per config as a **subprocess**,
**sequentially**. Neither adds any parallelism beyond what each config itself sets.

| | `run_batch.py` | `run_colon_batch.py` |
|---|---|---|
| Launches | `python run.py --config-name=<name>` (**SO only**) | `python run_colon_parallel.py --config-path=<abs dir> --config-name=<stem>` (**CoLON only**) |
| `--config-dir` | **required**, relative to `configs/` | optional, relative to `configs/`; default `LONs/penalty_colons` |
| File selection | `--pattern`, default `*.yaml`, **recursive** (`rglob`) | `--pattern`, default `*.yaml`, **this directory only** (`glob`) |
| Order | sorted by path | sorted by path |
| On failure | stops at the first failing config (exception) | stops at the first failure, unless `--keep-going`; then prints a failure summary |
| Nested files | flattens to a temporary `configs/<a>__<b>.yaml` symlink, removed afterwards | not needed (`--config-path`) |
| Other options | `--python` (interpreter), trailing Hydra overrides | `--python`, `--keep-going`, trailing Hydra overrides |
| Working directory | **must be `/workspace`**: the script is launched as the relative path `run.py` | **must be `/workspace`**: the script is launched as the relative path `run_colon_parallel.py` |

Trailing arguments after the options are passed unchanged to every run, as Hydra overrides.

```sh
# every SO config under configs/SingleObjective/Continuous/ (recursive: 8 files)
python run_batch.py --config-dir SingleObjective/Continuous

# only the 1+1 EA configs in that tree, with 5 seeds each
python run_batch.py --config-dir SingleObjective/Continuous --pattern '1p1ea.yaml' run.num_runs=5

# a directory of composed sweep experiments
python run_batch.py --config-dir experiments/SO_performance

# every CoLON config in configs/LONs/penalty_colons (the default), carrying on past failures
python run_colon_batch.py --keep-going
python run_colon_batch.py --config-dir LONs/CoLON --keep-going
```

There is **no batch driver for `run_mo.py` or the plain-LON runners**. A shell loop does the same
job:

```sh
for f in configs/LONs/LON/*.yaml; do
  python run_lon_parallel.py --config-path configs/LONs/LON --config-name "$(basename "$f" .yaml)" || break
done
```

**Recommended batch workflow:**

1. Write one config and check it composes: `--cfg job`, see
   [Validating a config](#validating-a-config-before-a-long-run).
2. Run it once, small (`run.num_runs=2`, a separate `experiment_name=…`), and check the result in
   the dashboard or MLflow.
3. Copy it for each variant into **one directory**, e.g. `configs/experiments/<study>/`, giving
   each variant a meaningful `experiment_name` and, for LONs, a distinct `lon.name`.
4. Launch the batch detached (`docker exec -d …`, as above) and follow its log.
5. Monitor: `data/outputs/*.log` (Hydra job logs), MLflow on 5000/8051, and `docker top
   evovis-runner-1`. Restart the dashboard when the batch finishes.

Do not run two experiments that may finish at the same moment. Appending to the warehouse is an
unlocked read-concatenate-write (B7), so simultaneous appends can lose rows.

---

## Configuration system

### How Hydra is used here

- Every runner is a `@hydra.main(version_base=None, config_path="configs", …)` script (Hydra 1.3.2).
  Hydra does not change the working directory.
- A config is plain YAML with a few fixed top-level sections (see the table below). The runners
  read these keys directly; there is no structured schema class.
- Before running, each workflow makes a resolved copy of the config and fills in derived values. It
  loads the problem instance, injects `items_dict`/`capacity`, and computes the mutation rate,
  population size, noise-dependent evaluation limit and problem ID. See
  `src/noisyvis/experiments/config/workflows.py`, which documents the differences between the
  SO, MO, LON and CoLON workflows.

**How names in a config reach code.** There are four channels:

| Key | Resolved by | Example |
|---|---|---|
| `_target_` | Hydra `instantiate`/`call`, with a full dotted path | `noisyvis.algorithms.single_objective.PCEA`, `noisyvis.experiments.hyperparams.inverse_n_mut_rate`, `noisyvis.problems.instances.load_problem_KP` |
| `problem.fitness_fn` | a bare name, looked up in the `noisyvis.problems` namespace | `eval_noisy_kp_v1`, `OneMax_fitness`, `rastrigin_eval` |
| `problem.attr_function` | a bare name, looked up in the `noisyvis.algorithms` namespace | `binary_attribute`, `Rastrigin_attribute` |
| `problem.violation_fn` (CoLON) | a full dotted path, imported with `importlib` | `noisyvis.problems.constraints.knap_violation` |

Use the canonical `noisyvis.*` paths in new configs. Some older MLflow artifacts still record
pre-reorganisation paths such as `src.problems.ViolationFunctions.knap_violation`. They are left as
recorded.

### Config directory structure

```
configs/
├── defaults/              reusable components (all "# @package _global_")
│   ├── mlflow.yaml          mlflow.tracking_uri
│   ├── algos/               1p1ea, mup1ea, pcea, umda, 1p1ea_forgetful, 1p1ea_estimated, mup1ea_forgetful
│   ├── algos_vpop/          1p1ea, mup1ea, pcea, umda with noise-dependent population sizes
│   ├── problems/            onemax100*, kp10_*, kp20_*, kp100_{cor,uncor}_* (+ knapsackPenalty/)
│   ├── run/                 run_budget_{onemax,kp10,kp20,PCEAconv_kp20}, run_100k_evals, run_1m_evals
│   ├── sweeps/noise/        0_10.yaml, 0_25_full.yaml (a noise sweep block)
│   └── multiobjective/      algos/, problems/, run/
├── experiments/<study>/   composed SO sweep experiments (run.py)
├── SingleObjective/       standalone SO configs (Continuous/…, Combinatorial/Knapsack/Largescale/…)
├── ParameterTesting/, jump/, 0TEST/   standalone SO configs (0TEST is scratch)
├── Multiobjective/        standalone MO configs (run_mo.py)
└── LONs/
    ├── LON/                 plain LONs (run_lon*.py)
    ├── test01_SO_KP_LON/    small LON and CoLON examples
    ├── CoLON/, CoLON/largeScale/, penalty_colons*/   CoLONs (run_colon_parallel.py)
```

The top-level directory name tells you which runner a config is for, but nothing enforces this. The
test suite classifies configs by content: `violation_fn` → CoLON; a `lon:` section → LON;
`noisyvis.algorithms.multi_objective.*` → MO; `noisyvis.algorithms.single_objective.*` → SO.

### Two config styles: standalone and composed

The repository uses **both** styles.

**(a) Standalone configs.** One YAML holds everything: `experiment_name`, `run`, `mlflow`,
`hydra`, `algo`/`lon` and `problem`, and there is no `defaults:` list. Every file under
`SingleObjective/`, `Multiobjective/`, `LONs/`, `jump/`, `ParameterTesting/` and `0TEST/` is of this
kind, as are the test configs in `tests/configs/`. What you read is what runs, apart from the
runtime resolution described above.

**(b) Composed sweep experiments.** Every file under `configs/experiments/` is of this kind, and
they are made of reusable components from `configs/defaults/`. A real example,
`configs/experiments/SO_performance/core.yaml`:

```yaml
# @package _global_
defaults:
  - /defaults/mlflow
  - _self_   # ensures overrides below take precedence

experiment_name: "performance_OneMaxBudget_core"
experiment_description: >
  Performance of 1p1ea, UMDA, and PCEA on onemax and knapsack problems ...

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      +defaults/run: run_budget_onemax
      +defaults/problems: onemax100, onemax100_1qbitwise, kp20_v1, ..., kp10_1qbitwise
      +defaults/algos: 1p1ea,umda,pcea
      problem.fitness_params.noise_intensity: 0,1,2,3,4,5,6,7,8,9,10
  sweep:
    dir: data/outputs
```

How one job of this sweep is composed:

1. `# @package _global_` puts the file's keys at the top level, as do all component files.
2. `defaults:` loads `/defaults/mlflow` (it contributes `mlflow.tracking_uri: data/mlruns`), then
   **`_self_`**, the file's own keys (`experiment_name`, `experiment_description`, `hydra`).
   Entries later in the list override earlier ones, so because `_self_` comes after
   `/defaults/mlflow`, the file's own values beat that component.
3. The sweeper turns each `+defaults/<group>: a, b, …` line into one job per value. The leading
   `+` *appends* a selection from that config group (`configs/defaults/run/…`,
   `configs/defaults/problems/…`, `configs/defaults/algos/…`) to the defaults list. Each supplies
   one top-level section: `run:`, `problem:` or `algo:`.
4. `problem.fitness_params.noise_intensity: 0,…,10` is an ordinary value override, swept.

The job's final config therefore has `experiment_*` and `hydra` from the file, `mlflow` from
`/defaults/mlflow`, and `run`, `problem` and `algo` from the three selected components, with
`noise_intensity` overridden. This example sweeps 1 × 14 × 3 × 11 = 462 jobs.

**Precedence, as verified with Hydra's compose API:**

- **(1)** `/defaults/mlflow` < `_self_`.
- **(2)** Selections appended with `+defaults/...` come *after* `_self_`, so where a component and
  the experiment file define the same key, **the component wins**.
- **(3)** Value overrides from the sweep or the command line (`run.num_runs=3`) are applied last and
  win over everything.

So to change a value that a component provides, override it with a value override (point 3). A key
written in the experiment file will not do it.

Components can also be selected on the command line of a standalone run. For example,
`+defaults/algos=pcea` works exactly as it does in the sweeper. This only works when the config is
loaded from the root of `configs/`, which `run.py`/`run_mo.py` guarantee by flattening nested names.

### Config fields by runner

Top-level sections, and what each runner reads. "—" means the runner ignores the key.

| Key | `run.py` (SO) | `run_mo.py` (MO) | `run_lon*.py` (LON) | `run_colon_parallel.py` (CoLON) |
|---|---|---|---|---|
| `experiment_name` | MLflow experiment | MLflow experiment | MLflow experiment | MLflow experiment |
| `experiment_description` | optional, stored in rows | optional | — | — |
| `mlflow.tracking_uri` | — (fixed `data/mlruns`) | — (fixed) | **used** (CWD-relative) | **used** (CWD-relative) |
| `hydra.run.dir` / `hydra.sweep.dir` | set to `data/outputs` (see below) | same | same | same |
| `run.seed`, `run.num_runs` | first seed, number of seeds | same | first seed, number of basin-hopping runs | same |
| `run.parallel` | yes | yes | parallel script only | yes |
| `run.num_workers` | — | — | parallel script only | yes |
| `run.override_max_workers` | optional | — | — | — |
| `run.eval_limit`, `run.max_gens` | evaluation budget; `max_gens` is logged | budget; `max_gens` = generation limit | — | — |
| `run.target_stop` | if true, stop at `problem.opt_global` | — | — (`BinaryLON` always stops at `opt_global`) | — (`BinaryCoLON` ignores the target) |
| `run.no_improve_limit`, `run.noisy_no_improve_limit` (or `*_fraction` of `eval_limit`) | optional stopping rules | — | — | — |
| `run.use_noise_dependent_eval_limit` + `run.eval_limit_for_noise` | per-noise-level budget map | same | — | — |
| `run.stop_without_improvement_in_gens`, `run.verbose_rate` | — | optional | — | — |
| `run.nvme_path` | optional LMDB fit-history location | — | — | — |
| `run.progress_print_interval`, `run.record_population` | optional | — | — | — |
| `algo.name`, `algo.type` | config labels (rows record the instance's own `name`/`type`) | same | — | — |
| `algo.init_args` | `_target_` + constructor hyperparameters | same | — | — |
| `algo.use_dynamic_mutation`, `algo.indpb_fn`, `algo.static_indpb` | sets `init_args.mutate_params.indpb` | same (or `mutate_kwargs`) | — | — |
| `algo.use_dynamic_pop_size`, `algo.pop_size_fn` | sets `init_args.pop_size` | same | — | — |
| `algo.starting_solution` | — | optional | — | — |
| `lon.name` | — | — | label only (MLflow run name, `LON_Algo` column) | same |
| `lon.pert_attempts`, `lon.n_flips_mut`, `lon.n_flips_pert`, `lon.compression_accs` | — | — | used | used |
| `problem.prob_name`, `prob_type`, `opt_goal` | metadata | metadata | metadata | metadata |
| `problem.dimensions`, `opt_global` | set inline, or filled by the loader | same | same | same |
| `problem.capacity`, `mean_value`, `mean_weight` | filled by the loader; OneMax defaults if missing | same | filled by the loader; defaults if falsy | same |
| `problem.PID` | optional (else `loader.filename`, else `<prob_name>_<dimensions>`) | same | **read directly; set it** | **read directly; set it** |
| `problem.loader` | `_target_` + `filename`, or `null` | same | same | same |
| `problem.fitness_fn`, `problem.fitness_params` | name + kwargs; the runner reads `fitness_params.noise_intensity` | same (a noise-free copy is used as "true" fitness) | name + kwargs | name + kwargs |
| `problem.attr_function`, `problem.weights` | gene generator; DEAP weights (`[1.0]` maximise, `[-1.0]` minimise) | same (two weights) | same | same |
| `problem.ref_point` | — | hypervolume reference (auto for knapsack) | — | — |
| `problem.violation_fn`, `problem.violation_params` | — | — | — | **required by the builder** |

The runner does **not** pass these keys to `BinaryCoLON`, so they currently have no effect:
`lon.include_start_nodes` and `lon.only_improving_perturbations` (which appear in CoLON configs),
and the commented-out `run.nvme_path` lines. The builder's defaults apply, `False`/`True`, which
happen to equal every current config's values.

### Command-line overrides

Any key can be overridden for **one invocation**; the YAML is not changed.

```sh
python run.py --config-name SingleObjective/Continuous/Rastrigin2D/1p1ea \
    run.num_runs=5 run.parallel=false experiment_name=rastrigin_smoke

python run.py --config-name experiments/SO_performance/core \
    run.num_runs=2 problem.fitness_params.noise_intensity=0,5      # replaces the swept values

python run_colon_parallel.py --config-path configs/LONs/penalty_colons --config-name kp5_v1_1 \
    run.num_runs=50 run.num_workers=8 lon.pert_attempts=500 'lon.compression_accs=[None]'

python run_lon.py --config-path configs/LONs/test01_SO_KP_LON --config-name test_lon_kp \
    run.num_runs=10 problem.loader.filename=f2_l-d_kp_20_878 problem.PID=f2_l-d_kp_20_878
```

- `key=value` changes an existing key.
- `+key=value` adds a key the config does not have, e.g. `+run.nvme_path=/workspace/fast_storage`
  or `+defaults/algos=pcea`.
- `++key=value` adds or overrides.
- `a=1,2,3` sweeps over the values, when running in `--multirun` / `hydra.mode: MULTIRUN`.
- Algorithm hyperparameters live under `algo.init_args`, e.g. `algo.init_args.mu=10` for
  `MuPlusLamdaEA`, `algo.init_args.pop_size=100` for `PCEA`/`UMDA`. If the config uses
  `use_dynamic_mutation`/`use_dynamic_pop_size`, the computed value wins over a hand-set `indpb`
  or `pop_size`.
- `'lon.compression_accs=[None]'`: in YAML the uncompressed setting is written as the string
  `"None"`; on the command line, quote the whole override so the shell leaves the brackets alone.

---

## Creating a new experiment config

The safest route is almost always to **copy the nearest existing config and change it**. Pick one
that uses the same runner, problem family and noise model. The step-by-step guide below explains
what each part means when you do.

### Step by step

1. **Choose the runner.** It determines the file's shape:
   - an algorithm on a problem → `run.py` (SO) or `run_mo.py` (MO);
   - a landscape network → `run_lon_parallel.py`/`run_lon.py` (LON) or `run_colon_parallel.py`
     (constrained, CoLON).
2. **Choose where to put it.**
   - An SO/MO config can go anywhere under `configs/`; name it with its path, e.g. `--config-name
     SingleObjective/MyStudy/onemax_1p1ea`.
   - A LON/CoLON config goes in a directory you will pass as `--config-path`.
   - Keep related variants in one directory so a batch script can run them all.
3. **Standalone or composed?**
   - For a one-off experiment, or any LON/CoLON, write a standalone file.
   - For a grid over algorithms × problems × noise levels that the existing `configs/defaults/`
     components already cover, write a composed file in `configs/experiments/<study>/`, modelled on
     `experiments/SO_performance/*.yaml`.
4. **Problem.** Set `problem.fitness_fn` to the name of a function exported by `noisyvis.problems`
   (see `src/noisyvis/problems/__init__.py`). Put its keyword arguments in `problem.fitness_params`.
   - For a knapsack instance, use `loader: {_target_: noisyvis.problems.instances.load_problem_KP,
     filename: <file in instances/knapsack/low-dimensional or large_scale>}`. Leave `dimensions`,
     `opt_global`, `capacity`, `items_dict` etc. as `null`: the loader fills them, and also
     injects `items_dict`/`capacity` into `fitness_params` (and `violation_params`).
   - Without a loader (OneMax, Jump, Rastrigin), set `loader: null`, and set `dimensions` and
     `opt_global` yourself.
   - Set `attr_function` (`binary_attribute` or `Rastrigin_attribute`) and `weights` (`[1.0]`
     maximise, `[-1.0]` minimise).
5. **Algorithm (SO/MO).**
   - Set `algo.init_args._target_` to the class's canonical path,
     `noisyvis.algorithms.single_objective.<Class>` or
     `noisyvis.algorithms.multi_objective.<Class>`.
   - The remaining keys of `init_args` are passed to the class's constructor, e.g. `mu`, `lam`,
     `mutate_function`, `mutate_params` for `MuPlusLamdaEA`, or `pop_size`, `select_size` for
     `UMDA`.
   - The runner supplies the problem-related arguments itself: solution length, weights, budget,
     fitness function and so on.
6. **Noise.** Noise is part of the fitness function:
   - *Posterior* noise adds Gaussian noise to the value: `OneMax_fitness`, `eval_noisy_kp_v1`,
     `rastrigin_eval`, …
   - *Prior* noise perturbs the solution before evaluating: `*_prior_bitflip*`, `*_1q_prior_bitwise`,
     `*_pq_prior_bitwise`, …
   - Set the level with `problem.fitness_params.noise_intensity`, which is commonly swept.
7. **Budget and stopping (SO/MO).**
   - Set `run.eval_limit` and `run.max_gens`.
   - SO has optional stopping rules: `run.target_stop` (needs `opt_global`) and
     `run.no_improve_limit`.
   - For a per-noise budget, set `run.use_noise_dependent_eval_limit: true` with
     `run.eval_limit_for_noise` (keys are the noise levels, as strings).
   - Then set `run.seed`, `run.num_runs` and `run.parallel`.
8. **LON/CoLON settings.**
   - `lon.name` is a label; make it distinctive, because it is how you tell variants apart in the
     dashboard.
   - Set `lon.pert_attempts`, `lon.n_flips_mut` (local-search neighbourhood),
     `lon.n_flips_pert` (perturbation strength) and `lon.compression_accs` (list of `"None"` and/or
     accuracies).
   - Set `run.num_runs`, `run.parallel` and `run.num_workers`.
   - For CoLON, also set `problem.violation_fn` and `problem.violation_params`.
9. **Results and MLflow.** Give the run a meaningful `experiment_name`; it becomes the MLflow
   experiment.
   - The SO/MO runners always log to `data/mlruns`. LON/CoLON runners use
     `mlflow.tracking_uri: "data/mlruns"`: include it in LON/CoLON configs.
   - The runners log `data/outputs/.hydra/config.yaml` as an MLflow artifact, so set `hydra.run.dir:
     data/outputs` (or, for sweeps, `hydra.sweep.dir: data/outputs`) as every existing config does.
10. **Validate it** before running at scale; see the next section.

### Example: a minimal single-objective config (`run.py`)

`configs/SingleObjective/MyStudy/onemax_1p1ea.yaml` (hypothetical location):

```yaml
experiment_name: "my_onemax_study"

run:
  max_gens: 1e6
  eval_limit: 20000
  target_stop: true          # stop when the true fitness reaches problem.opt_global
  seed: 1
  num_runs: 10
  parallel: true

hydra:
  run:
    dir: data/outputs        # the runner logs data/outputs/.hydra/config.yaml to MLflow

algo:
  name: OnePlusOneEA
  type: MuPlusLamdaEA
  use_dynamic_mutation: true           # indpb = 1/n, computed by indpb_fn
  indpb_fn:
    _target_: noisyvis.experiments.hyperparams.inverse_n_mut_rate
    n_items: null                      # filled from problem.dimensions
    noise: null
  init_args:
    _target_: noisyvis.algorithms.single_objective.MuPlusLamdaEA
    mu: 1
    lam: 1
    mutate_function: probFlipBit
    mutate_params:
      indpb: null                      # filled by use_dynamic_mutation

problem:
  prob_name: onemax
  prob_type: discrete
  opt_goal: maximise
  dimensions: 100
  opt_global: 100
  loader: null
  fitness_fn: OneMax_fitness
  fitness_params:
    noise_intensity: 1                 # posterior Gaussian noise, sd = 1
  attr_function: binary_attribute
  weights: [1.0]
```

```sh
python run.py --config-name SingleObjective/MyStudy/onemax_1p1ea
```

This relies on SO/MO-runner behaviour for OneMax: `capacity`, `mean_value` and `mean_weight` are
filled in when they are missing. For other loader-less problems, set them explicitly, as
`SingleObjective/Continuous/*` does. To sweep noise from the YAML, add a `hydra.mode: MULTIRUN`
block like `SingleObjective/Continuous/Rastrigin2D/1p1ea.yaml`.

### Example: a minimal CoLON config (`run_colon_parallel.py`)

`configs/LONs/MyCoLONs/kp10_colon_noisy.yaml` (hypothetical location):

```yaml
experiment_name: "my_colon_study"

run:
  seed: 1
  num_runs: 100              # basin-hopping runs merged into one network
  parallel: true
  num_workers: 16

mlflow:
  tracking_uri: "data/mlruns"   # read by the LON/CoLON runners (relative to /workspace)

hydra:
  run:
    dir: "data/outputs"

lon:
  name: CoLON_kp10_noisy_v1     # label: MLflow run name and LON_Algo column
  pert_attempts: 1500
  n_flips_mut: 1
  n_flips_pert: 2
  compression_accs: ["None"]

problem:
  PID: f1_l-d_kp_10_269         # the LON runners read PID directly
  prob_name: knapsack
  prob_type: discrete
  opt_goal: maximise
  loader:
    _target_: noisyvis.problems.instances.load_problem_KP
    filename: f1_l-d_kp_10_269
  fitness_fn: eval_noisy_kp_v1
  fitness_params:
    items_dict: null            # injected from the loader
    capacity: null
    noise_intensity: 1
  violation_fn: noisyvis.problems.constraints.knap_violation
  violation_params:
    items_dict: null            # injected from the loader
    capacity: null
  attr_function: binary_attribute
  weights: [1.0]
```

```sh
python run_colon_parallel.py --config-path configs/LONs/MyCoLONs --config-name kp10_colon_noisy
python run_colon_batch.py --config-dir LONs/MyCoLONs --keep-going      # every file in that directory
```

For a **plain LON**, delete `violation_fn` and `violation_params`, pick a fitness function such as
`eval_ind_kp` (deterministic, with `penalty: 1`), and run it with `run_lon_parallel.py`.

### Validating a config before a long run

1. **Print the composed config without running anything.** `--cfg job` makes Hydra compose and
   print the config and then exit:

   ```sh
   python run_colon_parallel.py --config-path configs/LONs/MyCoLONs --config-name kp10_colon_noisy --cfg job
   python run.py --config-name SingleObjective/MyStudy/onemax_1p1ea --cfg job
   ```

   For `run.py`/`run_mo.py`, this also exercises the temporary-symlink step.
2. **Check the names.** `fitness_fn` must appear in `src/noisyvis/problems/__init__.py`, and
   `attr_function` in `noisyvis.algorithms`. Every `_target_` must import, e.g.
   `python -c "import noisyvis.algorithms.single_objective as m; m.PCEA"`.
3. **Run the config-resolution gate.** It covers every file under `configs/` automatically:

   ```sh
   docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 \
     python -m pytest tests/test_config_resolution.py
   ```

   It takes a few minutes; see [Tests](#tests-and-reproducibility-contracts) for the one-time
   bootstrap.
4. **Do a small real run** with a throwaway experiment name, e.g. `run.num_runs=2
   run.parallel=false experiment_name=smoke_<study>`, plus a small `run.eval_limit` (SO/MO) or
   `run.num_runs`/`lon.pert_attempts` (LON). Note that this **does** write to the production
   warehouse and to MLflow, like every real run; the throwaway name is what lets you find the rows
   and ignore them.

---

## Results, MLflow and data locations

Everything below is gitignored (`data/`, `fast_storage/`, `outputs/`, `plots/`) and lives on the
host, shared with every container through the `/workspace` bind mount.

| Path (under `/workspace`) | Written by | Contents | Persistence |
|---|---|---|---|
| `data/warehouse/algo_results.pkl` | `run.py`, `run_mo.py` | every SO/MO results row ever run (a DataFrame pickle, appended to) | **primary persistent store**; read by the dashboard |
| `data/warehouse/lon_results.pkl` | all LON/CoLON runners | every LON/CoLON row (one per compression setting) | **primary persistent store**; read by the dashboard |
| `data/temp/payloads/stn_payload_seed<N>_sig<S>.pkl` | `run.py` | per-seed STN trajectory payloads; warehouse rows reference them by `payload_path` | persistent; needed to rebuild STNs |
| `data/temp/results.{csv,pkl}`, `data/temp/lon_results.{csv,pkl}` | all runners | the last run's results table | overwritten by every run |
| `data/mlruns/` | all runners | MLflow file store: experiments, parent/child runs, params, metrics, artifacts (results tables, payloads, `config.yaml`) | persistent |
| `data/outputs/` | Hydra | job output dirs: `.hydra/{config,hydra,overrides}.yaml`, `<script>.log`; sweeps use `data/outputs/<job-number>/` | reused and overwritten |
| `data/old_mlruns/` | — | the archived historical MLflow store (tens of GB); neither `mlflow-ui` nor `noisyvis-app` serves it | archival |
| `fast_storage/` | `run.py` with `run.nvme_path` | per-seed LMDB fit histories `lmdb_seed<N>/` | temporary: deleted after each seed |
| `plots/3dplot.html` | `noisyvis.viz.traces` | an exported figure | regenerated |
| `outputs/` | Hydra | default output dir for configs that do not set `hydra.run.dir` | disposable |

**Anchoring.**
- `noisyvis.results.paths` anchors the warehouse, `data/temp`, the SO/MO MLflow store,
  `instances/` and `plots/` to the repository root. `NOISYVIS_ROOT` overrides the root; the test
  harness uses this to isolate runs.
- Hydra's output directory and the LON configs' `mlflow.tracking_uri: "data/mlruns"` are
  **relative to the working directory**.

**How results reach the dashboard.**
- `noisyvis-dashboard` loads both warehouse pickles once, when it starts.
- After runs finish, restart it:
  `docker compose -p dashboard -f docker/compose.dashboard.yaml restart dashboard`.
- STN views read trajectory data that `run.py` merged from the payload pickles into the warehouse
  rows. LON views read `local_optima`, `fitness_values`, `edges` and, for CoLONs, the feasibility
  columns.

**How results reach MLflow.**
- Every entry point sets a local `file:` tracking URI explicitly. The runner container's
  `MLFLOW_TRACKING_URI` environment variable is therefore **not** used.
- SO: one parent run per job, named `SWEEP | PID=… | <algo.name>`, with one child run per seed
  carrying that seed's payload.
- MO: one run per job, named `algo.name`.
- LON/CoLON: one run per job, named after `lon.name`.
- `mlflow-ui` (5000) serves `data/mlruns` directly, and `noisyvis-app` (8051) lists the same store.
  Both see new runs without a restart.

**Caveats.**
- The runners log `data/outputs/.hydra/config.yaml` as the run's config artifact. In MULTIRUN
  sweeps, Hydra writes each job's config to `data/outputs/<job-number>/.hydra/` instead. The artifact
  logged for a sweep job is therefore whatever `data/outputs/.hydra/config.yaml` was left by the last
  single-run invocation, **not that job's config**. Rely on the MLflow parameters and the job
  directory instead.
- Historical runs and artifacts keep the module paths they were recorded with: `src.*` targets,
  `violation_fn: src.problems.ViolationFunctions.knap_violation`. They are deliberately not
  rewritten.
- Do not commit anything under `data/`, `fast_storage/`, `outputs/` or `plots/`.

---

## Fast storage (NVMe) and CWD-sensitive paths

The runner mounts the host's NVMe filesystem at `/workspace/fast_storage` (`/mnt/nvme` →
`/workspace/fast_storage` in `docker/compose.evovis.yaml`). Large SO runs keep their per-seed
noisy-fitness history in LMDB there instead of in RAM. Sixteen active configs, the large-scale
knapsack and continuous ones, enable it with:

```yaml
run:
  nvme_path: ${hydra:runtime.cwd}/fast_storage
```

`${hydra:runtime.cwd}` is **the directory the runner was launched from**. The Compose services keep
`working_dir: /workspace` so that `docker exec` starts there and this resolves to
`/workspace/fast_storage`, the NVMe mount. If you launch from anywhere else (`docker exec -w
/workspace/tests …`, or `cd` into a subdirectory first), the LMDB files go to
`<that dir>/fast_storage`. That is a directory on the ordinary root disk, or a missing one.

The same working-directory dependence applies to:

- Hydra output directories (`hydra.run.dir`/`hydra.sweep.dir: data/outputs`);
- the LON/CoLON `mlflow.tracking_uri: "data/mlruns"`;
- the runners' `data/outputs/.hydra/config.yaml` artifact;
- `run_batch.py`/`run_colon_batch.py`, which launch `run.py`/`run_colon_parallel.py` by relative
  path.

**Always launch experiments from `/workspace`.**

(`noisyvis.results.paths.FAST_STORAGE` / `NOISYVIS_FAST_STORAGE` exists but is not used by any
config or runner. The configs are the authority.)

---

## Extending the codebase

The characterization tests freeze the post-reorganisation inventory exactly. A **genuine addition**
will make some pinned tests fail by design:

- `test_core_library.py`: the public namespace of the algorithm modules;
- `test_problems_package.py`: the 31 problem definitions, 24 evaluators, definition locations and
  the 66-file instance manifest;
- `test_viz_package.py`: every visualisation definition, the plot registry, and dropdown values
  equal to registry keys;
- `test_dashboard_package.py`: the 41-callback contract and statement inventory.

When that happens, check that the failure names *only* what you added, then update that test's
pinned inventory deliberately, in a separate, reviewable commit. **Never edit
`tests/baselines/*.json` to accommodate an addition.** Adding code must not change existing seeded
results. If a baseline moves, you changed existing behaviour.

### Adding an algorithm

**Where it goes.** A single-objective algorithm goes in `src/noisyvis/algorithms/single_objective.py`,
subclassing that module's `OptimisationAlgorithm`. A multi-objective one goes in
`multi_objective.py`, subclassing that module's `OptimisationAlgorithm`. Operators (mutation,
crossover, attribute generators) go in `operators.py`.

**Contract for SO algorithms**, from `OptimisationAlgorithm` and `experiments/runner.py`:

- **Constructor.** `__init__(self, <your hyperparameters>, **kwargs)` must call
  `super().__init__(**kwargs)`. The runner passes these as kwargs: `sol_length`, `opt_weights`,
  `eval_limit`, `attr_function`, `starting_solution`, `target_stop`, `no_improve_limit`,
  `noisy_no_improve_limit`, `gen_limit`, `fitness_function` (a `(fn, params)` tuple),
  `progress_print_interval`, `record_population` and `nvme_path`. Your hyperparameters come from
  `algo.init_args`.
- **Set in the constructor:** `self.gens = 0`, `self.evals = 0`, `self.name`, `self.type`.
  Register operators on `self.toolbox`. Then call `self.initialise_population(n)`, which evaluates
  and counts `n` evaluations, followed by `self.record_state(self.population)`.
- **Implement `perform_generation()`.** Evaluate **only** through `self._evaluate_and_track(ind)`,
  and add to `self.evals` once per evaluation. Keep the current population in `self.population`.
  The inherited `run()` loop is: `stop_condition()` → `gens += 1` → `perform_generation()` →
  `record_state(population)`.
- **Stopping.** `stop_condition()` handles `eval_limit`, `target_stop`, `gen_limit` and the
  no-improvement limits, using the direction of `opt_weights[0]`. To add a criterion, override
  it, set `self.stop_trigger`, and fall back to `super().stop_condition()`, as `PCEA` does for
  convergence.
- **Logging.** `_evaluate_and_track` raises `RuntimeError` if the fitness function did not call
  `logger.log_noisy_eval(...)`; this is the `ExperimentLogger` contract. After `run()`, the runner
  reads `logger`, `seed_signature`, `name`, `type`, `gens`, `evals` and `stop_trigger`.
- **Randomness.** Use the global `random` and `numpy.random` modules only; the runner seeds both
  per seed. The base class draws `seed_signature` with one `random.randint` in `__post_init__`.
  Do not reorder anything in the base class.

**MO algorithms** get `true_fitness_function`, `ref_point`, `stop_without_improvement_in_gens` and
`verbose_rate` in addition. They do not use `ExperimentLogger` (B10). They must fill the Pareto and
hypervolume lists that `mo_algo_data_single` reads: `pareto_*`, `true_pareto_*`, the three
`*_hypervolumes` lists and `n_gens_pareto_best`.

**Worked flow:**

1. **Implement** the class next to its siblings.
2. **Export.** Nothing extra is needed: `noisyvis/algorithms/__init__.py` star-imports both modules.
   Configs name the class by its module path anyway:
   `_target_: noisyvis.algorithms.single_objective.MyEA`.
3. **Config.** Copy a neighbouring config (e.g. `defaults/algos/pcea.yaml` for a reusable component,
   or a standalone SO file). Change `algo.name`, `algo.type` and `algo.init_args`.
4. **Resolution test.** `tests/test_config_resolution.py` automatically routes and resolves the new
   config (by its `single_objective`/`multi_objective` target). Expect the
   `test_core_library.py` namespace pin to flag the new public name.
5. **Small run.** For example: `python run.py --config-name <your config> run.num_runs=2
   run.parallel=false run.eval_limit=2000 experiment_name=smoke_myea`. Check `stop_trigger` and
   `n_evals` in the output table.
6. **Full research run.**

No runner dispatch changes are needed: the SO/MO runners instantiate whatever `init_args._target_`
names.

### Adding a problem / fitness function

The code distinguishes four kinds of addition:

| You want… | Do this |
|---|---|
| **A new instance of an existing problem** (another knapsack file) | Add the instance to `instances/knapsack/low-dimensional/` (or `large_scale/`), and its optimum to the matching `*-optimum/` directory, under the same file name. Reference it with `problem.loader.filename` (and `problem.PID`). No code changes. The instance-manifest pin in `test_problems_package.py` will flag the new files. |
| **A new noise model for an existing problem** | Add a new evaluator next to its siblings, e.g. `eval_noisy_kp_<x>` in `problems/knapsack.py`. Export it from `problems/__init__.py`. Select it with `problem.fitness_fn`. |
| **A new problem family** | Create `problems/<family>.py` with its evaluators, and export them from `problems/__init__.py`. If solutions need a new gene type, add an attribute generator to `algorithms/operators.py` (it is star-exported into `noisyvis.algorithms`). |
| **A new constraint** (for CoLON) | Add `def my_violation(ind, **params) -> float` to `problems/constraints.py`, with `<= 0` meaning feasible. Reference it by dotted path in `problem.violation_fn`, and put its parameters in `violation_params`. |

**The evaluator contract:**

- Signature: `f(individual, <params>, noise_intensity=0, …)`. It returns a **tuple**, e.g.
  `(value,)` for SO or `(f1, f2)` for MO.
- If `get_active_logger()` returns a logger, the evaluator must call
  `logger.log_noisy_eval(original, noisy, true_fitness, noisy_fitness)` before returning.
  - `original` is the submitted solution.
  - For *prior* noise, `noisy` is the perturbed copy actually evaluated; for *posterior* noise it
    is the same object as `original`.
  - The SO algorithms refuse evaluators that do not log.
  - LONs run without a logger, so the call must be conditional, as in `OneMax_fitness`.
- Draw noise only from the global `random`/`numpy.random` modules.

**Export and lookup.** Runners resolve `problem.fitness_fn` with
`getattr(sys.modules['noisyvis.problems'], name)`. An evaluator not re-exported in
`problems/__init__.py` cannot be found.

**Instance loading.** `problem.loader` is called through Hydra. The config workflows unpack its
result as `n_items, capacity, optimal, values, weights, items_dict, info`, which is the shape
`load_problem_KP` returns. The workflows then inject `items_dict` and `capacity` into
`fitness_params` (and `violation_params`). A loader for a different family must return the same
7-tuple, or the problem must be configured inline with `loader: null`.

**Dashboard.** The noisy-LON node sampling in `viz/graph/lon.py` hard-codes a branch per knapsack
evaluator. A new noisy evaluator that should appear there needs its own branch, plus a dropdown
option in the dashboard layout.

**Tests.** `test_problems_package.py` pins every evaluator's definition, outputs and RNG
consumption, so it will flag the addition. Update those pins deliberately once you are satisfied
that nothing *existing* changed.

### Adding a visualisation / plot

**Where code belongs:**

- **Pure Plotly** figures and graph construction belong in `noisyvis.viz`:
  - `viz/plots/pareto/*`, `viz/plots/performance/*` and `viz/plots/lon_stats.py` for 2-D figures;
  - `viz/graph/{stn,lon}.py`, `layout.py`, `styling.py` and `traces.py` for the 3-D STN/LON network
    pipeline.
- These modules take data and config objects and return `go.Figure`s or traces. They must **not**
  import `dash` (layering rule 2) or `noisyvis.dashboard` (rule 1).
- **Dash** code belongs in `noisyvis.dashboard`: controls, dropdowns and tables
  (`layout/components.py`, `components.py`, `tables.py`) and callbacks (`callbacks/*.py`).
- **Derived numbers** belong in `noisyvis.analysis`.

**A new Pareto plot type** (the registry path):

1. Implement `plot_<name>(frontdata, series_labels, **kwargs) -> go.Figure` in a
   `viz/plots/pareto/` module, and export it from `viz/plots/pareto/__init__.py`.
2. Register it in `PARETO_PLOTS` in `viz/plots/registry.py` under a key. The dashboard's
   `callbacks/pareto.py` calls `get_pareto_plot(<dropdown value>)`.
3. Add a `{'label': …, 'value': <key>}` option to the Pareto plot-type dropdown in
   `dashboard/layout/components.py`. `test_viz_package.py` requires the dropdown values to equal
   the registry keys.
4. Restart the dashboard and check the plot. Then update the pinned registry and definition
   inventories in `test_viz_package.py` (and the dashboard layout pins, if flagged).

**A new performance plot.**
- `PERFORMANCE_PLOTS` exists in the registry, but `dashboard/callbacks/performance.py` imports the
  `plot2d_*` functions from `viz/plots/performance` directly.
- Add the function to `viz/plots/performance/`, export it, and call it from the performance
  callback, adding a control to the layout if needed.

**A change to the STN/LON network view.**
- The single orchestrator is `update_plot` in `dashboard/callbacks/visualization.py`. It parses
  the inputs into `PlotConfig`, then adds STN nodes and edges (`viz/graph/stn.py`) and LON nodes and
  edges (`viz/graph/lon.py`) to one graph. After that it computes positions (`viz/layout.py`),
  styles nodes (`viz/styling.py`) and builds traces (`viz/traces.py`).
- New visual options go in `viz/config.py`; the Dash controls feeding them go in the layout and
  callback.
- Node and edge attributes written by `viz/graph/*` are read downstream, so keep existing
  attribute names.

### Adding a new LON variant

A LON "variant" can mean several different changes. Each lives in a different place.

**The current pipeline:**

```
config (lon.*, problem.*)
  → root script (run_lon.py | run_lon_parallel.py | run_colon_parallel.py)        Hydra entry
  → noisyvis.experiments.lon_runner                                                orchestration
       run_lon_experiment | run_lon_parallel_experiment | run_colon_experiment
       resolve_lon_config | resolve_colon_config      (experiments/config/workflows.py)
       per seed: random.seed(seed); np.random.seed(seed)
                 _run_single_lon_worker → BinaryLON          (networks/lon.py)
                 _run_single_colon_worker → BinaryCoLON      (networks/colon.py)
       merge across seeds: _merge_lon | _merge_colon
       per compression setting: compress_lon_aggregated     (networks/compression.py)
       _log_and_persist_lon_df → data/warehouse/lon_results.pkl + MLflow
  → dashboard: LON table → callbacks/graph_data.update_lon_data → LON_data store
       → callbacks/visualization.update_plot → viz/graph/lon.add_lon_nodes/add_lon_edges
       → viz layout/styling/traces;  analysis/graph_stats (LON statistics);  viz/plots/lon_stats
```

**There is no LON dispatcher.**
- `lon.name` is only a label: the MLflow run name and the `LON_Algo` column.
- The builder is hard-coded: the LON workers call `BinaryLON` and the CoLON workers call
  `BinaryCoLON`.
- A genuinely new construction therefore needs runner code, not just config.

**Where each kind of change belongs:**

| Change | Where | Notes |
|---|---|---|
| Local-optimum definition / local search | inside the builder (`networks/lon.py` or `colon.py`) | Today this is best-improvement over all `n_flips_mut`-bit-flip neighbours, moving only on strict improvement. `improv_method='first'` is not implemented. |
| Neighbourhood | the builder's `generate_bit_flip_combinations` | size `C(n, n_flips_mut)` |
| Perturbation and acceptance | the builder loop | `random_bit_flip(n_flips=n_flips_pert)` from `algorithms/operators.py`; accept only if strictly better, which resets the attempt counter; stop after `pert_attempts` consecutive failures |
| Edge construction | the builder | LON: an edge from the previous to the current optimum when they differ, with weight = count. CoLON: self-loops are counted too, as are visit counts. |
| Constraint information | `networks/colon.py` | Deb's preorder via `violation_function`; per-optimum feasibility and neighbour-feasibility proportion |
| Noise | the fitness function (`problem.fitness_fn`) | A noisy evaluator makes the builder see noisy fitness. The dashboard's per-node noise *samples* are computed separately at display time in `viz/graph/lon.py`. |
| Sampling / aggregation / compression | `experiments/lon_runner.py` (`_merge_*`, compression rows), `networks/compression.py` | `run.num_runs` seeds; the first-seen fitness is kept per optimum; edge weights are summed |
| Which builder runs | the workers and workflow functions in `experiments/lon_runner.py` | hard-coded |

**Recipe for a genuinely different LON construction:**

1. **Implement the builder** as a new function in `noisyvis.networks`: a new module such as
   `networks/<variant>.py` or, if it is a constrained variant, alongside `colon.py`.
   - Keep the return contract of the builder it replaces. For `BinaryLON` that is `(local_optima:
     list[tuple], fitness_values: list[float], edges_list: list[(src, dst, weight)])`. For
     `BinaryCoLON` it is those three plus `optima_feasibility`, `edge_feasibility`,
     `neighbour_feasibility` and `visit_count`.
   - Reuse the DEAP `creator` set-up pattern at the top of the existing builders.
   - Import operators from `noisyvis.algorithms.operators`. **`noisyvis.algorithms` must never
     import `noisyvis.networks`**, or the import becomes a cycle.
   - Export the function from `networks/__init__.py`.
2. **Wire it into orchestration** in `experiments/lon_runner.py`. Prefer a **new workflow function
   and worker** over editing the existing ones.
   - Copy the smallest existing workflow (`run_lon_parallel_experiment`) and swap the worker call.
   - Keep reusing `resolve_lon_config`/`resolve_colon_config`, `_merge_lon`/`_merge_colon`,
     `_lon_compression_rows` and `_log_and_persist_lon_df`.
   - Add a thin root script (e.g. `run_<variant>_parallel.py`) modelled on `run_lon_parallel.py`.
     It must call `set_lon_module_tracking_uri()` at import and use `@hydra.main(version_base=None,
     config_path="configs", …)`.

   Adding a switch inside the existing workers instead (e.g. a new `lon.*` key, or forwarding the
   currently ignored `lon.include_start_nodes`/`only_improving_perturbations`) changes existing code
   paths. It is only acceptable if the default reproduces current behaviour exactly, including the
   order of RNG calls; `tests/baselines/lon.json` and `colon.json` will tell you.
3. **Configuration.** Keep the existing `lon:`/`problem:` shape and add only the new keys your
   builder needs. Give `lon.name` a distinctive value so rows are distinguishable in the dashboard's
   LON table.
4. **Preserve the output schema.**
   - Each row needs `problem_name`, `problem_type`, `problem_goal`, `dimensions`, `opt_global`,
     `PID`, `LON_Algo`, `n_flips_mut`, `n_flips_pert`, `compression_val`, `n_local_optima`,
     `local_optima` (tuples of genes), `fitness_values` (aligned) and `edges`
     (`{(src_tuple, dst_tuple): weight}`).
   - Optional per-node lists, aligned with `local_optima`, that the dashboard understands:
     `optima_feasibility`, `neighbour_feasibility`, `visit_proportions`.
   - Any new list-valued column should be added to `LON_HIDDEN_COLUMNS` in
     `dashboard/columns.py`, or it will appear in the LON table.
5. **Determinism.** Seed per run exactly as the existing workers do. Use only the global
   `random`/`numpy.random`. Remember that parallel merging (`as_completed`) is order-dependent: test
   reproducibility sequentially (`run.parallel=false`).
6. **Tests.**
   - `tests/test_config_resolution.py` routes configs by content (`violation_fn` → CoLON; a `lon:`
     section → LON runners). A new runner script must be added to its `RUNNERS`/`_classify` for its
     configs to be checked against it.
   - `test_core_library.py` pins the `BinaryLON`/`BinaryCoLON` bodies and the builders
     `lon_runner` imports, so editing an existing builder will fail there.
   - For a new builder, add a small sequential reproducibility case, following
     `tests/test_reproducibility.py` and a tiny config in `tests/configs/`. Record its new baseline
     once with `NOISYVIS_RECORD_BASELINES=1`, which never overwrites existing baselines.
7. **Validate.**
   - Run `--cfg job`, then a tiny sequential run (`run.num_runs=3 run.parallel=false
     lon.pert_attempts=100`, with a smoke `experiment_name`).
   - Inspect the new row in `data/temp/lon_results.csv`.
   - Restart the dashboard and view the network before any bulk run.

---

## Architecture / layering rules

`tests/test_layering.py` enforces two rules by scanning imports statically, without importing
anything:

1. **The science and visualisation packages never import the UI packages.** Nothing in
   `noisyvis.algorithms`, `problems`, `common`, `networks`, `tracking`, `analysis` or `viz` may
   import `noisyvis.dashboard`, `noisyvis.app` or `noisyvis.mlflow_app`, absolutely or relatively.
   `noisyvis.app` no longer exists; it stays in the rule as the MLflow app's former name.
2. **The visualisation layer never imports Dash.** Nothing in `noisyvis.viz` may import `dash` or
   any `dash_*` package.

Why the rules exist:
- The scientific code must be importable and testable without a web app. Rule 1 means experiments
  never pull in Dash or load the 1 GB+ warehouse.
- Figures should be reusable in notebooks, scripts and papers. Rule 2 keeps Plotly figure code free
  of Dash components.

`noisyvis.experiments` and `noisyvis.results` are not in rule 1's scanned set, but nothing in them
imports the UI packages either; keep it that way. `noisyvis.networks` may import
`noisyvis.algorithms`, never the reverse.

**Where should this code go?**

| You are writing… | Put it in |
|---|---|
| a search algorithm or operator | `noisyvis.algorithms` |
| a fitness function, loader or constraint | `noisyvis.problems` |
| a LON-style network builder or transformation | `noisyvis.networks` |
| per-evaluation or per-generation recording | `noisyvis.tracking` |
| config resolution, seeding, MLflow run structure, warehouse rows | `noisyvis.experiments` |
| file locations, warehouse read/write, MLflow queries | `noisyvis.results` |
| a statistic or derived number | `noisyvis.analysis` |
| a distance, embedding or geometry helper used by several layers | `noisyvis.common` |
| a Plotly figure, graph population, layout or styling | `noisyvis.viz` |
| a Dash control, table, layout piece or callback | `noisyvis.dashboard` (or `noisyvis.mlflow_app`) |
| an experiment definition | `configs/` |

---

## Tests and reproducibility contracts

The suite lives in `tests/`. Its detailed, maintained reference is **`tests/README.md`**, which
covers what each file gates, the expensive/targeted gate groups, and how baselines are recorded.

**Running the tests** (inside the runner):

```sh
# one-time, and again after the runner container is recreated: pytest lives outside the locked env
docker exec -w /workspace evovis-runner-1 \
  python -m pip install --no-deps --target /workspace/tests/.deps -r tests/requirements-test.txt

# full suite, about 5 minutes
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 python -m pytest tests

# a single file
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 \
  python -m pytest tests/test_layering.py
```

At Stage 12 acceptance the full suite was 296 tests: all pass apart from **one strict xfail**.

**Test categories:**

| Category | Files | What it protects |
|---|---|---|
| Golden / reproducibility | `test_reproducibility.py`, `baselines/{so_seq,so_par,mo,lon,colon}.json` | The real runners, run on small `tests/configs/*` configs in isolated temp roots, must reproduce recorded results exactly: SO sequential and parallel, MO, sequential LON, sequential CoLON. `run_lon.py` must also reproduce the LON baseline. |
| Config-workflow golden | `test_config_workflows.py`, `baselines/config_workflows.json` (frozen) | SO/MO/LON/CoLON config-resolution semantics |
| Config resolution | `test_config_resolution.py`, `known_broken_configs.py`, `legacy_paths.py` | Every `_target_`, `fitness_fn`, `attr_function` and `violation_fn` in `configs/` resolves through its runner's own lookup; no live config uses a removed legacy path |
| Layering | `test_layering.py` | The two rules above |
| Historical pickles | `test_historical_pickles.py`, `historical_fixtures.py` | Existing warehouse and payload pickles still unpickle, with the expected schema |
| Package characterization | `test_core_library.py`, `test_problems_package.py`, `test_viz_package.py`, `test_dashboard_package.py`, `test_experiments_package.py` | Frozen definitions, namespaces, figure digests, the Dash callback contract and entry points |
| Paths, URIs, CLI | `test_paths.py`, `test_tracking_uris.py`, `test_config_cli.py` | `results.paths` anchoring, each runner's MLflow store, the nested `--config-name` helper |
| Legacy-path assertions | inside the above | `src.*` compatibility modules and `run_helpers` are gone and cannot be imported |

**Isolation and the production guard.**
- The reproducibility tests run the real scripts in a temporary root: `NOISYVIS_ROOT` points there
  and MLflow gets a sentinel URI. A write fence forbids writes under `/workspace`.
- A session fixture snapshots `data/warehouse`, `data/mlruns`, `data/temp` and `configs/` before
  and after the session, and prints `UNCHANGED`/`CHANGED`. `CHANGED` means stop and investigate.

**Why RNG call order matters.**
- Seeded results depend on the exact **number and order** of `random.*` and `np.random.*` calls
  along the execution path. That includes module import order: the package `__init__` files of
  `noisyvis` and `noisyvis.experiments` are deliberately import-free for this reason.
- A "harmless" refactor that adds, removes or reorders a draw — an extra `random.choice`, a shuffled
  loop, a different sampling helper — changes every downstream result and fails the baselines.
- Do not casually restructure seeded code paths. If a behaviour change is intended, make it a
  deliberate, separately reviewed change that includes re-recording the affected baseline.

**The strict-xfail model.**
- `tests/known_broken_configs.py` lists configs that are known broken and deliberately not fixed.
  Each becomes a **strict** xfail: an unlisted failure fails the suite, and a listed config that
  starts passing (XPASS) also fails.
- The current list has one entry: **B1**, `Multiobjective/MO_knapsack_test/mo_1p1ea.yaml`, which
  targets the nonexistent `noisyvis.algorithms.multi_objective.MoMuPlusLamdaEA`.
- An xfail documents a known issue. It is **not** a way to quiet a new failure.

---

## Known and intentionally deferred issues

Some behaviours were deliberately preserved during the reorganisation. None of these is a new
regression; each is deferred as a separate behavioural decision.

| Issue | Status / what to do |
|---|---|
| **B1** `MoMuPlusLamdaEA` does not exist | The only strict xfail (`tests/known_broken_configs.py`). Do not use `Multiobjective/MO_knapsack_test/mo_1p1ea.yaml`. |
| **B4** Stale `@hydra.main` default config names | Always pass `--config-name` (and, for the LON runners, `--config-path`), as documented above. |
| MLflow browser with an empty store | `noisyvis.results.mlflow_query.list_experiments_df()` raises `KeyError: 'name'` when the store has no experiments, before the page's "No experiments found." message could be shown. At the time of writing `data/mlruns` holds no experiments (the history is archived in `data/old_mlruns`), so the `/experiments` page errors until the next run creates an experiment. |
| Dash assets are never served | `assets/` sits at the repository root, not next to the app module, so `GET :8050/assets/custom.css` returns **HTTP 500** (a Werkzeug `NotFound` raised inside the app). The custom CSS and MathJax have never loaded. Enabling them would be a visible UI change. |
| The dashboard loads data eagerly | `DashboardData.load()` runs once at import, which is why a restart is needed to see new results. |
| **B7** Unlocked warehouse append | Avoid concurrent runs that finish at the same time. |
| CoLON flags not forwarded | `lon.include_start_nodes` and `lon.only_improving_perturbations` are ignored; the builder defaults apply. |
| MULTIRUN config artifact | See [Results](#results-mlflow-and-data-locations): sweep jobs log a stale `config.yaml`. |
| **B8** Broken `__main__` demo | `algorithms/single_objective.py`'s demo imports a removed module. The module is never run as a script. |
| `mlflow-ui` "unhealthy" | The healthcheck needs `curl`, which the image lacks; the service works. |

The full register (B1–B10) and the risk log were kept in local, untracked planning notes.
`tests/README.md` and `tests/known_broken_configs.py` are the maintained, in-repository sources.

---

## Environment lock and recovery

- **Conda owns the scientific environment.** The image was built from `env/environment.yml` into
  the micromamba env `exp` (Python 3.11). `noisyvis` is added only as an editable install
  (`--no-deps`), so pip never re-resolves conda's packages.
- **Lock material, committed in `docker/env-lock/`:**

  | File | What it is |
  |---|---|
  | `conda-explicit.txt` | `micromamba env export -n exp --explicit` |
  | `micromamba-list.txt` | `micromamba list -n exp`, showing which packages came from conda and which from PyPI |
  | `pip-freeze.txt` | `pip freeze --all` |
  | `python-runtime.txt` | interpreter details |

  `pip-freeze.txt` is **reference material, not a reinstall script**. Most of its entries are
  conda-provided packages recorded with build-time `file://` origins; a bulk `pip install -r` would
  fail or clobber conda. To rebuild:
  - recreate the environment from `conda-explicit.txt`;
  - install only the PyPI-channel packages from `micromamba-list.txt`, pinned, with `--no-deps`;
  - leave `noisyvis` to the entrypoint.

  To check the live environment still matches the lock, compare `pip freeze --all` and
  `micromamba list -n exp` in the runner against these files. The only expected difference is the
  editable `noisyvis` entry.
- **Recovery image.**
  - `noisyvis:pre-reorg` is the pre-reorganisation image, currently the same image as
    `noisyvis:latest`. **Never overwrite it.** An older `noisyvis:pre-sklearn-1.8` tag also exists.
  - A `docker save` archive of the image (about 8.5 GiB), the original lock files and the container
    `inspect` state are kept outside Git on the NVMe filesystem, at
    `/mnt/nvme/backups/docker-images/noisyvis-pre-reorg/`. A data-warehouse backup is kept alongside
    them. This is a separate-disk local snapshot, not an off-host backup.
  - To roll back: `docker tag noisyvis:pre-reorg noisyvis:latest`, then recreate the three
    `noisyvis` services with the commands in [Starting and managing the
    services](#starting-and-managing-the-services). If the tag is ever lost, restore the image with
    `docker load < …/noisyvis-pre-reorg.tar.gz`.
- **Status.** The current runtime and recovery image are validated. A clean rebuild from the
  captured lock remains an optional future reproducibility verification step. It has not been
  performed, and any such rebuild must use a new tag.

---

## Development workflow

1. **Edit on the host.** The editable install makes the change visible in every container
   immediately. Restart `noisyvis-dashboard` / `noisyvis-app` to reload app code. Code you run in
   the runner picks up changes on its next invocation.
2. **Add or update a config**, and check it with `--cfg job`.
3. **Run the targeted tests** for what you touched. For example:
   - `tests/test_layering.py` for any import change;
   - `tests/test_config_resolution.py` for configs;
   - the fast `test_viz_package.py` subset from `tests/README.md` for visualisation;
   - `tests/test_dashboard_package.py` for the dashboard.
4. **Run a small experiment** with a throwaway `experiment_name`.
5. **Inspect the result**: the output table in the terminal, then MLflow (5000/8051), then the
   dashboard after a restart.
6. **Before structural changes**, run the full suite. Run it twice for anything touching seeded
   code, because a single pass cannot tell determinism from luck.
7. **Commit deliberately.** Keep behaviour changes, test-pin updates and refactors in separate
   commits.

**When is an image rebuild needed?** Only for new or changed dependencies, `env/environment.yml` or
lock changes, system packages, or `docker/Dockerfile` changes. Ordinary Python source, config and
test changes never need one. A `pyproject.toml` change only needs a container restart, because the
entrypoint reinstalls on start.
