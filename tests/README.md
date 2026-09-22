# Stage 1 — reproducibility harness and migration gate

This suite records the behaviour of the **current, unmodified** implementation so that every later
stage of the reorganisation (`plans/reorganisation_plan.md`, Revision 7) can be checked against it.
It is a behavioural gate, not a benchmark. The question it answers is:

> Given the same config and seed, does the code still produce the same scientific behaviour after
> restructuring?

Nothing here writes to production data. See "Isolation" below.

## Bootstrap the test runner

pytest is not part of the locked `exp` environment and must not be installed into it (risk R19: the
environment must keep matching `docker/env-lock/` except for the editable `noisyvis` entry). The test
runner therefore lives in the git-ignored `tests/.deps/`:

```sh
docker exec -w /workspace evovis-runner-1 \
  python -m pip install --no-deps --target /workspace/tests/.deps -r tests/requirements-test.txt
```

Re-run this after the runner container is recreated (Stages 3 and 4), because a `--target` install
lives in the bind-mounted repository but pip's metadata does not depend on the container.

**`tests/.deps` is deliberately not a standalone environment.** It contains only `pytest`, `pluggy`
and `iniconfig`. pytest's other runtime requirements — `packaging` (25.0) and `pygments` (2.19.2) —
come from the locked `exp` environment on purpose, so that the locked versions are never shadowed via
`PYTHONPATH`. `conftest.py` asserts both are importable and satisfy pytest's minimums. See the header
of `requirements-test.txt` for the full dependency analysis.

## Run the suite

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps \
  evovis-runner-1 python -m pytest tests
```

Stage 3 temporarily appended a bridge path (`PYTHONPATH=/workspace/tests/.deps:/workspace/src`).
From Stage 4 that is gone: `noisyvis` comes from the editable install's
`__editable__.noisyvis-0.1.0.pth`, which puts `/workspace/src` on `sys.path` in every interpreter in
the container. `PYTHONPATH` now carries only the test runner.

The harness strips `tests/.deps` from `PYTHONPATH` in every scientific subprocess, so experiment runs
see exactly the production environment.

## Stage 1 verification procedure

This is the sequence used to establish the gate, and the one to repeat at the end of every later
stage. Steps 1–3 are cheap and read-only; only step 4 writes baseline files.

### 1. Check the starting state

```sh
git status --short                                                    # expect only: ?? tests/
docker exec evovis-runner-1 bash -c 'ls -A /workspace/data/mlruns | wc -l'
```

Each session asserts that production data is unchanged, so it helps to know what that state is
before starting. `data/old_mlruns` is archival and deliberately outside the guard.

### 2. Infrastructure gates — no experiments run

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 \
  python -m pytest tests/test_layering.py tests/test_historical_pickles.py tests/test_config_resolution.py \
  tests/test_paths.py tests/test_config_workflows.py tests/test_experiments_package.py \
  tests/test_config_cli.py tests/test_core_library.py tests/test_problems_package.py \
  tests/test_viz_package.py tests/test_dashboard_package.py
```

Resolves every `_target_`, `violation_fn`, `fitness_fn` and `attr_function` in `configs/` through each
runner's namespace mechanism, loads the persisted artefacts, scans imports for the two layering rules,
checks the `noisyvis.results.paths` contract, replays the 26 frozen config-resolution cases, checks
the `noisyvis.experiments` package and CLI-helper contracts, checks the core-library contracts
that Stage 8 moves, checks the problems-package contracts that Stage 9 moves, checks the
visualisation/plotting contracts that Stage 10 moves, and checks the dashboard and MLflow-app
contracts that Stage 11 moves.
No experiment executes, so this is the quickest way to catch a broken import or a moved module.

**Expect:** `277 passed, 1 xfailed` in roughly 3.75 minutes (Stages 1–4: `219 passed, 3 xfailed`;
Stage 5: `220 passed, 2 xfailed`; Stage 6: `223 passed, 2 xfailed`, before the Stage 7 files existed;
Stage 7: `235 passed, 2 xfailed`, before `test_core_library.py` existed; Stage 8: `241 passed, 2 xfailed`,
before `test_problems_package.py` existed; Stage 9 Checkpoints 0–C: `253 passed, 2 xfailed`, before the
loader-anchoring test; Stage 9 complete: `254 passed, 2 xfailed`, before `test_viz_package.py` existed;
Stage 10 Checkpoints 0–B: `262 passed, 2 xfailed`, before Checkpoint C enforced layering rule 2;
Stage 10 complete: `263 passed, 1 xfailed`, before `test_dashboard_package.py` existed).
Most of the added time is the Stage 10 characterization; see "Stage 10 gate groups" and "Stage 11
gate groups" below when a checkpoint does not need all of it.

### 3. Harness smoke — before anything is recorded

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 \
  python -m pytest tests/test_reproducibility.py -k so_seq --durations=10
```

Runs the real `run.py` twice inside temp roots. On a tree with no baselines yet, the **expected
result is exactly one failure**: `no baseline recorded for 'so_seq'`. That is the harness working
correctly while the gate refuses to silently skip a missing baseline.

### 3b. Inspect what would be frozen — recommended before recording

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps -i evovis-runner-1 python - <<'PY'
import sys, json
sys.path.insert(0, "/workspace/tests")
import conftest
from harness.run_isolated import run_isolated

before = conftest.production_snapshot()
res = run_isolated("run.py", "so_onemax", "so", overrides=["run.parallel=false"])
print(json.dumps(res.extracted["meta"], indent=2))
for seed, case in sorted(res.extracted["cases"].items(), key=lambda kv: int(kv[0])):
    print(seed, case["n_evals"], case["stop_trigger"], case["final_fit"], len(case["rep_sols"]))
print("PRODUCTION UNCHANGED:", before == conftest.production_snapshot())
PY
```

pytest only reports pass or fail; this prints the actual values a baseline would capture. Use it to
confirm the runs are scientifically non-trivial before freezing them — several accepted solutions per
seed, non-empty LON/CoLON optimum and edge maps. A single-point trajectory makes a weak gate, and it
is far cheaper to retune a config now than to re-record later. Swap the arguments for the other
cases: `("run_mo.py", "mo_knapsack", "mo")`, `("run_lon_parallel.py", "lon_knapsack_noisy", "lon")`,
`("run_colon_parallel.py", "colon_knapsack_noisy", "colon")`.

Note `sys.path.insert` rather than `PYTHONPATH`: it keeps the child environment byte-identical to a
pytest-driven run. The manual guard snapshot keeps the isolation evidence even outside pytest.

### 4. Record the five baselines — once, on an unmodified tree

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps \
  -e NOISYVIS_RECORD_BASELINES=1 evovis-runner-1 python -m pytest tests/test_reproducibility.py
```

**Expect:** `6 passed`, with five `recorded baseline ...` lines in the observations block. Existing
baselines are never overwritten by this; that needs `NOISYVIS_RECORD_BASELINES=overwrite`, which
should be a deliberate decision, not a way to make a failing comparison pass.

### 5. Full suite, twice consecutively

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 python -m pytest tests
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 python -m pytest tests
```

**Expect:** `289 passed, 1 xfailed` both times (Stages 1–4: `225 passed, 3 xfailed`; Stage 5:
`226 passed, 2 xfailed`; Stage 6: `229 passed, 2 xfailed`; Stage 7: `247 passed, 2 xfailed`; Stage 8:
`253 passed, 2 xfailed`; Stage 9 Checkpoints 0–C: `265 passed, 2 xfailed`; Stage 9 complete:
`266 passed, 2 xfailed`; Stage 10 Checkpoints 0–B: `274 passed, 2 xfailed`; Stage 10 complete:
`275 passed, 1 xfailed`), about 300 seconds each.
Two consecutive identical runs
are the completion criterion: a single run cannot distinguish genuine determinism from luck.

### Reading the output

Every session ends with the guard block:

```
production-data guard (plan §5.5a)
  warehouse: UNCHANGED {...}
  mlruns: UNCHANGED {...}
  temp: UNCHANGED {...}
  configs/ entries: UNCHANGED (8 entries)
```

`CHANGED` anywhere means the session touched production data: stop and investigate rather than
re-running. When an isolated run fails, its temp root is **kept** and the path printed in the error,
so the Hydra output, MLflow store and payloads can be inspected; successful runs clean up after
themselves.

Summary of expected results:

| Step | Command | Expected |
|---|---|---|
| 2 | gates only | `277 passed, 1 xfailed` |
| 3 | `-k so_seq`, no baselines yet | 1 failed: `no baseline recorded` |
| 4 | record mode (Stage 1, historical) | `6 passed`, five baselines written |
| 5 | full suite ×2 | `289 passed, 1 xfailed` each |

## What each file gates

| File | Gate |
|---|---|
| `test_reproducibility.py` | The five baselines of §5.5: SO-seq, SO-par, MO, LON, CoLON; plus `run_lon.py` (inline sequential LON) reproducing the LON baseline |
| `test_config_resolution.py` + `known_broken_configs.py` | Every `_target_`, `violation_fn`, `fitness_fn` and `attr_function` in `configs/` still resolves (§5.6). From Stage 7 the `sys.modules` keys and `_import_from_dotted` are also read from the `noisyvis.experiments` modules a thin runner delegates to. From Stage 12 each config's runner routing is decided on canonical target names (`legacy_paths.py`) and pinned to be the same under either dotted-path spelling; no live config input (`configs/` including gitignored files, `tests/configs/`, the workflow cases) uses one of the five migrated legacy paths; and no runner loads a compatibility module while resolving |
| `test_config_workflows.py` + `config_workflow_cases.py` | 26 synthetic configs pin the SO, MO, LON and CoLON config-resolution semantics (§5.4) against the frozen golden `baselines/config_workflows.json`, recorded once from the pre-Stage-7 resolvers. From Stage 12 golden and outcomes are compared through `legacy_paths.canonicalise_tree`, so only the dotted-path spelling may differ; the golden itself is never rewritten, and the translation is pinned to exactly its 44 legacy path leaves; the resolution loads no compatibility module |
| `test_experiments_package.py` | `noisyvis.experiments.hyperparams` keeps the original `run_helpers` namespace and every configured hyperparams target resolves to its object, whichever spelling the config uses, without `/workspace` on `sys.path`; no Python source imports the root `run_helpers`; the compatibility modules Stage 12 removed (`src.algorithms.*`, `src.problems.*`, `run_helpers`) are not importable from a runner-like `sys.path`, and neither `src/src/` nor `run_helpers.py` exists; importing any `noisyvis.experiments` module sets no MLflow URI and creates no files |
| `test_tracking_uris.py` | Each entry point logs to its intended MLflow store (R24): SO/MO to `data/mlruns`, LON/CoLON to the config's `tracking_uri` |
| `test_config_cli.py` | The nested `--config-name` helper behind `run.py`/`run_mo.py`: argument rewriting, symlink target, cleanup, exception propagation (temporary config root only) |
| `test_core_library.py` | Core-library contracts that Stage 8 moves, pinned to frozen values from the pre-Stage-8 commit 465ca06: one active-logger singleton shared by set/get/clear; the configured `attr_function` names in the `noisyvis.algorithms` namespace; the D4 operator definitions every consumer reaches; every configured algorithm target resolves to its canonical module's object, whichever spelling the config uses, and the canonical modules keep the frozen forwarder namespace (B1 the only unresolvable target, compared in canonical spelling); the `BinaryLON`/`BinaryCoLON`/`compress_lon_aggregated` bodies. Location-agnostic, so it holds before and after each move (from Stage 9 Checkpoint 0 the fitness consumers are found through evaluator `__globals__`, not the `FitnessFunctions` path) Deliberately amended by the MO package split (MO refactor Stage 1): `noisyvis.algorithms.multi_objective` is now a package, so its frozen namespace is the package's `__all__` plus its four submodules (`MoUMDABase` added in Stage 3), the MO anchor (`SEMO`) may be defined in a submodule, and the MO D4 consumer is the package, which reaches no D4 name (the old module only imported them unused). |
| `test_problems_package.py` | Problems-package contracts that Stage 9 moves, pinned to frozen values captured from a `git archive` of the pre-Stage-9 commit 8424f5e under the runner's Python 3.11: the 19 configured `fitness_fn` names resolve through the dynamic namespace to unchanged definitions; the 31 top-level problem definitions (only `mean_weight` twice); the globals each of the 24 evaluators reads; every evaluator's output, log records and RNG consumption on fixed, fully isolated inputs; every configured problem path resolves to its canonical module's object, whichever spelling the config uses, with the frozen namespace; loader output for all 31 knapsack instances plus the stats/correlation helpers; the `knap_violation` clamp divergence; both `mean_weight` copies; the problem imports of `Dashboard.py` and `graph_builder.add_lon_nodes`; the 66-file instance-tree manifest and git tree; each definition in its pre- or intended post-split module; from Checkpoint A, the loader finding instances through `NOISYVIS_ROOT` (`INSTANCES_DIR / "knapsack"`) from an unrelated cwd, with the loaders' frozen hashes compared after reading `_KNAPSACK_DIR + '/x/'` back as the pre-Stage-9 literal. Location-agnostic across the Stage 9 checkpoints. Any PRE/post differential run must use separate fresh subprocesses, never two copies of `noisyvis` in one interpreter |
| `test_viz_package.py` | Visualisation/plotting contracts that Stage 10 moves, pinned to frozen values captured from a `git archive` of the pre-Stage-10 commit 4530785 under the runner's Python 3.11. From Checkpoint E the location contract is **post-only**: definitions are accepted solely at `noisyvis.viz.*`, `noisyvis.viz.graph.{stn,lon}`, `noisyvis.viz.plots.*`, `noisyvis.analysis.graph_stats` and `noisyvis.dashboard.components`; the pre-move packages `noisyvis.visualization` and `noisyvis.plotting` must neither be imported by any source file nor be importable at all, which is checked in a fresh subprocess. The pinned values themselves are unchanged. Contents: every top-level definition of the 19 visualisation/plotting modules by normalised AST and string-literal multiset, each existing exactly once in its pre- or intended post-move module; the plot registry's key order, callable identity, Dashboard dropdown values and aliases; every reachable Pareto plot with the Dashboard's own argument variants; both performance families including their missing-column fallbacks; the LON-stats figures, graph-statistics correlations and both Dash table builders; the real `update_plot` orchestration over one shared graph for M1–M10 and the nine-layout smoke matrix, pinned at three levels (graph immediately before layout, returned positions, rendered outputs); the Dashboard/DashboardHelpers/layout-components bodies with imports excluded; and the visualisation names those files import, resolved to objects. Location-agnostic across the Stage 10 checkpoints. `PYTHONHASHSEED` is pinned to 0 for the probe subprocess only, because `update_plot` iterates a `set` of node labels when building advanced-misjudgement traces; that pre-existing presentation-order nondeterminism is additionally covered by an order-insensitive `figure_semantic` digest over those traces alone. From Stage 11 Checkpoint 0 the three Dashboard-side contracts follow the files rather than their paths: `update_plot` and the helpers are found by what they define, and the two bodies are **reassembled** from wherever their statements now live, in their frozen PRE order, with the Stage-11 transformations inverted (`harness/dashboard_inventory.py`), so the expected `BODIES` and `IMPORTS` values are unchanged |
| `test_historical_pickles.py` + `historical_fixtures.py` | Existing persisted data still loads with the expected schema (§7.5); gates Stages 8 and 12 |
| `test_layering.py` | The two architectural rules of §5.1: rule 1 enforced from Stage 5, rule 2 from Stage 10 Checkpoint C, which split `lon_stats_plots.py` so no visualisation module imports Dash |
| `test_dashboard_package.py` + `harness/dashboard_inventory.py` | Dashboard and MLflow-app contracts that Stage 11 moves, pinned to frozen values captured from a `git archive` of the pre-Stage-11 commit fa8d250 under the runner's Python 3.11, twice, with identical reports. Contents: the 41-callback Dash contract — registration order, every Output/Input/State, `prevent_initial_call`, and the digest the live `/_dash-dependencies` serves; each callback's normalised AST and string literals, hashed under a fixed name so a B9 rename does not move the hash, with its name and defining module pinned separately; every module global each callback reads, resolved to the object it names; every top-level statement of the seven redistributed files, found exactly once at its pre- or intended post-move module; the app configuration, layout digest and 110 component ids; that `DashboardData.load()` runs exactly once per application import (R9); every callback driven over the real Dash HTTP protocol plus a chained run of the populated pipeline (selection → STN/LON stores → the one shared-graph orchestrator, I-10c); the four table builders' schemas and the column constants; the import surface — loaded third-party packages and every existing explicit import, which amendment A1 preserves, so the six wildcards are the only import change; the MLflow browser's page registry, dependencies, layout and `MLRUNS_DIR` equivalence; and the entrypoint rules — Compose commands resolve to modules that exist, `[project.scripts]` targets, and no module imports an entrypoint module. The probe writes its own synthetic warehouse into the harness temp root and lets the real loader read it through `NOISYVIS_ROOT`: nothing is monkeypatched and the production warehouse is never opened. Location-agnostic across the Stage 11 checkpoints The MO package split added the four `noisyvis.algorithms.multi_objective` submodules to `NEW_MODULES`. |
| `test_mo_algorithms.py` + `baselines/mo_algorithms.json` | Multi-objective algorithm contracts for the MO package split and the MoUMDA changes: every public MO name resolves at its flat `noisyvis.algorithms.multi_objective.<name>` path; every configured MO target instantiates and runs; MoUMDA and MoUMDA_ParetoArchive (with and without probability margins, archive fingerprinted every generation) and NSGA-II match the frozen pre-refactor characterisation, except that a run whose probability vector collapses must be exactly the baseline prefix through the first generation sampled from it; the archive's genotype-duplicate semantics; the probability-vector helpers and frozen legacy-wrapper outputs; the post-generation convergence stop and generic-criteria precedence; commit-on-success; and the duplicate-free MoUMDA contract (unique initial and generated populations, the pre-generation `probability_vector_converged` / `insufficient_unique_support` stops, prepared-vector reuse, construction errors) |
| `test_paths.py` | `noisyvis.results.paths` (§5.9): repository root by default from any cwd, `NOISYVIS_ROOT` override, no writes on import |
| `conftest.py` | Session-scoped production-data guard (§5.5a) |
| `harness/` | Isolated subprocess runner, write fence, extractors, canonical comparison |

## Stage 10 gate groups

The Stage 10 characterization is expensive, so it is grouped rather than weakened: the assertions are
unchanged, only *when* each group runs differs. An ordinary `python -m pytest tests` still runs all of
it, and every checkpoint that can affect a contract runs the group that covers it.

`test_viz_package.py` spends its time in three independent probe subprocesses, one per fixture, so a
targeted selection really does skip the cost:

| Probe | Covers | Used by | Measured |
|---|---|---|---|
| `core` | definitions, registry, performance plots, LON stats/tables, Dashboard bodies, import surface | tests 1, 2, 4, 5, 7, 8 | ~3 s |
| `pareto` | all 12 Pareto registry keys, including the MDS/t-SNE/Isomap distance variants | `test_pareto_figures_pinned` | ~41 s |
| `heavy` | M1–M10 through the real `update_plot`, plus the nine-layout smoke matrix | `test_update_plot_mixed_pipeline_pinned` | ~92 s |

**FAST / TARGETED GATE** — roughly 10 s of visualisation cost:

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 \
  python -m pytest tests/test_viz_package.py \
  -k "definitions or registry or performance or lon_stats or bodies or imports"
```

**EXPENSIVE VISUAL-SEMANTICS GATE** — roughly 135 s:

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 \
  python -m pytest tests/test_viz_package.py::test_pareto_figures_pinned \
                   tests/test_viz_package.py::test_update_plot_mixed_pipeline_pinned
```

**FULL ACCEPTANCE GATE** — the focused list of step 2, then the full suite (twice at a stage boundary).

| Checkpoint | Gate |
|---|---|
| 0 | Full capture: both gates, plus two PRE-archive probe runs per section for the determinism gate |
| A (`plotting/` → `viz/plots/`) | Fast/targeted **and** the Pareto half of the expensive gate; no routine M1–M10 rerun |
| B (`statistics` → `analysis/graph_stats`) | Fast/targeted (definitions, LON stats and graph statistics, bodies, imports) |
| C (lon_stats split, Rule 2) | Fast/targeted **and** `tests/test_layering.py` |
| D1 (viz core + graph-population split) | **Full expensive gate**: this is where population, layout, styling and traces move |
| D2 (facade + consumer imports) | Fast/targeted, plus the full `heavy` test once as the mixed smoke. The heavy probe is all-or-nothing, so there is no cheaper single-case variant; skip it here only if D1 was clean and E is imminent |
| E (tighten + structural acceptance) | Full expensive gate **and** the full repository acceptance: the harness is tightened to post-only locations, the cache-only `visualization/` and `plotting/` directories are removed inside the runner, the focused suite and the full suite (twice) run, then the dashboard is restarted once for the runtime/manual acceptance |
| F (Pareto legacy removal) | Pareto half of the expensive gate, plus fast/targeted. After F the characterization requires the legacy monolith `plotParetoFrontMain.py` and the 12 camelCase Pareto aliases to be **absent** from both `viz/plots/__init__.py` and `viz/plots/pareto/__init__.py` (`__dict__` and `__all__`), while the 12 canonical `plot_*` functions and the 9 live `plot2d_*` performance aliases must remain |
| G (final) | Full expensive gate and the full acceptance, full suite twice |

Pytest markers (`slow`, `viz_expensive`) were considered and deliberately not added: registering a
marker requires `pytest.ini`, which Checkpoint 0 may not touch, and an unregistered marker would emit
warnings. `-k` and node-id selection give exactly the same targeting without changing collection.

## Stage 11 gate groups

`test_dashboard_package.py` is cheap — two probe subprocesses, about 12 s in total — so there is no
targeted subset to maintain. Its two fixtures are `app` (the dashboard: contract, config, layout,
HTTP, tables, import surface) and `mlflow` (the browser app).

**STAGE-11 GATE** — run after every checkpoint that touches the dashboard or the MLflow app:

```sh
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps evovis-runner-1 \
  python -m pytest tests/test_dashboard_package.py tests/test_layering.py
```

| Checkpoint | Gate |
|---|---|
| 11-0 (characterization) | The Stage-11 gate, the fast/targeted and expensive Stage-10 gates, `test_problems_package.py`, then the focused suite and the full suite twice |
| 11-A (columns move) | Stage-11 gate, fast/targeted |
| 11-B (transformers split, `dataio` dissolved) | Stage-11 gate, fast/targeted, `test_problems_package.py`, `test_layering.py` |
| 11-C (helpers module, wildcards removed) | Stage-11 gate, fast/targeted, **the heavy `update_plot` test**, `test_problems_package.py` — `update_plot`'s globals stop coming from a wildcard here |
| 11-D (B9 renames) | Stage-11 gate, fast/targeted |
| 11-E (instance + data extraction) | Stage-11 gate |
| 11-F1..F6 (one callback group each) | Stage-11 gate; F3 also fast/targeted, F5 also the heavy test |
| 11-I (`app.py`, Dashboard.py removed, compose retargeted) | Everything, plus the runtime acceptance; `ALLOW_PRE_LOCATIONS` (dashboard) goes to False, `ALLOW_PRE_MLFLOW_LOCATIONS` stays True until 11-J |
| 11-J (MLflow app renamed, `MLRUNS_DIR`) | Stage-11 gate plus the runtime acceptance; `ALLOW_PRE_MLFLOW_LOCATIONS` goes to False, so no PRE location is accepted anywhere |
| 11-final | Full expensive gate, focused suite, full suite twice |

**Re-capturing the frozen values** (only ever from PRE_STAGE_11, and only if the runner's Python
changes): extract `git archive fa8d250c65fb9b778a18262f165967476f824f98` into a scratch directory and
run the probe with that as its source root, twice, requiring identical reports.

**Negative controls.** The characterization was checked against mutated copies of PRE_STAGE_11 at
11-0: a swapped registration order, a changed callback body, an in-body relative import left at the
wrong level after a move, the unused `matplotlib` import dropped, a second `DashboardData.load()`,
a callback renamed outside the B9 map, and the orchestrator split into separate STN and LON
callbacks. Each was detected. A positive control — a simulated post-split tree with the helpers
rename, an `instance.py`, one callback group moved into `callbacks/`, one B9 rename and the
DashboardHelpers wildcard replaced by explicit imports — passes the whole characterization,
including the reassembled Stage-10 `BODIES` hashes.

## Isolation

The entry-point scripts write to the production warehouse, `data/mlruns` and `data/temp`. A test run
must never touch any of it, so isolation is enforced in three independent layers:

1. **Temp root.** Each run executes the real entry point as a subprocess whose working directory is a
   fresh temporary root outside the repository, holding its own `data/outputs`, `data/temp`,
   `data/warehouse`, `data/mlruns` and a symlink to the instance directory. `NOISYVIS_ROOT` points at
   the same root, so every `noisyvis.results.paths` location (warehouse, temp, SO/MO MLflow) resolves
   inside it; Hydra's outputs and the LON configs' `tracking_uri: "data/mlruns"` follow the cwd, which
   is the root too.
   **MLflow is not patched** (from Stage 6; Stages 1–5 patched `mlflow.set_tracking_uri`). Each entry
   point sets its own tracking URI, and `MLFLOW_TRACKING_URI` is an unreachable sentinel, so a missed
   call fails loudly instead of logging to the runner's configured server (R24).
2. **Write fence.** The subprocess installs an audit hook (`harness/fence.py`) that raises before any
   write, rename or delete whose target resolves under `/workspace`. Contamination fails loudly
   instead of happening silently.
3. **Production guard.** A session fixture records the warehouse pickles (`data/warehouse`, which it
   requires to exist), `data/mlruns` and `data/temp` before the session and asserts they are unchanged afterwards, printing both snapshots
   in the terminal summary.

## Baselines

`tests/baselines/*.json` hold the recorded behaviour. They are compared exactly — no tolerances.

```sh
# record (refuses to overwrite; use NOISYVIS_RECORD_BASELINES=overwrite to replace)
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps \
  -e NOISYVIS_RECORD_BASELINES=1 evovis-runner-1 python -m pytest tests/test_reproducibility.py
```

In normal mode a missing baseline is a failure, never a skip, so the gate cannot be bypassed.

`baselines/config_workflows.json` is the config-resolution golden for `test_config_workflows.py`. It
was recorded once, from the pre-Stage-7 `resolve_config_dependencies` copies (commit 777f46d), and is
**frozen**: the test refuses to re-record it in any record mode. A mismatch means config-resolution
behaviour changed.

`baselines/mo_algorithms.json` is the MO algorithm characterisation for `test_mo_algorithms.py`,
captured before the MO package split (commit 06328b8, recorded in the file) by running the test's own
probe twice in fresh processes with identical output. It is **frozen** and never re-derived. Since
the MoUMDA convergence stop, a baseline run whose probability vector collapses is compared as a prefix,
derived from the baseline's own per-generation record, through the first collapsed generation.

LON and CoLON baselines are **deliberately sequential** (`run.parallel=false`). Parallel LON output
depends on worker completion order and is nondeterministic by design (risk R27); do not try to make
it reproducible.

## Expected result on the unmodified tree

- All five baselines and the config-resolution golden pass.
- `test_config_resolution.py`: exactly one xfail — B1, `Multiobjective/MO_knapsack_test/mo_1p1ea.yaml`,
  which targets the nonexistent `MOAlgorithms.MoMuPlusLamdaEA`. A deferred behavioural fix.
- `test_layering.py`: both rules pass. Rule 1 from Stage 5, rule 2 from Stage 10 Checkpoint C.
- `test_viz_package.py`: 8 passed, no xfails.
- `test_dashboard_package.py`: 14 passed, no xfails.
- `test_mo_algorithms.py`: all pass, no xfails. Its two strict xfails from MO refactor Stage 0 pinned
  the `MoUMDA_noDuplicates` defect (duplicates never prevented, in generated or initial
  populations). They were removed in Stage 5, which fixed it, and now pass as ordinary tests.
- Everything else passes.

## Extending the compatibility fixtures

`historical_fixtures.py` is a declarative registry, one entry per artefact. When a new experiment
family, payload structure or schema version appears, preserve a small representative artefact and
append one `Fixture(...)` entry; the test body does not change. Candidates worth covering as they
arise: coping methods such as resampling/median, multi-objective runs, LON, CoLON, continuous
problems, and any schema version where fields are added or removed.
