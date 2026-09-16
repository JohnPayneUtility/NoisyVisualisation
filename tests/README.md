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
  python -m pytest tests/test_layering.py tests/test_historical_pickles.py tests/test_config_resolution.py
```

Resolves every `_target_`, `violation_fn`, `fitness_fn` and `attr_function` in `configs/` through each
runner's own namespace, loads the persisted artefacts, and scans imports for the two layering rules.
No experiment executes, so this is the quickest way to catch a broken import or a moved module.

**Expect:** `220 passed, 2 xfailed` in roughly 50 seconds (Stages 1–4: `219 passed, 3 xfailed`).

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

**Expect:** `226 passed, 2 xfailed` both times (Stages 1–4: `225 passed, 3 xfailed`), about 110 seconds each. Two consecutive identical runs
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
| 2 | gates only | `220 passed, 2 xfailed` |
| 3 | `-k so_seq`, no baselines yet | 1 failed: `no baseline recorded` |
| 4 | record mode | `6 passed`, five baselines written |
| 5 | full suite ×2 | `226 passed, 2 xfailed` each |

## What each file gates

| File | Gate |
|---|---|
| `test_reproducibility.py` | The five baselines of §5.5: SO-seq, SO-par, MO, LON, CoLON |
| `test_config_resolution.py` + `known_broken_configs.py` | Every `_target_`, `violation_fn`, `fitness_fn` and `attr_function` in `configs/` still resolves (§5.6) |
| `test_historical_pickles.py` + `historical_fixtures.py` | Existing persisted data still loads with the expected schema (§7.5); gates Stages 8 and 12 |
| `test_layering.py` | The two architectural rules of §5.1: rule 1 enforced from Stage 5, rule 2 xfailed until Stage 10 |
| `conftest.py` | Session-scoped production-data guard (§5.5a) |
| `harness/` | Isolated subprocess runner, write fence, extractors, canonical comparison |

## Isolation

The entry-point scripts write to the production warehouse, `data/mlruns` and `data/temp`. A test run
must never touch any of it, so isolation is enforced in three independent layers:

1. **Temp root.** Each run executes the real entry point as a subprocess whose working directory is a
   fresh temporary root outside the repository, holding its own `data/outputs`, `data/temp`,
   `data/dashboard_dw`, `mlruns` and a symlink to the instance directory.
2. **Write fence.** The subprocess installs an audit hook (`harness/fence.py`) that raises before any
   write, rename or delete whose target resolves under `/workspace`. Contamination fails loudly
   instead of happening silently.
3. **Production guard.** A session fixture records the warehouse pickles, `data/mlruns` and
   `data/temp` before the session and asserts they are unchanged afterwards, printing both snapshots
   in the terminal summary.

## Baselines

`tests/baselines/*.json` hold the recorded behaviour. They are compared exactly — no tolerances.

```sh
# record (refuses to overwrite; use NOISYVIS_RECORD_BASELINES=overwrite to replace)
docker exec -w /workspace -e PYTHONPATH=/workspace/tests/.deps \
  -e NOISYVIS_RECORD_BASELINES=1 evovis-runner-1 python -m pytest tests/test_reproducibility.py
```

In normal mode a missing baseline is a failure, never a skip, so the gate cannot be bypassed.

LON and CoLON baselines are **deliberately sequential** (`run.parallel=false`). Parallel LON output
depends on worker completion order and is nondeterministic by design (risk R27); do not try to make
it reproducible.

## Expected result on the unmodified tree

- All five baselines pass.
- `test_config_resolution.py`: exactly one xfail — B1, `Multiobjective/MO_knapsack_test/mo_1p1ea.yaml`,
  which targets the nonexistent `MOAlgorithms.MoMuPlusLamdaEA`. A deferred behavioural fix.
- `test_layering.py`: exactly one xfail — rule 2, until Stage 10. Rule 1 passes from Stage 5.
- Everything else passes.

## Extending the compatibility fixtures

`historical_fixtures.py` is a declarative registry, one entry per artefact. When a new experiment
family, payload structure or schema version appears, preserve a small representative artefact and
append one `Fixture(...)` entry; the test body does not change. Candidates worth covering as they
arise: coping methods such as resampling/median, multi-objective runs, LON, CoLON, continuous
problems, and any schema version where fields are added or removed.
