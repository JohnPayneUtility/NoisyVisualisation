"""Filesystem locations for persisted data and results (plan §5.9).

Anchored to the repository, not the working directory. `parents[3]` counts up from
src/noisyvis/results/paths.py to the project root; `NOISYVIS_ROOT` overrides it, which is how the
test harness isolates every run (§5.5a). Constants only: importing this module creates, reads and
writes nothing.

Deliberately still relative to the working directory after Stage 6: the 16
`${hydra:runtime.cwd}/fast_storage` configs (R12), the LON configs' `tracking_uri: "data/mlruns"`,
Hydra's output directory, and the knapsack instance literals (anchored to INSTANCES_DIR in Stage 9).
"""

import os
from pathlib import Path

PROJECT_ROOT  = Path(os.environ.get("NOISYVIS_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR      = PROJECT_ROOT / "data"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
TEMP_DIR      = DATA_DIR / "temp"
MLRUNS_DIR    = DATA_DIR / "mlruns"
PLOTS_DIR     = PROJECT_ROOT / "plots"
INSTANCES_DIR = PROJECT_ROOT / "instances"
FAST_STORAGE  = Path(os.environ.get("NOISYVIS_FAST_STORAGE", PROJECT_ROOT / "fast_storage"))
