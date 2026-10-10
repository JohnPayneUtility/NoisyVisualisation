"""Per-run wall time, CPU time and RAM, measured the same way by the SO and MO runners.

One measurement window per algorithm run (one seed): it opens immediately before `instantiate(...)` and
closes immediately after `algo_instance.run()`. It therefore includes construction, where every algorithm
evaluates its initial population, and whatever recording the algorithm does while it runs (the SO logger,
LMDB writes, the MO evaluation log and its per-generation hypervolumes). It excludes post-run finalisation,
payloads, MLflow, CSV/warehouse writing and orchestration. Measuring touches no RNG and no algorithm state.

    run_wall_time_s   time.perf_counter() over the window
    run_cpu_time_s    time.process_time() over the window: user + system CPU of this process, all of its
                      threads (BLAS/OpenMP threads can make it exceed the wall time), no child processes
    ram_baseline_mb   resident set size (VmRSS) when the window opens, MiB
    peak_ram_mb       peak resident set size (VmHWM) within the window, MiB, absolute: the kernel's peak
                      is reset to the current RSS when the window opens, so an earlier run in the same
                      process (sequential runs, or a reused pool worker) is not carried over. It includes
                      ram_baseline_mb: the interpreter and imports, pages a forked worker inherited from
                      its parent, and memory an earlier run freed that the allocator kept for reuse.

Values are stored at full precision. The peak reset is Linux-only (/proc/self/clear_refs, "5"). Where
/proc cannot be read or the peak cannot be reset the RAM values are NaN rather than a number that would
silently carry over between runs; measurement never fails a run and never replaces an algorithm error.

Legacy rows: single-objective rows written before this module carry `peak_ram_mb` but no
`ram_baseline_mb`. Their peak_ram_mb is getrusage(RUSAGE_SELF).ru_maxrss, a process-lifetime high-water
mark (cumulative over earlier seeds in the same process, and including the parent's inherited memory),
not a per-run peak. It is not comparable with the per-run values; it is kept as recorded, not backfilled.
"""

import math
import time
from contextlib import contextmanager

# The four row fields, in row order.
RESOURCE_COLUMNS = ("run_wall_time_s", "run_cpu_time_s", "peak_ram_mb", "ram_baseline_mb")

_PROC_STATUS = "/proc/self/status"
_PROC_CLEAR_REFS = "/proc/self/clear_refs"


def _status_mib(field: str) -> float:
    """A kB field of /proc/self/status (VmRSS, VmHWM) in MiB, or NaN if it cannot be read."""
    try:
        with open(_PROC_STATUS) as status:
            for line in status:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return math.nan


def _reset_peak_rss() -> bool:
    """Reset this process's peak RSS (VmHWM) to its current RSS; False if that is not possible."""
    try:
        with open(_PROC_CLEAR_REFS, "w") as clear_refs:
            clear_refs.write("5")
        return True
    except Exception:
        return False


@contextmanager
def measure_run():
    """
    Measure one run: `with measure_run() as usage:` around instantiate + run(). `usage` is a dict of
    RESOURCE_COLUMNS, NaN until the window closes, filled in even if the run raises (the exception
    propagates unchanged). The peak is reset and the baseline read before the clocks start, and the clocks
    stop before the peak is read, so the measurement's own /proc I/O is not timed.
    """
    usage = dict.fromkeys(RESOURCE_COLUMNS, math.nan)
    reset = _reset_peak_rss()
    usage["ram_baseline_mb"] = _status_mib("VmRSS")
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        yield usage
    finally:
        usage["run_wall_time_s"] = time.perf_counter() - wall_start
        usage["run_cpu_time_s"] = time.process_time() - cpu_start
        # The kernel's RSS counters are batched per CPU, so VmHWM and VmRSS can disagree by about a MiB;
        # the window's peak is by definition at least the RSS it started with. (max() keeps a NaN baseline
        # from hiding a valid peak and a NaN peak NaN.)
        peak = _status_mib("VmHWM") if reset else math.nan
        usage["peak_ram_mb"] = peak if math.isnan(peak) or math.isnan(usage["ram_baseline_mb"]) \
            else max(peak, usage["ram_baseline_mb"])
