"""
Read API over one persisted multi-objective run record (`mo_record`, MO recording plan v6, Work Group 5).

A `mo_record` is the plain-data freeze of a run's MO evaluation log (tracking.mo_logger.finalise): NumPy
arrays and Python primitives only. `MORunView` validates one record once and then answers every query
from the persisted structures: it never replays the evaluation events, recomputes a front or a
hypervolume, evaluates anything or draws RNG. It depends on NumPy only (no DEAP, Dash, MLflow, pandas,
algorithm classes or file I/O); the caller passes an already-loaded record.

Semantics, as recorded:

    event e                 x (orig_geno: the solution the optimiser generated), x~ (eval_geno: the
                            genotype actually evaluated, == x without prior noise), f(x) (the
                            deterministic objectives of x, an independent analytical measurement) and
                            y (the observed objectives the optimiser received)
    gen_last_eval[g]        exclusive end offset: the events of generation g are
                            range(gen_last_eval[g-1] if g else 0, gen_last_eval[g])
    current fronts          ND(current population): noisy under y, members are canonical observations;
                            clean under f(x), members are genotypes (computed independently)
    noisy archive           ND(all canonical observations so far, under y)
    clean archive           ND(all genotypes generated as x so far, under f(x)); a genotype seen only as
                            x~ is not a candidate until it is generated
    archive intervals       evaluation precision, half-open: a member is present after event e iff
                            enter_eval <= e < exit_eval (exit_eval == NOT_EXITED while still a member)

Member arrays in snapshots are sorted by member id (observation id or genotype id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

MO_RECORD_SCHEMA = "noisyvis.mo_record"
MO_RECORD_VERSION = 1
NOT_EXITED = -1

CURRENT_FRONT_METRICS = ("current_noisy_front__noisy", "current_noisy_front__clean", "current_clean_front__clean")
ARCHIVE_METRICS = ("noisy_archive__noisy", "noisy_archive__clean", "clean_archive__clean")
HV_METRICS = CURRENT_FRONT_METRICS + ARCHIVE_METRICS


class MORecordError(ValueError):
    """A mo_record is malformed or incompatible with this reader."""


# ==============================
# Value objects
# ==============================

@dataclass(frozen=True)
class MOEvent:
    """One genuine evaluation. x_tilde is x without prior noise; true_obj is f(x)."""
    eval_id: int
    generation: int
    orig_geno: int
    eval_geno: int
    obs_id: int
    x: np.ndarray
    x_tilde: np.ndarray
    true_obj: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class MOEvents:
    """All evaluations as parallel arrays (row i is eval_id i)."""
    eval_id: np.ndarray
    generation: np.ndarray
    orig_geno: np.ndarray
    eval_geno: np.ndarray
    obs_id: np.ndarray
    true_obj: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class ArchiveIntervals:
    """Evaluation-precision membership rows of a passive archive, in entry order."""
    member: np.ndarray
    enter_eval: np.ndarray
    exit_eval: np.ndarray
    exit_by: np.ndarray

    @property
    def still_present(self) -> np.ndarray:
        return self.exit_eval == NOT_EXITED


@dataclass(frozen=True)
class FrontSnapshot:
    """
    One front or archive state. For the clean kinds (current_clean_front, clean_archive,
    lost_clean_solutions) the observation fields are None: a clean state has no x~ or y.

        geno_ids, x, true_obj   the original genotypes (for noisy kinds, the x of each observation)
        first_generated_eval    first event whose x is each genotype
        obs_ids, eval_ids       noisy kinds: canonical observations and the first event of each
        x_tilde_ids, x_tilde    noisy kinds: the evaluated genotype of each observation
        observed                noisy kinds: y of each observation
        hv                      the kind's hypervolume metrics at this state
    """
    kind: str
    generation: Optional[int]
    n_evals: int
    geno_ids: np.ndarray
    x: np.ndarray
    true_obj: np.ndarray
    first_generated_eval: np.ndarray
    obs_ids: Optional[np.ndarray]
    eval_ids: Optional[np.ndarray]
    x_tilde_ids: Optional[np.ndarray]
    x_tilde: Optional[np.ndarray]
    observed: Optional[np.ndarray]
    hv: Mapping[str, float]

    @property
    def members(self) -> np.ndarray:
        """The identities: observation ids for noisy kinds, genotype ids for clean kinds."""
        return self.obs_ids if self.obs_ids is not None else self.geno_ids

    def __len__(self) -> int:
        return len(self.geno_ids)


class FrontHistory:
    """
    Ordered change-point states of a front or archive. resolution "generation": state i holds from
    generation starts[i]; resolution "evaluation": state i is the membership after evaluation starts[i]
    (every change at one evaluation is one state). States are never de-duplicated globally: a
    membership that disappears and returns is a later state.
    """

    def __init__(self, kind: str, resolution: str, starts: np.ndarray, state: Callable[[int], FrontSnapshot],
                 gen_last_eval: np.ndarray):
        self.kind = kind
        self.resolution = resolution
        self.starts = starts
        self._state = state
        self._gen_last_eval = gen_last_eval

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> FrontSnapshot:
        if not -len(self) <= i < len(self):
            raise IndexError(f"state {i} of {len(self)}")
        return self._state(i % len(self))

    def __iter__(self):
        return (self._state(i) for i in range(len(self)))

    def index_at(self, position: int) -> int:
        """Index of the state in force at `position` (a generation, or an eval_id)."""
        bound = len(self._gen_last_eval) if self.resolution == "generation" else int(self._gen_last_eval[-1])
        if not 0 <= position < bound:
            raise IndexError(f"{self.resolution} {position} is not recorded (0..{bound - 1})")
        i = int(np.searchsorted(self.starts, position, side="right")) - 1
        if i < 0:
            raise IndexError(f"no {self.kind} state at {self.resolution} {position}")
        return i

    def at(self, position: int) -> FrontSnapshot:
        """The state in force at a generation (generation resolution) or after an eval_id (evaluation)."""
        return self._state(self.index_at(position))

    def at_generation(self, gen: int) -> FrontSnapshot:
        if self.resolution == "generation":
            return self.at(gen)
        if not 0 <= gen < len(self._gen_last_eval):
            raise IndexError(f"generation {gen} is not recorded")
        return self.at(int(self._gen_last_eval[gen]) - 1)


@dataclass(frozen=True)
class HVSeries:
    """
    A hypervolume metric: its persisted change points (positions are generations for the current-front
    metrics, eval_ids for the archive metrics) and the dense per-generation trajectory derived from them.
    """
    metric: str
    resolution: str
    positions: np.ndarray
    values: np.ndarray
    by_generation: np.ndarray

    def at_generation(self, gen: int) -> float:
        if not 0 <= gen < len(self.by_generation):
            raise IndexError(f"generation {gen} is not recorded")
        return float(self.by_generation[gen])

    def after(self, n_events: int) -> Optional[float]:
        """Evaluation-resolution metrics: the value after the first n_events events (None before any)."""
        if self.resolution != "evaluation":
            raise ValueError(f"{self.metric} is recorded per generation; use at_generation")
        i = int(np.searchsorted(self.positions, n_events - 1, side="right")) - 1
        return None if i < 0 else float(self.values[i])

    def at_eval(self, eval_id: int) -> Optional[float]:
        return self.after(eval_id + 1)


# ==============================
# The view
# ==============================

def _readonly(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.flags.writeable = False
    return view


class MORunView:
    """Read-only, NumPy-only view over one validated mo_record."""

    def __init__(self, record: Mapping):
        _validate(record)
        meta = record["meta"]
        self.meta = dict(meta)
        self.n_evals = meta["n_evals"]
        self.n_generations = meta["n_generations"]
        self.n_obj = meta["n_obj"]
        self.sol_length = meta["sol_length"]
        self.weights = tuple(meta["opt_weights"])
        self.ref_point = None if meta["ref_point"] is None else tuple(meta["ref_point"])
        self.prior_noise_observed = meta["prior_noise_observed"]

        genotypes, events = record["genotypes"], record["events"]
        self._values = _readonly(genotypes["values"])
        self._true_obj = _readonly(genotypes["true_obj"])
        self._first_orig_eval = _readonly(genotypes["first_orig_eval"])
        self._orig = _readonly(events["orig_geno"])
        self._eval = _readonly(events["orig_geno"] if events["eval_geno"] is None else events["eval_geno"])
        self._obs_obj = _readonly(events["obs_obj"])
        self._obs_id = _readonly(events["obs_id"])
        self.gen_last_eval = _readonly(record["gen_last_eval"])
        _, first = np.unique(self._obs_id, return_index=True)
        self._obs_first_eval = _readonly(first)
        self._generation = _readonly(np.searchsorted(self.gen_last_eval, np.arange(self.n_evals), side="right"))

        self._current = {"noisy": record["current_noisy_front"], "clean": record["current_clean_front"]}
        self._archive = {"noisy": record["noisy_archive"], "clean": record["clean_archive"]}

    # ------------------------------------------------------------------ generations and events

    def evals_through(self, gen: int) -> int:
        """Events completed by the end of generation gen (exclusive end offset)."""
        return int(self.gen_last_eval[self._gen(gen)])

    def evals_in(self, gen: int) -> range:
        gen = self._gen(gen)
        return range(int(self.gen_last_eval[gen - 1]) if gen else 0, int(self.gen_last_eval[gen]))

    def generation_of(self, eval_id: int) -> int:
        return int(self._generation[self._eval_id(eval_id)])

    def event(self, eval_id: int) -> MOEvent:
        e = self._eval_id(eval_id)
        orig, evaluated = int(self._orig[e]), int(self._eval[e])
        return MOEvent(e, int(self._generation[e]), orig, evaluated, int(self._obs_id[e]), self._values[orig],
                       self._values[evaluated], self._true_obj[orig], self._obs_obj[e])

    def events(self) -> MOEvents:
        return MOEvents(np.arange(self.n_evals), self._generation, self._orig, self._eval, self._obs_id,
                        self._true_obj[self._orig], self._obs_obj)

    def genotype(self, geno_id: int) -> np.ndarray:
        return self._values[geno_id]

    def true_objectives(self, geno_id: int) -> np.ndarray:
        """f(x) of a genotype the optimiser generated; genotypes only ever seen as x~ have none."""
        if self._first_orig_eval[geno_id] < 0:
            raise KeyError(f"genotype {geno_id} was never generated as x, so f of it is not recorded")
        return self._true_obj[geno_id]

    # ------------------------------------------------------------------ snapshots

    def _noisy_snapshot(self, kind, obs_ids, generation, n_evals, hv) -> FrontSnapshot:
        obs_ids = np.asarray(obs_ids, dtype=np.int64)
        first = self._obs_first_eval[obs_ids]
        geno, x_tilde_ids = self._orig[first], self._eval[first]
        return FrontSnapshot(kind, generation, n_evals, geno, self._values[geno], self._true_obj[geno],
                             self._first_orig_eval[geno], obs_ids, first, x_tilde_ids, self._values[x_tilde_ids],
                             self._obs_obj[first], hv)

    def _clean_snapshot(self, kind, geno_ids, generation, n_evals, hv) -> FrontSnapshot:
        geno = np.asarray(geno_ids, dtype=np.int64)
        return FrontSnapshot(kind, generation, n_evals, geno, self._values[geno], self._true_obj[geno],
                             self._first_orig_eval[geno], None, None, None, None, None, hv)

    # ------------------------------------------------------------------ current fronts

    def _current_row(self, which: str, gen: int) -> int:
        return int(np.searchsorted(self._current[which]["gen"], self._gen(gen), side="right")) - 1

    def _current_state(self, which: str, row: int, generation: int) -> FrontSnapshot:
        block = self._current[which]
        members = block["members"][block["member_offsets"][row]:block["member_offsets"][row + 1]]
        n_evals = int(self.gen_last_eval[generation])
        if which == "noisy":
            hv = {"current_noisy_front__noisy": float(block["hv_noisy"][row]),
                  "current_noisy_front__clean": float(block["hv_clean"][row])}
            return self._noisy_snapshot("current_noisy_front", members, generation, n_evals, hv)
        hv = {"current_clean_front__clean": float(block["hv_clean"][row])}
        return self._clean_snapshot("current_clean_front", members, generation, n_evals, hv)

    def current_noisy_front(self, gen: int) -> FrontSnapshot:
        return self._current_state("noisy", self._current_row("noisy", gen), self._gen(gen))

    def current_clean_front(self, gen: int) -> FrontSnapshot:
        return self._current_state("clean", self._current_row("clean", gen), self._gen(gen))

    def _current_history(self, which: str) -> FrontHistory:
        starts = _readonly(self._current[which]["gen"])
        return FrontHistory(f"current_{which}_front", "generation", starts,
                            lambda i: self._current_state(which, i, int(starts[i])), self.gen_last_eval)

    def current_noisy_front_history(self) -> FrontHistory:
        return self._current_history("noisy")

    def current_clean_front_history(self) -> FrontHistory:
        return self._current_history("clean")

    # ------------------------------------------------------------------ passive archives

    def _archive_members_after(self, which: str, n_events: int) -> np.ndarray:
        block = self._archive[which]
        exit_eval = block["exit_eval"]
        present = (block["enter_eval"] < n_events) & ((exit_eval == NOT_EXITED) | (exit_eval >= n_events))
        return np.sort(block["member"][present])

    def _archive_hv(self, which: str, n_events: int) -> dict:
        metrics = ("noisy_archive__noisy", "noisy_archive__clean") if which == "noisy" else ("clean_archive__clean",)
        return {m: self.hv(m).after(n_events) for m in metrics}

    def _archive_state(self, which: str, n_events: int, generation: Optional[int]) -> FrontSnapshot:
        members = self._archive_members_after(which, n_events)
        kind = f"{which}_archive"
        hv = self._archive_hv(which, n_events)
        if which == "noisy":
            return self._noisy_snapshot(kind, members, generation, n_events, hv)
        return self._clean_snapshot(kind, members, generation, n_events, hv)

    def noisy_archive_after(self, n_events: int) -> FrontSnapshot:
        return self._archive_state("noisy", self._n_events(n_events), None)

    def clean_archive_after(self, n_events: int) -> FrontSnapshot:
        return self._archive_state("clean", self._n_events(n_events), None)

    def noisy_archive_at_eval(self, eval_id: int) -> FrontSnapshot:
        return self.noisy_archive_after(self._eval_id(eval_id) + 1)

    def clean_archive_at_eval(self, eval_id: int) -> FrontSnapshot:
        return self.clean_archive_after(self._eval_id(eval_id) + 1)

    def noisy_archive(self, gen: int) -> FrontSnapshot:
        return self._archive_state("noisy", self.evals_through(gen), self._gen(gen))

    def clean_archive(self, gen: int) -> FrontSnapshot:
        return self._archive_state("clean", self.evals_through(gen), self._gen(gen))

    def _archive_history(self, which: str, resolution: str) -> FrontHistory:
        if resolution == "evaluation":
            # every membership change happens when a candidate enters (evicting others at the same
            # evaluation), so each entry evaluation is exactly one post-evaluation state
            starts = _readonly(np.unique(self._archive[which]["enter_eval"]))
            return FrontHistory(f"{which}_archive", "evaluation", starts,
                                lambda i: self._archive_state(which, int(starts[i]) + 1, None), self.gen_last_eval)
        if resolution == "generation":
            candidates = np.unique(self._generation[self._archive[which]["enter_eval"]])
            starts, previous = [], None
            for gen in candidates:
                members = self._archive_members_after(which, int(self.gen_last_eval[gen]))
                if previous is None or not np.array_equal(members, previous):
                    starts.append(int(gen))
                    previous = members
            starts = _readonly(np.asarray(starts, dtype=np.int64))
            return FrontHistory(f"{which}_archive", "generation", starts,
                                lambda i: self._archive_state(which, int(self.gen_last_eval[starts[i]]), int(starts[i])),
                                self.gen_last_eval)
        raise ValueError(f"resolution must be 'evaluation' or 'generation', not {resolution!r}")

    def noisy_archive_history(self, resolution: str = "evaluation") -> FrontHistory:
        return self._archive_history("noisy", resolution)

    def clean_archive_history(self, resolution: str = "evaluation") -> FrontHistory:
        return self._archive_history("clean", resolution)

    def _intervals(self, which: str) -> ArchiveIntervals:
        block = self._archive[which]
        return ArchiveIntervals(*(_readonly(block[k]) for k in ("member", "enter_eval", "exit_eval", "exit_by")))

    def noisy_archive_intervals(self) -> ArchiveIntervals:
        return self._intervals("noisy")

    def clean_archive_intervals(self) -> ArchiveIntervals:
        return self._intervals("clean")

    # ------------------------------------------------------------------ derived

    def lost_clean_solutions(self, gen: int) -> FrontSnapshot:
        """
        Genotypes non-dominated under f(x) among everything the optimiser generated up to generation gen
        (the clean archive) that its population no longer holds at gen: clean_archive(gen) minus
        current_clean_front(gen). (A clean-archive member still in the population is necessarily on the
        current clean front, so this is exactly the archive members absent from the population.)
        """
        archive = self.clean_archive(gen).geno_ids
        front = self.current_clean_front(gen).geno_ids
        lost = np.setdiff1d(archive, front)
        return self._clean_snapshot("lost_clean_solutions", lost, self._gen(gen), self.evals_through(gen), {})

    # ------------------------------------------------------------------ hypervolume

    def hv(self, metric: str) -> HVSeries:
        if metric in CURRENT_FRONT_METRICS:
            which = "noisy" if metric.startswith("current_noisy") else "clean"
            block = self._current[which]
            values = block["hv_noisy"] if metric == "current_noisy_front__noisy" else block["hv_clean"]
            rows = np.searchsorted(block["gen"], np.arange(self.n_generations), side="right") - 1
            return HVSeries(metric, "generation", _readonly(block["gen"]), _readonly(values), _readonly(values[rows]))
        if metric in ARCHIVE_METRICS:
            block = self._archive["noisy" if metric.startswith("noisy") else "clean"]
            values = block["hv_noisy"] if metric == "noisy_archive__noisy" else block["hv_clean"]
            rows = np.searchsorted(block["hv_eval"], self.gen_last_eval - 1, side="right") - 1
            dense = np.where(rows >= 0, values[np.maximum(rows, 0)] if len(values) else np.nan, np.nan)
            return HVSeries(metric, "evaluation", _readonly(block["hv_eval"]), _readonly(values), _readonly(dense))
        raise KeyError(f"unknown metric {metric!r}; expected one of {HV_METRICS}")

    # ------------------------------------------------------------------ summaries

    def summary_scalars(self) -> dict:
        """The final-generation scalars stored alongside mo_record in a result row."""
        last = self.n_generations - 1
        scalars = {f"final_hv_{m}": self.hv(m).at_generation(last) for m in HV_METRICS}
        scalars["n_generated_genotypes"] = int(np.count_nonzero(self._first_orig_eval >= 0))
        scalars["final_noisy_archive_size"] = len(self.noisy_archive(last))
        scalars["final_clean_archive_size"] = len(self.clean_archive(last))
        return scalars

    # ------------------------------------------------------------------ bounds

    def _gen(self, gen: int) -> int:
        if not 0 <= gen < self.n_generations:
            raise IndexError(f"generation {gen} is not recorded (0..{self.n_generations - 1})")
        return int(gen)

    def _eval_id(self, eval_id: int) -> int:
        if not 0 <= eval_id < self.n_evals:
            raise IndexError(f"eval_id {eval_id} is not recorded (0..{self.n_evals - 1})")
        return int(eval_id)

    def _n_events(self, n_events: int) -> int:
        if not 0 <= n_events <= self.n_evals:
            raise IndexError(f"{n_events} events requested; the record holds {self.n_evals}")
        return int(n_events)


# ==============================
# Validation (once, at construction)
# ==============================

def _fail(message: str):
    raise MORecordError(message)


def _array(block: Mapping, key: str, kind: str, ndim: int, where: str) -> np.ndarray:
    value = block.get(key) if isinstance(block, Mapping) else None
    if not isinstance(value, np.ndarray):
        _fail(f"{where}.{key} must be a numpy array, got {type(value).__name__}")
    if value.dtype.kind not in kind:
        _fail(f"{where}.{key} has dtype {value.dtype}, expected kind {kind!r}")
    if value.ndim != ndim:
        _fail(f"{where}.{key} has {value.ndim} dimensions, expected {ndim}")
    return value


def _in_range(values: np.ndarray, low: int, high: int, where: str):
    if values.size and (values.min() < low or values.max() >= high):
        _fail(f"{where} has values outside [{low}, {high})")


def _validate(record):
    if not isinstance(record, Mapping):
        _fail(f"a mo_record must be a mapping, got {type(record).__name__}")
    if record.get("schema") != MO_RECORD_SCHEMA:
        _fail(f"schema is {record.get('schema')!r}, expected {MO_RECORD_SCHEMA!r}")
    if record.get("version") != MO_RECORD_VERSION:
        _fail(f"version {record.get('version')!r} is not supported (this reader reads {MO_RECORD_VERSION})")
    for key in ("meta", "genotypes", "events", "gen_last_eval", "current_noisy_front", "current_clean_front",
                "noisy_archive", "clean_archive"):
        if key not in record:
            _fail(f"missing {key!r}")
    meta = record["meta"]
    for key, kind in (("n_evals", int), ("n_generations", int), ("n_genotypes", int), ("n_observations", int),
                      ("n_obj", int), ("sol_length", int), ("prior_noise_observed", bool), ("opt_weights", tuple)):
        if not isinstance(meta.get(key), kind):
            _fail(f"meta.{key} must be {kind.__name__}, got {type(meta.get(key)).__name__}")
    n, t, g, o, m, l = (meta[k] for k in ("n_evals", "n_generations", "n_genotypes", "n_observations", "n_obj",
                                          "sol_length"))
    if len(meta["opt_weights"]) != m:
        _fail("meta.opt_weights does not have n_obj entries")
    if meta.get("ref_point") is not None and len(meta["ref_point"]) != m:
        _fail("meta.ref_point does not have n_obj entries")

    # genotypes
    values = _array(record["genotypes"], "values", "uif", 2, "genotypes")
    true_obj = _array(record["genotypes"], "true_obj", "f", 2, "genotypes")
    first_orig = _array(record["genotypes"], "first_orig_eval", "i", 1, "genotypes")
    if values.shape != (g, l) or true_obj.shape != (g, m) or first_orig.shape != (g,):
        _fail("genotype table shapes do not match meta (n_genotypes, sol_length, n_obj)")
    if not np.array_equal(np.isfinite(true_obj).all(axis=1), first_orig >= 0):
        _fail("genotypes.true_obj must be finite exactly for the genotypes generated as x")

    # events
    events = record["events"]
    orig = _array(events, "orig_geno", "i", 1, "events")
    evaluated = orig if events.get("eval_geno", "missing") is None else _array(events, "eval_geno", "i", 1, "events")
    obs_obj = _array(events, "obs_obj", "f", 2, "events")
    obs_id = _array(events, "obs_id", "i", 1, "events")
    if orig.shape != (n,) or evaluated.shape != (n,) or obs_id.shape != (n,) or obs_obj.shape != (n, m):
        _fail("event array shapes do not match meta (n_evals, n_obj)")
    _in_range(orig, 0, g, "events.orig_geno")
    _in_range(evaluated, 0, g, "events.eval_geno")
    _in_range(obs_id, 0, o, "events.obs_id")
    if bool(meta["prior_noise_observed"]) != bool((evaluated != orig).any()):
        _fail("meta.prior_noise_observed disagrees with the events")
    first_as_x = np.full(g, -1, dtype=np.int64)
    genos, first_idx = np.unique(orig, return_index=True)
    first_as_x[genos] = first_idx
    if not np.array_equal(first_as_x, first_orig):
        _fail("genotypes.first_orig_eval is not the first event of each genotype as x")

    # canonical observation identity: one obs_id per distinct (x, x~, y), numbered in first-appearance order
    ids, first_seen = np.unique(obs_id, return_index=True)
    if not np.array_equal(ids, np.arange(o)) or np.any(np.diff(first_seen) <= 0):
        _fail("events.obs_id is not numbered 0..n_observations-1 in first-appearance order")
    if n:
        keys = np.column_stack([orig.astype(np.float64), evaluated.astype(np.float64), obs_obj])
        _, key_first, key_of = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        key_of = key_of.reshape(-1)
        if len(key_first) != o or not np.array_equal(obs_id, obs_id[key_first][key_of]):
            _fail("events.obs_id does not identify exactly the distinct (orig_geno, eval_geno, observed) events")

    # generations
    gen_last_eval = _array(record, "gen_last_eval", "i", 1, "record")
    if gen_last_eval.shape != (t,) or t < 1:
        _fail("gen_last_eval must hold n_generations >= 1 boundaries")
    if np.any(np.diff(gen_last_eval) < 0) or gen_last_eval[0] < 0 or gen_last_eval[-1] != n:
        _fail("gen_last_eval must be non-decreasing and end at n_evals")

    # current fronts
    for which, ids_bound, hv_keys in (("current_noisy_front", o, ("hv_noisy", "hv_clean")),
                                      ("current_clean_front", g, ("hv_clean",))):
        block = record[which]
        gens = _array(block, "gen", "i", 1, which)
        offsets = _array(block, "member_offsets", "i", 1, which)
        members = _array(block, "members", "i", 1, which)
        if len(gens) == 0 or gens[0] != 0 or np.any(np.diff(gens) <= 0) or gens[-1] >= t:
            _fail(f"{which}.gen must start at 0, strictly increase and stay below n_generations")
        if offsets.shape != (len(gens) + 1,) or offsets[0] != 0 or offsets[-1] != len(members) \
                or np.any(np.diff(offsets) < 0):
            _fail(f"{which}.member_offsets is not a valid CSR index")
        _in_range(members, 0, ids_bound, f"{which}.members")
        for key in hv_keys:
            if _array(block, key, "f", 1, which).shape != gens.shape:
                _fail(f"{which}.{key} does not have one value per row")

    # passive archives
    for which, ids_bound, hv_keys in (("noisy_archive", o, ("hv_noisy", "hv_clean")),
                                      ("clean_archive", g, ("hv_clean",))):
        block = record[which]
        member, enter, exit_eval, exit_by = (_array(block, k, "i", 1, which)
                                             for k in ("member", "enter_eval", "exit_eval", "exit_by"))
        if not (member.shape == enter.shape == exit_eval.shape == exit_by.shape):
            _fail(f"{which} interval columns differ in length")
        _in_range(member, 0, ids_bound, f"{which}.member")
        _in_range(enter, 0, n, f"{which}.enter_eval")
        open_rows = exit_eval == NOT_EXITED
        if np.any(~open_rows & ((exit_eval <= enter) | (exit_eval >= n))):
            _fail(f"{which}.exit_eval must be NOT_EXITED or after enter_eval and within the events")
        if not np.array_equal(open_rows, exit_by == NOT_EXITED):
            _fail(f"{which}.exit_by must be NOT_EXITED exactly for members still present")
        if len(np.unique(member)) != len(member) or np.any(np.diff(enter) <= 0):
            _fail(f"{which}: each member enters once, one candidate per evaluation, in order")
        hv_eval = _array(block, "hv_eval", "i", 1, which)
        if not np.array_equal(hv_eval, enter):
            _fail(f"{which}.hv_eval must be the evaluations at which the membership changed")
        for key in hv_keys:
            if _array(block, key, "f", 1, which).shape != hv_eval.shape:
                _fail(f"{which}.{key} does not have one value per change")
