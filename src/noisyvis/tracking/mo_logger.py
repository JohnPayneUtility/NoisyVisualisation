"""Multi-objective evaluation log (MO recording plan v6, Work Group 2).

One MOEvaluationLogger per algorithm instance, hence per run. It records every genuine algorithm
evaluation exactly once:

    eval_id           position in the log (0, 1, 2, ...)
    x                 the solution the optimiser generated         -> orig_geno (genotype id)
    x~                the genotype actually evaluated               -> eval_geno (genotype id)
                      (x~ == x for posterior noise; may differ under prior noise)
    f(x)              deterministic objectives of x, as the evaluator computed them
    y                 the observed objective vector returned to the algorithm
    obs_id            canonical observation id: exact (x, x~, y) twins share one

It is purely observational: it draws no RNG, evaluates nothing and never changes what the algorithm
receives.

Scoping. Events are created only by `evaluate()`, which OptimisationAlgorithm._evaluate_and_track
calls for each toolbox.evaluate. For the duration of that one evaluator call it makes this logger the
active logger (the singleton in noisyvis.tracking.logger), and the evaluator reports (x~, f(x), y)
through `log_mo_eval`. Outside that call the previous active logger is restored, so evaluations that
are not algorithm evaluations (the legacy recorder's clean re-evaluations) are never logged.

Provenance. After each evaluation the individual gets

    ind._eval_tag = (eval_id, orig_geno_id, wvalues)

Lifecycle: the evaluator returns y -> the event is committed -> the tag is attached, with wvalues
computed from y exactly as deap does -> the caller assigns fitness.values = y -> later, `resolve()`
validates the individual's current genotype and fitness against the tag. Between tagging and the
caller's assignment the individual briefly has a tag but not yet that fitness; that is intended.
The tag survives toolbox.clone (deepcopy), is replaced by a genuine re-evaluation, and plays no part
in selection, equality or dominance. It carries no run identity: resolve an individual only against
the logger of the run that owns it.

Current fronts (Work Group 3). OptimisationAlgorithm._observe_generation calls `observe_generation`
once per generation g = 0, 1, 2, ... with the current population. It records:

    gen_last_eval[g]        exclusive end offset: the number of genuine evaluations completed by the end
                            of generation g (not the last eval_id). Generation g's events are
                            range(gen_last_eval[g-1] if g else 0, gen_last_eval[g]).
    current noisy front     ND(current population) under the observed y; members are canonical obs_ids
    current clean front     ND(unique original genotypes of the current population) under f(x); members
                            are orig_geno ids. Computed independently, never the noisy front re-scored.

Both fronts are change-point histories: a row is appended only when the (sorted) membership differs
from the previous row, and the state in force at generation g is the last row starting at or before g.
Each row carries its hypervolumes:

    current_noisy_front__noisy   HV of the noisy-front members' observed y
    current_noisy_front__clean   HV of the same members re-scored with f(x), one point per member
    current_clean_front__clean   HV of the clean front under f(x)

Passive historical archives (Work Group 4), replayed from the event stream only (never from the
population or the current fronts) and never read by the optimiser:

    noisy archive   ND(all canonical observations seen so far, under y); members are obs_ids, offered
                    at their first event (exact (x, x~, y) twins are not re-offered)
    clean archive   ND(all genotypes generated as x so far, under f(x)); members are genotype ids,
                    offered at the first event whose x is that genotype (never when only seen as x~)

Membership intervals are at evaluation precision and half-open (see common.pareto.ArchiveReplay):
present after event e iff enter <= e < exit, exit = NOT_EXITED while still a member. The archive at
generation g is the archive after the first gen_last_eval[g] events. HVs are recorded at each
membership change: noisy_archive__noisy (y of the members), noisy_archive__clean (f(x) of the same
members) and clean_archive__clean. The replay is cached by event count; generation views always read
the current gen_last_eval.
"""

from bisect import bisect_right
from operator import mul
from typing import NamedTuple, Tuple

from noisyvis.common.pareto import hypervolume_of, nondominated_mask, replay_nondominated_archive

from .logger import get_active_logger, set_active_logger

HV_METRICS = ("current_noisy_front__noisy", "current_noisy_front__clean", "current_clean_front__clean")
ARCHIVE_HV_METRICS = ("noisy_archive__noisy", "noisy_archive__clean", "clean_archive__clean")


class EvaluationLogError(RuntimeError):
    """The exactly-one-log contract of a genuine evaluation was broken."""


class ProvenanceError(RuntimeError):
    """An individual's _eval_tag does not match its current state or this run's evaluation log."""


class MOEvaluation(NamedTuple):
    """Read-only view of one logged evaluation."""
    eval_id: int
    orig_geno: int
    eval_geno: int
    obs_id: int
    x: tuple
    x_tilde: tuple
    true_objectives: tuple
    observed: tuple


class MOEvaluationLogger:
    """
    Run-scoped, columnar log of genuine MO evaluations (one row per eval_id), with genotype interning
    and canonical observation ids.

    Genotypes are interned by their raw tuple of genes (no int() cast, so real-valued genes keep their
    values). Numerically equal genes of different Python types (1, 1.0, True, np.int64(1)) compare and
    hash equal and so intern to the same genotype. Interning order is not discovery order: a genotype
    may be interned first as some evaluation's x~ and only later appear as an x (orig_geno).
    """

    def __init__(self, weights, ref_point=None):
        self.weights = tuple(weights)
        self.ref_point = None if ref_point is None else tuple(ref_point)

        # genotype interning: geno_id -> genotype, and back
        self.genotypes = []
        self._geno_index = {}

        # event columns, indexed by eval_id
        self.orig_geno = []
        self.eval_geno = []
        self.true_objectives = []
        self.observed = []
        self.obs_id = []

        # canonical observations: (orig_geno, eval_geno, observed) -> obs_id; first eval of each obs_id
        self._obs_index = {}
        self.obs_first_eval = []

        # an eval_id whose x is each genotype (the first), to look up f(x) by genotype
        self._orig_eval_by_geno = {}

        # log_mo_eval calls of the evaluation in progress; None when no evaluation is open
        self._open = None

        # generation boundaries: gen_last_eval[g] is the exclusive end offset of generation g's events
        self.gen_last_eval = []

        # current-front change-point histories, one row per membership change
        self.noisy_front_gens = []      # generation each row starts at
        self.noisy_front_members = []   # sorted tuple of obs_ids
        self.noisy_front_hv_noisy = []  # current_noisy_front__noisy
        self.noisy_front_hv_clean = []  # current_noisy_front__clean
        self.clean_front_gens = []
        self.clean_front_members = []   # sorted tuple of orig_geno ids
        self.clean_front_hv = []        # current_clean_front__clean

        # passive archives: replayed from the events on demand, cached by the event count they cover
        self._archive_cache = None

    def __len__(self):
        return len(self.orig_geno)

    # ------------------------------------------------------------------ interning

    def intern(self, genes) -> int:
        """Genotype id of `genes`, assigning the next id to a genotype not seen before."""
        key = tuple(genes)
        geno_id = self._geno_index.get(key)
        if geno_id is None:
            geno_id = len(self.genotypes)
            self._geno_index[key] = geno_id
            self.genotypes.append(key)
        return geno_id

    # ------------------------------------------------------------------ the scoped evaluation

    def evaluate(self, fn, individual, kwargs) -> Tuple[tuple, int]:
        """
        One genuine evaluation: call fn(individual, **kwargs) with this logger active, require that the
        evaluator reported it exactly once through log_mo_eval, and commit the event.

        Returns (y exactly as fn returned it, eval_id). If fn raises, nothing is committed and the
        exception propagates; the previous active logger is restored in every case.
        """
        active = get_active_logger()
        if self._open is not None or getattr(active, "_open", None) is not None:
            raise EvaluationLogError("nested MO evaluation: an evaluation is already in progress")

        x = tuple(individual)
        self._open = []
        set_active_logger(self)
        try:
            observed = fn(individual, **kwargs)
        finally:
            set_active_logger(active)
            staged, self._open = self._open, None

        if len(staged) != 1:
            raise EvaluationLogError(
                f"an evaluation must be logged exactly once through log_mo_eval; it was logged "
                f"{len(staged)} times (evaluator {getattr(fn, '__name__', fn)!r})"
            )
        if tuple(individual) != x:
            raise EvaluationLogError("the evaluator modified the individual it evaluated")
        x_tilde, true_objectives, logged = staged[0]
        if logged != tuple(observed):
            raise EvaluationLogError(
                f"the evaluator logged y={logged!r} but returned {observed!r}"
            )
        return observed, self._commit(x, x_tilde, true_objectives, logged)

    def log_mo_eval(self, evaluated, true_objectives, observed):
        """
        Evaluator side-channel: the genotype actually evaluated (x~), the deterministic objectives of
        the original solution f(x), and the observed vector y it is about to return.
        """
        if self._open is None:
            raise EvaluationLogError("log_mo_eval called outside a genuine evaluation")
        self._open.append((tuple(evaluated), tuple(true_objectives), tuple(observed)))

    def _commit(self, x, x_tilde, true_objectives, observed) -> int:
        eval_id = len(self.orig_geno)
        orig_geno = self.intern(x)
        eval_geno = self.intern(x_tilde)

        key = (orig_geno, eval_geno, observed)
        obs_id = self._obs_index.get(key)
        if obs_id is None:
            obs_id = len(self.obs_first_eval)
            self._obs_index[key] = obs_id
            self.obs_first_eval.append(eval_id)

        self._orig_eval_by_geno.setdefault(orig_geno, eval_id)
        self.orig_geno.append(orig_geno)
        self.eval_geno.append(eval_geno)
        self.true_objectives.append(true_objectives)
        self.observed.append(observed)
        self.obs_id.append(obs_id)
        return eval_id

    # ------------------------------------------------------------------ provenance

    def _weighted(self, values, weights=None) -> tuple:
        # deap.base.Fitness.setValues: wvalues = tuple(map(mul, values, weights))
        return tuple(map(mul, values, self.weights if weights is None else weights))

    def tag(self, individual, eval_id: int):
        """Attach (eval_id, orig_geno_id, wvalues of y) to the individual just evaluated."""
        individual._eval_tag = (
            eval_id,
            self.orig_geno[eval_id],
            self._weighted(self.observed[eval_id], individual.fitness.weights),
        )

    def resolve(self, individual) -> int:
        """
        The eval_id of the evaluation that produced the individual's current genotype and fitness.

        Raises ProvenanceError if the individual has no tag, if its tag is inconsistent with this log
        (corrupt, or from another run's log and disagreeing with this one), or if its genotype or fitness
        changed since that evaluation without a genuine re-evaluation. The tag carries no run identity,
        so call this only on the logger of the run that owns the individual.
        """
        tag = getattr(individual, "_eval_tag", None)
        if tag is None:
            raise ProvenanceError("individual has no _eval_tag: it was never evaluated under this logger")
        eval_id, orig_geno, wvalues = tag

        if not (0 <= eval_id < len(self)) or self.orig_geno[eval_id] != orig_geno:
            raise ProvenanceError(
                f"_eval_tag {tag!r} does not match this run's evaluation log (corrupt or foreign tag)"
            )
        if self._weighted(self.observed[eval_id]) != tuple(wvalues):
            raise ProvenanceError(
                f"_eval_tag wvalues {wvalues!r} do not match evaluation {eval_id}'s observed objectives "
                f"{self.observed[eval_id]!r}"
            )
        if tuple(individual) != self.genotypes[orig_geno]:
            raise ProvenanceError(
                f"genotype changed since evaluation {eval_id} without a genuine re-evaluation (stale _eval_tag)"
            )
        if not individual.fitness.valid or tuple(individual.fitness.wvalues) != tuple(wvalues):
            raise ProvenanceError(
                f"fitness changed since evaluation {eval_id} without a genuine re-evaluation (stale _eval_tag)"
            )
        return eval_id

    def event(self, eval_id: int) -> MOEvaluation:
        return MOEvaluation(
            eval_id=eval_id,
            orig_geno=self.orig_geno[eval_id],
            eval_geno=self.eval_geno[eval_id],
            obs_id=self.obs_id[eval_id],
            x=self.genotypes[self.orig_geno[eval_id]],
            x_tilde=self.genotypes[self.eval_geno[eval_id]],
            true_objectives=self.true_objectives[eval_id],
            observed=self.observed[eval_id],
        )

    # ------------------------------------------------------------------ observations and genotypes

    def obs_orig_geno(self, obs_id: int) -> int:
        return self.orig_geno[self.obs_first_eval[obs_id]]

    def obs_eval_geno(self, obs_id: int) -> int:
        return self.eval_geno[self.obs_first_eval[obs_id]]

    def obs_observed(self, obs_id: int) -> tuple:
        return self.observed[self.obs_first_eval[obs_id]]

    def true_objectives_of(self, geno_id: int) -> tuple:
        """f(x) of a genotype that has been generated as an x (orig_geno) at least once."""
        return self.true_objectives[self._orig_eval_by_geno[geno_id]]

    # ------------------------------------------------------------------ current fronts

    @property
    def n_generations(self) -> int:
        return len(self.gen_last_eval)

    def observe_generation(self, gen: int, population):
        """
        Record generation `gen` (observed once each, in order from 0): its evaluation boundary and the
        current noisy and clean fronts of `population`. Reads only: no RNG, no evaluation, no change to
        any individual.
        """
        if gen != len(self.gen_last_eval):
            raise EvaluationLogError(
                f"generation {gen} observed out of order: generations 0..{len(self.gen_last_eval) - 1} "
                f"are recorded, so the next must be {len(self.gen_last_eval)}"
            )
        if self._open is not None:
            raise EvaluationLogError("a generation cannot be observed while an evaluation is in progress")

        eval_ids = [self.resolve(ind) for ind in population]

        # current noisy front: ND of the population's distinct observations under the observed y
        obs = sorted({self.obs_id[e] for e in eval_ids})
        keep = nondominated_mask([self._weighted(self.obs_observed(o)) for o in obs])
        noisy = tuple(o for o, k in zip(obs, keep) if k)

        # current clean front, independently: ND of the population's distinct genotypes under f(x)
        for e in eval_ids:
            if self.true_objectives[e] != self.true_objectives_of(self.orig_geno[e]):
                raise EvaluationLogError(
                    f"f(x) is not deterministic: genotype {self.orig_geno[e]} has true objectives "
                    f"{self.true_objectives_of(self.orig_geno[e])!r} and {self.true_objectives[e]!r}"
                )
        genos = sorted({self.orig_geno[e] for e in eval_ids})
        keep = nondominated_mask([self._weighted(self.true_objectives_of(g)) for g in genos])
        clean = tuple(g for g, k in zip(genos, keep) if k)

        # everything validated: record (nothing above changes the log)
        self.gen_last_eval.append(len(self))  # exclusive end offset of this generation's events
        if not self.noisy_front_members or noisy != self.noisy_front_members[-1]:
            self.noisy_front_gens.append(gen)
            self.noisy_front_members.append(noisy)
            self.noisy_front_hv_noisy.append(
                hypervolume_of([self.obs_observed(o) for o in noisy], self.weights, self.ref_point))
            # one re-scored point per member (duplicate genotypes included), as the legacy recorder
            self.noisy_front_hv_clean.append(hypervolume_of(
                [self.true_objectives_of(self.obs_orig_geno(o)) for o in noisy], self.weights, self.ref_point))
        if not self.clean_front_members or clean != self.clean_front_members[-1]:
            self.clean_front_gens.append(gen)
            self.clean_front_members.append(clean)
            self.clean_front_hv.append(
                hypervolume_of([self.true_objectives_of(g) for g in clean], self.weights, self.ref_point))

    def _row_at(self, gens, gen: int) -> int:
        """Index of the change-point row in force at generation `gen`."""
        if not 0 <= gen < len(self.gen_last_eval):
            raise IndexError(f"generation {gen} is not recorded (0..{len(self.gen_last_eval) - 1})")
        return bisect_right(gens, gen) - 1

    def noisy_front_at(self, gen: int) -> tuple:
        """obs_ids of the current noisy front at generation `gen`."""
        return self.noisy_front_members[self._row_at(self.noisy_front_gens, gen)]

    def clean_front_at(self, gen: int) -> tuple:
        """orig_geno ids of the current clean front at generation `gen`."""
        return self.clean_front_members[self._row_at(self.clean_front_gens, gen)]

    def hv_trajectory(self, metric: str) -> list:
        """One value per recorded generation 0..n_generations-1 of a current-front HV metric."""
        if metric == "current_noisy_front__noisy":
            gens, values = self.noisy_front_gens, self.noisy_front_hv_noisy
        elif metric == "current_noisy_front__clean":
            gens, values = self.noisy_front_gens, self.noisy_front_hv_clean
        elif metric == "current_clean_front__clean":
            gens, values = self.clean_front_gens, self.clean_front_hv
        else:
            raise KeyError(f"unknown metric {metric!r}; expected one of {HV_METRICS}")
        return [values[self._row_at(gens, g)] for g in range(self.n_generations)]

    # ------------------------------------------------------------------ passive historical archives

    def _archives(self) -> dict:
        """
        Both archives replayed from the events, with their HV change rows. A pure function of the event
        columns, cached by the number of events it covers (generation views are derived on each call).
        """
        n = len(self)
        if self._archive_cache is not None and self._archive_cache[0] == n:
            return self._archive_cache[1]

        # f(x) is recorded for each event's original genotype, so determinism is checked per orig_geno
        # only (an event where a genotype appears only as x~ carries f of its x, not of that genotype)
        for e in range(n):
            first = self._orig_eval_by_geno[self.orig_geno[e]]
            if self.true_objectives[e] != self.true_objectives[first]:
                raise EvaluationLogError(
                    f"f(x) is not deterministic: genotype {self.orig_geno[e]} has true objectives "
                    f"{self.true_objectives[first]!r} (eval {first}) and {self.true_objectives[e]!r} (eval {e})"
                )

        # noisy: each canonical observation offered once, at its first event
        noisy = replay_nondominated_archive(
            (e, self.obs_id[e], self._weighted(self.observed[e]))
            for e in range(n) if self.obs_first_eval[self.obs_id[e]] == e
        )
        # clean: each genotype offered once, at the first event whose x it is
        clean = replay_nondominated_archive(
            (e, self.orig_geno[e], self._weighted(self.true_objectives[e]))
            for e in range(n) if self._orig_eval_by_geno[self.orig_geno[e]] == e
        )
        noisy_evals, (noisy_noisy, noisy_clean) = self._archive_hv_rows(
            noisy,
            self.obs_observed,
            lambda obs: self.true_objectives_of(self.obs_orig_geno(obs)),
        )
        clean_evals, (clean_clean,) = self._archive_hv_rows(clean, self.true_objectives_of)
        archives = {
            "noisy": noisy,
            "clean": clean,
            "hv": {
                "noisy_archive__noisy": (noisy_evals, noisy_noisy),
                "noisy_archive__clean": (noisy_evals, noisy_clean),
                "clean_archive__clean": (clean_evals, clean_clean),
            },
        }
        self._archive_cache = (n, archives)
        return archives

    def _archive_hv_rows(self, replay, *point_of):
        """At each membership change: the eval_id, and the HV of the members' points under each point_of."""
        current, evals, columns = set(), [], [[] for _ in point_of]
        for eval_id, entered, evicted in replay.changes:
            current.difference_update(evicted)
            current.add(entered)
            members = sorted(current)
            evals.append(eval_id)
            for column, point in zip(columns, point_of):
                column.append(hypervolume_of([point(m) for m in members], self.weights, self.ref_point))
        return evals, columns

    @property
    def noisy_archive(self):
        """Noisy-archive membership intervals (common.pareto.ArchiveReplay over obs_ids)."""
        return self._archives()["noisy"]

    @property
    def clean_archive(self):
        """Clean-archive membership intervals (common.pareto.ArchiveReplay over genotype ids)."""
        return self._archives()["clean"]

    def _events(self, n_events: int) -> int:
        if not 0 <= n_events <= len(self):
            raise IndexError(f"{n_events} events requested; the log holds {len(self)}")
        return n_events

    def _eval(self, eval_id: int) -> int:
        if not 0 <= eval_id < len(self):
            raise IndexError(f"eval_id {eval_id} is not in the log (0..{len(self) - 1})")
        return eval_id

    def _generation_end(self, gen: int) -> int:
        # always the current gen_last_eval: a generation may add no evaluations
        if not 0 <= gen < len(self.gen_last_eval):
            raise IndexError(f"generation {gen} is not recorded (0..{len(self.gen_last_eval) - 1})")
        return self.gen_last_eval[gen]

    def noisy_archive_after(self, n_events: int) -> tuple:
        """obs_ids in the noisy archive after the first n_events events."""
        return self.noisy_archive.members_after(self._events(n_events))

    def clean_archive_after(self, n_events: int) -> tuple:
        """Genotype ids in the clean archive after the first n_events events."""
        return self.clean_archive.members_after(self._events(n_events))

    def noisy_archive_at_eval(self, eval_id: int) -> tuple:
        """The noisy archive just after evaluation eval_id."""
        return self.noisy_archive_after(self._eval(eval_id) + 1)

    def clean_archive_at_eval(self, eval_id: int) -> tuple:
        """The clean archive just after evaluation eval_id."""
        return self.clean_archive_after(self._eval(eval_id) + 1)

    def noisy_archive_at(self, gen: int) -> tuple:
        """The noisy archive at the end of generation gen (all evaluations up to gen_last_eval[gen])."""
        return self.noisy_archive_after(self._generation_end(gen))

    def clean_archive_at(self, gen: int) -> tuple:
        """The clean archive at the end of generation gen."""
        return self.clean_archive_after(self._generation_end(gen))

    def archive_hv_after(self, metric: str, n_events: int):
        """An archive HV metric after the first n_events events (None before the first member)."""
        if metric not in ARCHIVE_HV_METRICS:
            raise KeyError(f"unknown metric {metric!r}; expected one of {ARCHIVE_HV_METRICS}")
        evals, values = self._archives()["hv"][metric]
        row = bisect_right(evals, self._events(n_events) - 1) - 1  # last change at an eval < n_events
        return values[row] if row >= 0 else None

    def archive_hv_trajectory(self, metric: str) -> list:
        """One value per recorded generation 0..n_generations-1 of an archive HV metric."""
        return [self.archive_hv_after(metric, self._generation_end(g)) for g in range(self.n_generations)]

    # ------------------------------------------------------------------ persistence (Work Group 5)

    def finalise(self) -> dict:
        """
        Freeze the log into a plain-data mo_record (noisyvis.results.mo_view reads it): NumPy arrays and
        Python primitives only. Replays the passive archives at most once (the cached replay), validates
        the log, and returns a new record on every call; the records of repeated calls are equal. No
        evaluation, no RNG, no change to the algorithm or to the logged data. Must be called at a
        generation boundary (after run(), every event belongs to an observed generation).
        """
        import numpy as np
        from noisyvis.results.mo_view import MO_RECORD_SCHEMA, MO_RECORD_VERSION

        if self._open is not None:
            raise EvaluationLogError("cannot finalise while an evaluation is in progress")
        if not self.gen_last_eval or self.gen_last_eval[-1] != len(self):
            raise EvaluationLogError(
                "finalise at a generation boundary: every event must belong to an observed generation "
                f"({len(self)} events, last boundary {self.gen_last_eval[-1] if self.gen_last_eval else None})"
            )
        # canonical observation ids: first-appearance numbering of the distinct (x, x~, y)
        expected = {}
        for e in range(len(self)):
            key = (self.orig_geno[e], self.eval_geno[e], self.observed[e])
            if self.obs_id[e] != expected.setdefault(key, len(expected)):
                raise EvaluationLogError(f"obs_id of eval {e} is not canonical")
        archives = self._archives()  # also checks that f(x) is deterministic per genotype

        n, n_obj = len(self), len(self.weights)
        genotypes = np.asarray(self.genotypes)
        if genotypes.dtype.kind in "biu" and np.isin(genotypes, (0, 1)).all():
            genotypes = genotypes.astype(np.uint8)
        elif genotypes.dtype.kind in "biu":
            genotypes = genotypes.astype(np.int64)
        else:
            genotypes = genotypes.astype(np.float64)
        true_obj = np.full((len(self.genotypes), n_obj), np.nan)
        first_orig_eval = np.full(len(self.genotypes), -1, dtype=np.int32)
        for geno, eval_id in self._orig_eval_by_geno.items():
            true_obj[geno] = self.true_objectives[eval_id]
            first_orig_eval[geno] = eval_id

        orig_geno = np.asarray(self.orig_geno, dtype=np.int32)
        eval_geno = np.asarray(self.eval_geno, dtype=np.int32)
        prior_noise = bool((eval_geno != orig_geno).any())

        def hv_array(values):
            return np.array([np.nan if v is None else v for v in values], dtype=np.float64)

        def csr(rows):
            offsets = np.zeros(len(rows) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(r) for r in rows])
            return offsets, np.array([m for r in rows for m in r], dtype=np.int32)

        def archive_block(replay, hv_rows):
            block = {
                "member": np.array(replay.members, dtype=np.int32),
                "enter_eval": np.array(replay.enter, dtype=np.int32),
                "exit_eval": np.array(replay.exit, dtype=np.int32),
                "exit_by": np.array(replay.exit_by, dtype=np.int32),
                "hv_eval": np.array(hv_rows[0][0], dtype=np.int32),
            }
            for key, (_, values) in hv_rows[1].items():
                block[key] = hv_array(values)
            return block

        noisy_offsets, noisy_members = csr(self.noisy_front_members)
        clean_offsets, clean_members = csr(self.clean_front_members)
        hv = archives["hv"]
        return {
            "schema": MO_RECORD_SCHEMA,
            "version": MO_RECORD_VERSION,
            "meta": {
                "opt_weights": tuple(float(w) for w in self.weights),
                "ref_point": None if self.ref_point is None else tuple(float(r) for r in self.ref_point),
                "n_obj": n_obj,
                "sol_length": int(genotypes.shape[1]),
                "gene_dtype": str(genotypes.dtype),
                "n_evals": n,
                "n_generations": len(self.gen_last_eval),
                "n_genotypes": len(self.genotypes),
                "n_observations": len(self.obs_first_eval),
                "prior_noise_observed": prior_noise,
            },
            "genotypes": {"values": genotypes, "true_obj": true_obj, "first_orig_eval": first_orig_eval},
            "events": {
                "orig_geno": orig_geno,
                "eval_geno": eval_geno if prior_noise else None,
                "obs_obj": np.asarray(self.observed, dtype=np.float64).reshape(n, n_obj),
                "obs_id": np.asarray(self.obs_id, dtype=np.int32),
            },
            "gen_last_eval": np.asarray(self.gen_last_eval, dtype=np.int64),
            "current_noisy_front": {
                "gen": np.asarray(self.noisy_front_gens, dtype=np.int32),
                "member_offsets": noisy_offsets,
                "members": noisy_members,
                "hv_noisy": hv_array(self.noisy_front_hv_noisy),
                "hv_clean": hv_array(self.noisy_front_hv_clean),
            },
            "current_clean_front": {
                "gen": np.asarray(self.clean_front_gens, dtype=np.int32),
                "member_offsets": clean_offsets,
                "members": clean_members,
                "hv_clean": hv_array(self.clean_front_hv),
            },
            "noisy_archive": archive_block(archives["noisy"], (
                hv["noisy_archive__noisy"],
                {"hv_noisy": hv["noisy_archive__noisy"], "hv_clean": hv["noisy_archive__clean"]})),
            "clean_archive": archive_block(archives["clean"], (
                hv["clean_archive__clean"], {"hv_clean": hv["clean_archive__clean"]})),
        }
