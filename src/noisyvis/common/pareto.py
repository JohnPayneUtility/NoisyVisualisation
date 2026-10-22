"""
Pure Pareto-front utilities shared by the multi-objective algorithms (front stagnation), the MO
evaluation logger (current fronts and passive historical archives) and the MO read view (population
ranks). No RNG, no evaluation, no deap Individuals required.

Dominance throughout is Pareto dominance on weighted values (every objective maximised), as
deap.base.Fitness.dominates: a dominates b iff a is no worse in every objective and strictly better in at
least one. Equal vectors never dominate each other.
"""

import functools
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from deap import base as deap_base
from deap import tools as deap_tools

try: # try import fast hypervolume else fallback to python implementation
    from deap.tools._hypervolume import hv as _hv # fast compiled version
    hypervolume = _hv.hypervolume
except Exception:
    from deap.tools._hypervolume.pyhv import hypervolume # python version


def nondominated_mask(wvalues):
    """
    Boolean mask of the rows of an (N, M) array of weighted objective values (deap wvalues, so every
    objective is maximised) that no other row dominates. Pareto dominance, as
    deap.base.Fitness.dominates: equal rows do not dominate each other. Draws no RNG.
    """
    w = np.asarray(wvalues, dtype=float)
    if w.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    # dominated_by[i, j]: row j dominates row i
    dominated_by = (w[None, :, :] >= w[:, None, :]).all(axis=2) & (w[None, :, :] > w[:, None, :]).any(axis=2)
    return ~dominated_by.any(axis=1)


# ==============================
# Non-dominated ranks (DEAP's sorts)
# ==============================

_SORTS = {"log": deap_tools.sortLogNondominated, "standard": deap_tools.sortNondominated}


@functools.lru_cache(maxsize=None)
def _fitness_class(weights):
    # a deap Fitness with these weights, built directly: no creator registration, no global state
    return type("RankFitness", (deap_base.Fitness,), {"weights": weights})


class _Ranked:
    """One input row for DEAP's sorts: its Fitness and its row index."""
    __slots__ = ("fitness", "row")

    def __init__(self, fitness, row):
        self.fitness = fitness
        self.row = row


def nondominated_ranks(objectives, weights, nd="log"):
    """
    0-based non-dominated rank of each row of an (N, M) array of (unweighted) objective values, in row
    order: rank 0 is the non-dominated rows, rank 1 those non-dominated once rank 0 is removed, and so on.
    Each row's weighted values are computed by deap's Fitness (values times weights, every objective then
    maximised), and the ranks are DEAP's front indices: nd="log" uses tools.sortLogNondominated,
    nd="standard" tools.sortNondominated (the sort selNSGA2 uses by default). Both give the same ranks.
    Equal rows share a rank, so duplicate rows are all kept and ranked alike. Draws no RNG.

    Dominance is plain Pareto dominance, as nondominated_mask: ranks == 0 is exactly its mask. Ranking under
    constraint-domination would need constraint data per row and constraint-aware fitness objects; choosing
    nd="standard" alone does not provide it.
    """
    if nd not in _SORTS:
        raise ValueError(f"nd must be one of {tuple(_SORTS)}, not {nd!r}")
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"objectives must be an (N, M) array, got shape {values.shape}")
    ranks = np.zeros(len(values), dtype=np.int64)
    if len(values) == 0:
        return ranks
    fitness = _fitness_class(tuple(float(w) for w in weights))
    rows = [_Ranked(fitness(row), i) for i, row in enumerate(values.tolist())]
    for rank, front in enumerate(_SORTS[nd](rows, len(rows))):
        for ranked in front:
            ranks[ranked.row] = rank
    return ranks


def hypervolume_of(values, opt_weights, ref_point):
    """
    Hypervolume of a set of (unweighted) objective vectors, with the MO recorder's convention: every
    objective with a positive weight is negated so all are minimised, and the reference point is flipped
    the same way (only the sign of each weight is used). Returns None without a reference point and 0.0
    for an empty set. The points are passed as given (duplicates and dominated points included).
    """
    if ref_point is None:
        return None
    if len(values) == 0:
        return 0.0
    w = np.asarray(opt_weights, dtype=float)
    sign = np.where(w > 0, -1.0, 1.0)  # flip max->min for HV
    hv_ref = np.asarray(ref_point, dtype=float) * sign
    return float(hypervolume(np.asarray(values, dtype=float) * sign, hv_ref))


# ==============================
# Non-dominated archive replay
# ==============================

NOT_EXITED = -1  # exit / exit_by of a member still in the archive


@dataclass(frozen=True)
class ArchiveReplay:
    """
    Membership history of a non-dominated archive over a stream of candidates, at evaluation precision.

    Row i describes members[i]: it entered at eval enter[i] and was evicted at eval exit[i] by the
    candidate exit_by[i] (NOT_EXITED for both while it is still a member). Intervals are half-open: the
    member is in the archive after event e iff enter <= e and (exit == NOT_EXITED or e < exit).

    changes lists every membership change in order: (eval_id, entered member, evicted members).
    """
    members: Tuple[int, ...]
    enter: Tuple[int, ...]
    exit: Tuple[int, ...]
    exit_by: Tuple[int, ...]
    changes: Tuple[Tuple[int, int, Tuple[int, ...]], ...]

    def members_after(self, n_events: int) -> tuple:
        """Sorted member ids of the archive after the first n_events events (evals 0..n_events-1)."""
        return tuple(sorted(
            m for m, enter, exit in zip(self.members, self.enter, self.exit)
            if enter < n_events and (exit == NOT_EXITED or exit >= n_events)
        ))


def _dominates(a, b) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def replay_nondominated_archive(candidates) -> ArchiveReplay:
    """
    Replay candidates (eval_id, member_id, wpoint), in evaluation order and each member_id at most once,
    through a non-dominated archive. A candidate that a current member dominates never enters; otherwise
    it enters and evicts every current member it dominates. Candidates with equal points are distinct
    members. After each candidate the archive is exactly ND(all candidates offered so far).
    """
    members, enter, exit, exit_by, changes = [], [], [], [], []
    current = {}  # member id -> (row, wpoint)
    for eval_id, member, point in candidates:
        point = tuple(point)
        if any(_dominates(other, point) for _, other in current.values()):
            continue
        evicted = tuple(m for m, (_, other) in current.items() if _dominates(point, other))
        for m in evicted:
            row, _ = current.pop(m)
            exit[row] = eval_id
            exit_by[row] = member
        current[member] = (len(members), point)
        members.append(member)
        enter.append(eval_id)
        exit.append(NOT_EXITED)
        exit_by.append(NOT_EXITED)
        changes.append((eval_id, member, evicted))
    return ArchiveReplay(tuple(members), tuple(enter), tuple(exit), tuple(exit_by), tuple(changes))
