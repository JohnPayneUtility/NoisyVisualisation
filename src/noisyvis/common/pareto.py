"""
Pure Pareto-front utilities shared by the multi-objective algorithms (front stagnation) and the MO
evaluation logger (current fronts and passive historical archives). No RNG, no evaluation, no deap
Individuals required.

Dominance throughout is Pareto dominance on weighted values (every objective maximised), as
deap.base.Fitness.dominates: a dominates b iff a is no worse in every objective and strictly better in at
least one. Equal vectors never dominate each other.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

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
