"""
Server-side multi-objective plot data (MO recording plan v6, Work Group 6).

Result rows keep their full MO record (`mo_record`) on the server, in `dashboard.data.df`. Browser stores
only ever carry a row's metadata and its stable row id (`_row`, the index of `df`); the callbacks resolve
the record here, open it with `results.mo_view.MORunView`, and build just the plot data they need.

Plot entries are recorded where the selected front changes (`sequence_entries`): by default the current
noisy front's genotype set, as the plots have always consumed them, or the current clean front, or either
passive archive (the Pareto tab's historical option); each entry carries its real generation (`gen_idx`)
and, separately, its position in that sequence of changes (`change_idx`). Front members are listed in
descending weighted-objective order.

Semantics (population-based current fronts, event-based passive archives) are those of MORunView.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..results.mo_view import MORunView

# The legacy recorder's per-row payload (removed in Work Group 6). Rows written before then carry it
# instead of `mo_record`; the dashboard skips them (regenerate those experiments to see them).
LEGACY_MO_COLUMNS = [
    'pareto_solutions', 'pareto_fitnesses', 'pareto_true_fitnesses', 'true_pareto_solutions',
    'true_pareto_fitnesses', 'noisy_pf_noisy_hypervolumes', 'noisy_pf_true_hypervolumes',
    'true_pf_hypervolumes', 'n_gens_pareto_best', 'final_true_hv', 'max_true_hv', 'min_true_hv',
    'final_noisy_pf_hv', 'max_noisy_pf_hv', 'min_noisy_pf_hv',
]


def _has_value(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def without_legacy_mo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the legacy MO rows and columns, row by row: a row is legacy when its own legacy payload is present
    and its own `mo_record` is absent (old and new rows can share one frame, so both columns may exist).
    The result is re-indexed so its index is the stable row id.
    """
    legacy = df['pareto_solutions'].map(_has_value) if 'pareto_solutions' in df.columns \
        else pd.Series(False, index=df.index)
    has_record = df['mo_record'].map(lambda v: isinstance(v, dict)) if 'mo_record' in df.columns \
        else pd.Series(False, index=df.index)
    skip = legacy & ~has_record
    if skip.any():
        print(f"[dashboard] skipping {int(skip.sum())} legacy multi-objective rows without mo_record "
              "(regenerate those experiments to view them)", flush=True)
    return df[~skip].drop(columns=LEGACY_MO_COLUMNS, errors='ignore').reset_index(drop=True)


class _IdentityCache:
    """
    A small LRU of values derived from an object (a record or a view), keyed by that object's identity. Each
    entry holds the object itself, so its id cannot be reused by another object while the entry is cached.
    Records and views are never modified, and callers only read the cached values.
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._items = OrderedDict()

    def get(self, owner, key, build):
        slot = (id(owner), key)
        hit = self._items.get(slot)
        if hit is not None and hit[0] is owner:
            self._items.move_to_end(slot)
            return hit[1]
        value = build()
        self._items[slot] = (owner, value)
        self._items.move_to_end(slot)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)
        return value


# Opened records (validation runs once per record) and built front sequences (per view, source and front)
_VIEWS = _IdentityCache(maxsize=512)
_SEQUENCES = _IdentityCache(maxsize=256)


def row_view(df: pd.DataFrame, row_id) -> MORunView | None:
    """The MORunView of one row of the server-side frame, or None for a row without an MO record."""
    record = df.at[row_id, 'mo_record'] if 'mo_record' in df.columns else None
    return _VIEWS.get(record, None, lambda: MORunView(record)) if isinstance(record, dict) else None


# ------------------------------------------------------------------ plot entries

def change_generations(view: MORunView) -> list:
    """
    Generations where the current noisy front's genotype set changes (generation 0 always). The front's
    membership only changes at its recorded change points, so those are the only candidates (a membership
    change need not change the genotype set).
    """
    generations, last = [], None
    for state in view.current_noisy_front_history():
        signature = frozenset(tuple(int(v) for v in x) for x in state.x)
        if signature != last:
            generations.append(int(state.generation))
            last = signature
    return generations


def _order(values, weights) -> list:
    wvalues = np.asarray(values, dtype=float) * np.asarray(weights, dtype=float)
    return sorted(range(len(wvalues)), key=lambda i: tuple(wvalues[i]), reverse=True)


def _solutions(x, order) -> list:
    return [[int(v) for v in x[i]] for i in order]


def _points(values, order) -> list:
    return [[float(v) for v in values[i]] for i in order]


def _hv(value) -> float:
    return float('nan') if value is None else float(value)


def _entry(noisy, clean, hvs, weights, gen, change_idx) -> dict:
    """One plot entry: a noisy front (or archive) and a clean front (or archive) at generation gen."""
    n_order, c_order = _order(noisy.observed, weights), _order(clean.true_obj, weights)
    hv_noisy, hv_cross, hv_clean = (_hv(v) for v in hvs)
    return {
        'algo_front_solutions': _solutions(noisy.x, n_order),
        'algo_front_noisy_fitnesses': _points(noisy.observed, n_order),
        'algo_front_clean_fitnesses': _points(noisy.true_obj, n_order),
        'algo_front_noisy_hypervolume': hv_noisy,
        'algo_front_clean_hypervolume': hv_cross,
        'clean_front_solutions': _solutions(clean.x, c_order),
        'clean_front_fitnesses': _points(clean.true_obj, c_order),
        'clean_front_hypervolume': hv_clean,
        'opt_weights': [float(w) for w in weights],
        'gen_idx': gen,
        'change_idx': change_idx,
    }


def _fronts(view: MORunView, source: str, gen: int) -> tuple:
    """(noisy, clean, (HV of noisy under y, HV of noisy under f(x), HV of clean)) at generation gen."""
    if source == 'historical':
        noisy, clean = view.noisy_archive(gen), view.clean_archive(gen)
        return noisy, clean, (noisy.hv['noisy_archive__noisy'], noisy.hv['noisy_archive__clean'],
                              clean.hv['clean_archive__clean'])
    noisy, clean = view.current_noisy_front(gen), view.current_clean_front(gen)
    return noisy, clean, (noisy.hv['current_noisy_front__noisy'], noisy.hv['current_noisy_front__clean'],
                          clean.hv['current_clean_front__clean'])


def sequence_generations(view: MORunView, source: str, front: str) -> list:
    """Generations where the selected front changes: the current noisy front's genotype set (as the plots
    have always consumed it), the current clean front's members, or a passive archive's members."""
    if source == 'current' and front == 'noisy':
        return change_generations(view)
    if source == 'current' and front == 'clean':
        history = view.current_clean_front_history()
    elif source == 'historical' and front == 'noisy':
        history = view.noisy_archive_history('generation')
    elif source == 'historical' and front == 'clean':
        history = view.clean_archive_history('generation')
    else:
        raise ValueError(f"unknown front sequence {source!r}/{front!r}")
    return [int(g) for g in history.starts]


def sequence_entries(view: MORunView, source: str, front: str) -> list:
    """
    One entry per change of the selected front, `source` 'current' (population fronts) or 'historical'
    (passive archives) and `front` 'noisy' or 'clean':
        algo_front_*    the noisy front (or noisy archive): its solutions x, observed y, the same members
                        under f(x), and its HVs under y and under f(x)
        clean_front_*   the clean front (or clean archive, independently non-dominated under f(x)) and its HV
        opt_weights     the objective weights (directions)
        gen_idx         the real generation; change_idx its position among the changes
    Each entry holds both fronts as they stood at the end of gen_idx. The last entry also carries
    final_clean_front_fitnesses: the clean front (or clean archive) at the run's final generation.
    Built once per view and sequence; callers must not modify the returned entries.
    """
    return _SEQUENCES.get(view, (source, front), lambda: _build_sequence(view, source, front))


def _build_sequence(view: MORunView, source: str, front: str) -> list:
    entries = []
    for change_idx, gen in enumerate(sequence_generations(view, source, front)):
        noisy, clean, hvs = _fronts(view, source, gen)
        entries.append(_entry(noisy, clean, hvs, view.weights, gen, change_idx))
    if entries:
        final = _fronts(view, source, view.n_generations - 1)[1]
        entries[-1]['final_clean_front_fitnesses'] = _points(final.true_obj, _order(final.true_obj, view.weights))
    return entries


def front_entries(view: MORunView) -> list:
    """The current noisy front's sequence (sequence_entries(view, 'current', 'noisy'))."""
    return sequence_entries(view, 'current', 'noisy')


def front_runs(df: pd.DataFrame, stn_rows, source: str, front: str, group_cols=('algo_name', 'noise')) -> list:
    """MO_data_PPP's shape (series -> runs -> entries) for one front sequence, grouped and ordered as
    process_STN_data groups the selected rows."""
    frame = pd.DataFrame(stn_rows or [])
    groups = []
    if frame.empty:
        return groups
    for _, group_df in frame.groupby(list(group_cols)):
        runs = []
        for _, row in group_df.iterrows():
            view = row_view(df, row['_row']) if '_row' in row and pd.notna(row['_row']) else None
            if view is None:
                continue
            entries = sequence_entries(view, source, front)
            if entries:
                runs.append(entries)
        groups.append(runs)
    return groups


def stn_entries(entries: list, mode: str) -> list:
    """The STN multi-objective nodes of one run, per plot mode (the dropdown's `mo_plot_type`)."""
    stn = []
    for e in entries:
        noisy, clean = e['algo_front_solutions'], e['clean_front_solutions']
        hv_noisy, hv_cross, hv_clean = (e['algo_front_noisy_hypervolume'], e['algo_front_clean_hypervolume'],
                                        e['clean_front_hypervolume'])
        if mode == 'bpbhv':            # both fronts, both metrics
            front1, front2, metric1, metric2 = clean, noisy, hv_clean, hv_noisy
        elif mode == 'bpbhv_algo_pov':  # both fronts, both metrics, from the algorithm's point of view
            front1, front2, metric1, metric2 = noisy, clean, hv_noisy, hv_clean
        elif mode == 'tpthv':           # clean front, clean HV
            front1, front2, metric1, metric2 = clean, None, hv_clean, None
        elif mode == 'npthv':           # noisy front, its members' clean HV
            front1, front2, metric1, metric2 = noisy, None, hv_cross, None
        elif mode == 'npbhv':           # noisy front, clean and noisy HV
            front1, front2, metric1, metric2 = noisy, None, hv_cross, hv_noisy
        elif mode == 'tpbhv':           # noisy front with the clean front's HV (not offered in the dropdown)
            front1, front2, metric1, metric2 = noisy, None, hv_clean, None
        else:                           # npnhv: noisy front, noisy HV
            front1, front2, metric1, metric2 = noisy, None, hv_noisy, None
        stn.append({'front1': front1, 'front2': front2, 'metric1': metric1, 'metric2': metric2,
                    'gen_idx': e['gen_idx'], 'change_idx': e['change_idx']})
    return stn


def archive_snapshot(view: MORunView, gen: int) -> dict:
    """
    The passive historical archives at generation gen, for server-side analysis: the noisy archive
    (best observations so far, under y and under f(x)), the clean archive (best genuinely generated
    solutions so far, under f(x)), and the clean-archive members the population no longer holds.
    """
    noisy, clean, lost = view.noisy_archive(gen), view.clean_archive(gen), view.lost_clean_solutions(gen)
    n_order, c_order = _order(noisy.observed, view.weights), _order(clean.true_obj, view.weights)
    l_order = _order(lost.true_obj, view.weights)
    return {
        'noisy_archive_solutions': _solutions(noisy.x, n_order),
        'noisy_archive_noisy_fitnesses': _points(noisy.observed, n_order),
        'noisy_archive_clean_fitnesses': _points(noisy.true_obj, n_order),
        'clean_archive_solutions': _solutions(clean.x, c_order),
        'clean_archive_fitnesses': _points(clean.true_obj, c_order),
        'lost_clean_solutions': _solutions(lost.x, l_order),
        'lost_clean_fitnesses': _points(lost.true_obj, l_order),
        **{metric: value for snapshot in (noisy, clean) for metric, value in snapshot.hv.items()},
        'gen_idx': gen,
    }
