"""
Server-side multi-objective plot data (MO recording plan v6, Work Group 6).

Result rows keep their full MO record (`mo_record`) on the server, in `dashboard.data.df`. Browser stores
only ever carry a row's metadata and its stable row id (`_row`, the index of `df`); the callbacks resolve
the record here, open it with `results.mo_view.MORunView`, and build just the plot data they need.

Plot entries are recorded where the current noisy front's genotype set changes, as the plots have always
consumed them; each entry carries its real generation (`gen_idx`) and, separately, its position in that
sequence of changes (`change_idx`). Front members are listed in descending weighted-objective order.

Semantics (population-based current fronts, event-based passive archives) are those of MORunView.
"""

from __future__ import annotations

import math

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


def row_view(df: pd.DataFrame, row_id) -> MORunView | None:
    """The MORunView of one row of the server-side frame, or None for a row without an MO record."""
    record = df.at[row_id, 'mo_record'] if 'mo_record' in df.columns else None
    return MORunView(record) if isinstance(record, dict) else None


# ------------------------------------------------------------------ plot entries

def change_generations(view: MORunView) -> list:
    """Generations where the current noisy front's genotype set changes (generation 0 always)."""
    generations, last = [], None
    for g in range(view.n_generations):
        signature = frozenset(tuple(int(v) for v in x) for x in view.current_noisy_front(g).x)
        if signature != last:
            generations.append(g)
            last = signature
    return generations


def _order(values, weights) -> list:
    wvalues = np.asarray(values, dtype=float) * np.asarray(weights, dtype=float)
    return sorted(range(len(wvalues)), key=lambda i: tuple(wvalues[i]), reverse=True)


def _solutions(x, order) -> list:
    return [[int(v) for v in x[i]] for i in order]


def _points(values, order) -> list:
    return [[float(v) for v in values[i]] for i in order]


def front_entries(view: MORunView) -> list:
    """
    One entry per change of the current noisy front's genotype set:
        algo_front_*    the current noisy front: its solutions x, observed y, the same members under
                        f(x), and current_noisy_front__noisy / __clean
        clean_front_*   the current clean front (independently non-dominated under f(x)) and its HV
        gen_idx         the real generation; change_idx its position among the changes
    """
    entries = []
    for change_idx, gen in enumerate(change_generations(view)):
        noisy, clean = view.current_noisy_front(gen), view.current_clean_front(gen)
        n_order, c_order = _order(noisy.observed, view.weights), _order(clean.true_obj, view.weights)
        entries.append({
            'algo_front_solutions': _solutions(noisy.x, n_order),
            'algo_front_noisy_fitnesses': _points(noisy.observed, n_order),
            'algo_front_clean_fitnesses': _points(noisy.true_obj, n_order),
            'algo_front_noisy_hypervolume': noisy.hv['current_noisy_front__noisy'],
            'algo_front_clean_hypervolume': noisy.hv['current_noisy_front__clean'],
            'clean_front_solutions': _solutions(clean.x, c_order),
            'clean_front_fitnesses': _points(clean.true_obj, c_order),
            'clean_front_hypervolume': clean.hv['current_clean_front__clean'],
            'gen_idx': gen,
            'change_idx': change_idx,
        })
    return entries


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
