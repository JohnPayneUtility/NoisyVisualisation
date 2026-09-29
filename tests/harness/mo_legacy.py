"""Test-only projection of an MO run record onto the removed legacy recorder's outputs (MO recording
plan v6, Work Group 6).

The legacy recorder (record_pareto_data, deleted in Work Group 6) wrote, per recorded entry, the current
noisy front (solutions, observed and true fitnesses), the clean front and three hypervolumes, and it
appended an entry only when the noisy front's genotype set changed (or every generation with
record_every_gen). The frozen reference-#3 baselines (baselines/mo.json, baselines/mo_algorithms.json)
hold those outputs. This module reproduces them from `MORunView` alone, so the baselines stay the
oracle without any legacy code in production.

Member order follows deap.tools.ParetoFront, which the legacy lists used: descending weighted values.
ParetoFront orders equal-valued distinct members by reverse insertion (population) order, which a
record does not hold; `tied_members` reports such ties so a caller can tell an ordering limit apart from
a real difference.
"""

from __future__ import annotations

import numpy as np

from noisyvis.results.mo_view import MORunView


def change_generations(view: MORunView, every_generation: bool = False) -> list:
    """Generations of the legacy entries: where the noisy front's genotype set changed (always g0)."""
    if every_generation:
        return list(range(view.n_generations))
    generations, last = [], None
    for g in range(view.n_generations):
        signature = frozenset(tuple(int(v) for v in x) for x in view.current_noisy_front(g).x)
        if signature != last:
            generations.append(g)
            last = signature
    return generations


def run_lengths(view: MORunView) -> list:
    """n_gens_pareto_best: generations between noisy-front genotype-set changes."""
    changes = change_generations(view)
    return [b - a for a, b in zip(changes, changes[1:] + [view.n_generations])]


def _pareto_front_order(values: np.ndarray, weights) -> list:
    wvalues = np.asarray(values, dtype=float) * np.asarray(weights, dtype=float)
    return sorted(range(len(wvalues)), key=lambda i: tuple(wvalues[i]), reverse=True)


def _tied(values: np.ndarray, weights) -> bool:
    wvalues = [tuple(row) for row in np.asarray(values, dtype=float) * np.asarray(weights, dtype=float)]
    return len(set(wvalues)) != len(wvalues)


def entry(view: MORunView, gen: int) -> dict:
    """The legacy entry at generation gen, members in ParetoFront order."""
    noisy, clean = view.current_noisy_front(gen), view.current_clean_front(gen)
    n_order = _pareto_front_order(noisy.observed, view.weights)
    c_order = _pareto_front_order(clean.true_obj, view.weights)
    return {
        "pareto_solutions": [[int(v) for v in noisy.x[i]] for i in n_order],
        "pareto_fitnesses": [[float(v) for v in noisy.observed[i]] for i in n_order],
        "pareto_true_fitnesses": [[float(v) for v in noisy.true_obj[i]] for i in n_order],
        "true_pareto_solutions": [[int(v) for v in clean.x[i]] for i in c_order],
        "true_pareto_fitnesses": [[float(v) for v in clean.true_obj[i]] for i in c_order],
        "noisy_pf_noisy_hypervolume": noisy.hv["current_noisy_front__noisy"],
        "noisy_pf_true_hypervolume": noisy.hv["current_noisy_front__clean"],
        "true_pf_hypervolume": clean.hv["current_clean_front__clean"],
        "tied_members": _tied(noisy.observed, view.weights) or _tied(clean.true_obj, view.weights),
    }


def legacy_columns(view: MORunView, every_generation: bool = False) -> dict:
    """The legacy recorder's list columns and scalar summaries for one run."""
    entries = [entry(view, g) for g in change_generations(view, every_generation)]
    columns = {key: [e[key] for e in entries] for key in (
        "pareto_solutions", "pareto_fitnesses", "pareto_true_fitnesses", "true_pareto_solutions",
        "true_pareto_fitnesses")}
    columns["noisy_pf_noisy_hypervolumes"] = [e["noisy_pf_noisy_hypervolume"] for e in entries]
    columns["noisy_pf_true_hypervolumes"] = [e["noisy_pf_true_hypervolume"] for e in entries]
    columns["true_pf_hypervolumes"] = [e["true_pf_hypervolume"] for e in entries]
    columns["n_gens_pareto_best"] = run_lengths(view)
    true_hv, noisy_pf_hv = columns["true_pf_hypervolumes"], columns["noisy_pf_true_hypervolumes"]
    columns.update(
        final_true_hv=true_hv[-1], max_true_hv=max(true_hv), min_true_hv=min(true_hv),
        final_noisy_pf_hv=noisy_pf_hv[-1], max_noisy_pf_hv=max(noisy_pf_hv), min_noisy_pf_hv=min(noisy_pf_hv),
    )
    columns["tied_members"] = any(e["tied_members"] for e in entries)
    return columns
