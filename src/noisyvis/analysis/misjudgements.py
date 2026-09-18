"""Misjudgement analysis over a run's representative trajectory.

Pure functions of the per-visit true and noisy fitness lists: the steps where the true fitness moved
the wrong way, where the noise magnitude grew, where the noisy signal favoured a worse solution, and
where the solution violated a constraint. The dashboard tables count them per run
(`noisyvis.dashboard.tables`); the trajectory plot marks the individual steps.
"""

import numpy as np


def _compute_n_misjudgements(row) -> int:
    """
    Count misjudgements in a run: steps where the representative solution's
    true fitness moved in the wrong direction (worse than the previous step).

    For maximisation: fits[i+1] < fits[i] is a misjudgement.
    For minimisation: fits[i+1] > fits[i] is a misjudgement.
    """
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    # MO rows and old SO rows (pre-Logger rename) have rep_fits=NaN, not a list.
    # NaN is truthy so `not rep_fits` doesn't catch it — guard with isinstance.
    # TODO: compute MO equivalent from noisy_pf_true_hypervolumes (HV decreased steps)
    if not isinstance(rep_fits, (list, np.ndarray)) or len(rep_fits) < 2:
        return 0
    problem_goal = row.get('problem_goal', 'maximise') if hasattr(row, 'get') else row['problem_goal']
    minimising = str(problem_goal)[:3].lower() == 'min'
    if minimising:
        return sum(1 for i in range(len(rep_fits) - 1) if rep_fits[i + 1] > rep_fits[i])
    return sum(1 for i in range(len(rep_fits) - 1) if rep_fits[i + 1] < rep_fits[i])


def increasing_noise_step_indices(rep_fits, rep_noisy_fits) -> list:
    """
    Return visit indices i+1 where the noise magnitude |true_fit - noisy_fit|
    grew relative to the previous step's noise magnitude.
    """
    if not isinstance(rep_fits, (list, np.ndarray)) or not isinstance(rep_noisy_fits, (list, np.ndarray)):
        return []
    if len(rep_fits) != len(rep_noisy_fits) or len(rep_fits) < 2:
        return []
    diffs = [abs(t - n) for t, n in zip(rep_fits, rep_noisy_fits)]
    return [i + 1 for i in range(len(diffs) - 1) if diffs[i + 1] > diffs[i]]


def comparison_misjudgement_step_indices(rep_fits, rep_noisy_fits, minimising) -> list:
    """
    Return visit indices i+1 where the true fitness moved in the wrong direction
    (worse than the previous step) while the noisy fitness moved in the right
    direction (better than the previous step) — i.e. the noisy signal would have
    favoured adopting a solution that was actually worse.

    For maximisation: true_fit[i+1] < true_fit[i] AND noisy_fit[i+1] > noisy_fit[i].
    For minimisation: true_fit[i+1] > true_fit[i] AND noisy_fit[i+1] < noisy_fit[i].
    """
    if not isinstance(rep_fits, (list, np.ndarray)) or not isinstance(rep_noisy_fits, (list, np.ndarray)):
        return []
    if len(rep_fits) != len(rep_noisy_fits) or len(rep_fits) < 2:
        return []
    indices = []
    for i in range(len(rep_fits) - 1):
        if minimising:
            true_worse = rep_fits[i + 1] > rep_fits[i]
            noisy_better = rep_noisy_fits[i + 1] < rep_noisy_fits[i]
        else:
            true_worse = rep_fits[i + 1] < rep_fits[i]
            noisy_better = rep_noisy_fits[i + 1] > rep_noisy_fits[i]
        if true_worse and noisy_better:
            indices.append(i + 1)
    return indices


def constraint_misjudgement_step_indices(rep_fits) -> list:
    """Return visit indices i where the true fitness is negative (a constraint violation)."""
    if not isinstance(rep_fits, (list, np.ndarray)):
        return []
    return [i for i, fit in enumerate(rep_fits) if fit < 0]


def _compute_n_increasing_noise(row) -> int:
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    rep_noisy_fits = row.get('rep_noisy_fits') if hasattr(row, 'get') else row['rep_noisy_fits']
    return len(increasing_noise_step_indices(rep_fits, rep_noisy_fits))


def _compute_n_comparison_misjudgements(row) -> int:
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    rep_noisy_fits = row.get('rep_noisy_fits') if hasattr(row, 'get') else row['rep_noisy_fits']
    problem_goal = row.get('problem_goal', 'maximise') if hasattr(row, 'get') else row['problem_goal']
    minimising = str(problem_goal)[:3].lower() == 'min'
    return len(comparison_misjudgement_step_indices(rep_fits, rep_noisy_fits, minimising))


def _compute_n_constraint_misjudgements(row) -> int:
    rep_fits = row.get('rep_fits') if hasattr(row, 'get') else row['rep_fits']
    return len(constraint_misjudgement_step_indices(rep_fits))
