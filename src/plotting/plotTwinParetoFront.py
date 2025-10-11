import ast
import pandas as pd
from typing import List, Tuple, Optional, Union
import os
import matplotlib.pyplot as plt

import ast
import pandas as pd
from typing import List, Tuple, Union, Sequence, Optional

def _maybe_eval(x):
    if isinstance(x, str):
        x = x.strip()
        if x.startswith(("[", "(", "{")):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
    return x

def load_results_csv(path: str = "data/temp/results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "unique_sols","unique_fits","noisy_fits",
        "sol_iterations","sol_transitions",
        "pareto_solutions","pareto_fitnesses","pareto_true_fitnesses",
        "hypervolumes"
    ]:
        if col in df.columns:
            df[col] = df[col].map(_maybe_eval)
    return df

def get_pareto_front(
    df: pd.DataFrame,
    run_index: int = 0,
    generation: Union[int, str, Sequence[int]] = "last",
    use_true_fitness: bool = True,
    every_n: int = 1,
    include_indices: bool = False
) -> Union[List[Tuple], List[List[Tuple]], List[Tuple[int, List[Tuple]]]]:
    """
    Extract Pareto front(s) for a given run.

    Parameters
    ----------
    df : DataFrame
        Loaded via `load_results_csv`.
    run_index : int
        Row index selecting which run to use.
    generation : int | 'last' | 'all' | sequence[int]
        - int: return the front at that generation index
        - 'last': return the final generation's front
        - 'all': return all generations' fronts (optionally thinned by `every_n`)
        - sequence[int]: return those specific generation indices
    use_true_fitness : bool
        Prefer `pareto_true_fitnesses` when available; else use `pareto_fitnesses`.
    every_n : int
        When `generation='all'`, keep every Nth generation (default 1 = keep all).
    include_indices : bool
        If True and returning multiple fronts, return list of (gen_idx, front) pairs.

    Returns
    -------
    - Single front: List[Tuple]
    - Multiple fronts: List[List[Tuple]] or List[Tuple[int, List[Tuple]]]
    """
    row = df.iloc[run_index]
    key_primary = "pareto_true_fitnesses" if use_true_fitness and "pareto_true_fitnesses" in df.columns else "pareto_fitnesses"
    fronts_over_time = row[key_primary]

    if fronts_over_time is None:
        raise ValueError(f"No data in column '{key_primary}' for run_index={run_index}.")
    if not isinstance(fronts_over_time, (list, tuple)):
        raise TypeError(f"Expected list/tuple for {key_primary}, got {type(fronts_over_time)}.")
    if len(fronts_over_time) == 0:
        raise ValueError(f"{key_primary} list is empty for run_index={run_index}.")

    # Normalize each front to list[tuple]
    def _to_tuple_list(front):
        if front is None or len(front) == 0:
            return []
        if not all(isinstance(t, (list, tuple)) for t in front):
            raise TypeError("Pareto front is not a list of objective tuples.")
        return [tuple(t) for t in front]

    if generation == "last":
        return _to_tuple_list(fronts_over_time[-1])

    if isinstance(generation, int):
        if generation < 0 or generation >= len(fronts_over_time):
            raise IndexError(f"generation={generation} out of range [0, {len(fronts_over_time)-1}].")
        return _to_tuple_list(fronts_over_time[generation])

    if generation == "all":
        idxs = list(range(0, len(fronts_over_time), max(1, int(every_n))))
    elif isinstance(generation, (list, tuple)):
        idxs = list(generation)
        for g in idxs:
            if g < 0 or g >= len(fronts_over_time):
                raise IndexError(f"generation={g} out of range [0, {len(fronts_over_time)-1}].")
    else:
        raise TypeError("`generation` must be int, 'last', 'all', or a sequence of ints.")

    fronts = [ _to_tuple_list(fronts_over_time[g]) for g in idxs ]
    if include_indices:
        return list(zip(idxs, fronts))
    return fronts


Front = List[Tuple[float, float]]
MultiFront = List[Front]
IdxMultiFront = List[Tuple[int, Front]]


def plot_pareto_fronts(
    fronts: Union[Front, MultiFront, IdxMultiFront],
    run_index: int = 0,
    generation: Union[str, int] = "last",
    save_dir: str = "plots",
    filename: str = "pareto_front.png",
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    title_prefix: str = "Pareto Front",
    marker_size: int = 36,
    line: bool = False,
    colour: str = "red",
) -> str:
    """
    Plot a single Pareto front or multiple fronts with older generations faded.

    Parameters
    ----------
    fronts : 
        - Front: list[(f1, f2)] for a single generation
        - MultiFront: list[Front] for multiple generations (oldest→newest)
        - IdxMultiFront: list[(gen_idx, Front)]
    run_index : int
        Run identifier for title.
    generation : str|int
        Used in title for single fronts.
    save_dir : str
        Directory for saving.
    filename : str
        Output file name.
    xlabel, ylabel : str
        Axis labels.
    title_prefix : str
        Title prefix.
    marker_size : int
        Point size.
    line : bool
        Connect points within a front if True.
    colour : str
        Matplotlib colour for all generations.

    Returns
    -------
    str : Path of saved figure.
    """
    if not fronts:
        raise ValueError("No fronts provided.")

    # Normalize into [(gen_idx, front)] list
    if isinstance(fronts[0], tuple) and isinstance(fronts[0][0], (int, float)):
        # Single front
        idx_fronts: IdxMultiFront = [(generation if isinstance(generation, int) else 0, list(fronts))]  # type: ignore
        multi = False
    else:
        if isinstance(fronts[0], tuple) and isinstance(fronts[0][0], int):
            idx_fronts = [(gi, list(fr)) for gi, fr in fronts]  # type: ignore
        else:
            idx_fronts = list(enumerate(fronts))  # type: ignore
        multi = True

    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, filename)

    plt.figure(figsize=(7, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    if not multi:
        gi, fr = idx_fronts[0]
        xs, ys = zip(*fr)
        plt.scatter(xs, ys, s=marker_size, color=colour, label=f"Gen {generation}")
        if line and len(fr) > 1:
            plt.plot(xs, ys, color=colour, linewidth=1.2)
        plt.title(f"{title_prefix} (Run {run_index}, Gen {generation})")
        plt.legend()
    else:
        n = len(idx_fronts)
        # Alphas: 0.2 .. 1.0 across gens
        alphas = [0.2 + 0.8 * (i / max(1, n-1)) for i in range(n)]
        for i, (gi, fr) in enumerate(idx_fronts):
            xs, ys = zip(*fr)
            is_last = (i == n-1)
            plt.scatter(xs, ys, s=marker_size*(1.2 if is_last else 1.0),
                        color=colour, alpha=alphas[i],
                        label=f"Gen {gi}" if is_last else None)
            if line and len(fr) > 1:
                plt.plot(xs, ys, color=colour, alpha=alphas[i], linewidth=1.0)
        plt.title(f"{title_prefix} (Run {run_index}, Gens {idx_fronts[0][0]}→{idx_fronts[-1][0]})")
        plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return outpath

# df = load_results_csv("data/temp/results.csv")
# fronts_all = get_pareto_front(df, run_index=0, generation="all", use_true_fitness=False, every_n=1)
# plot_pareto_fronts(fronts_all, run_index=0, generation="all",
#                    save_dir="plots", filename="pareto_front_over_time.png",
#                    xlabel="Value", ylabel="Weight", line=False)

from typing import Dict, Iterable

def get_true_noisy_fronts_with_solutions(
    df: pd.DataFrame,
    run_index: int = 0,
    generation: Union[int, str, Sequence[int]] = "last",
    every_n: int = 1,
    include_indices: bool = False,
) -> Union[
    Tuple[Front, Front, List[Tuple]],                    # single gen: (true_front, noisy_front, solutions)
    List[Tuple[Front, Front, List[Tuple]]],              # multi gen: [(true, noisy, sols), ...]
    List[Tuple[int, Front, Front, List[Tuple]]]          # multi gen with indices
]:
    """
    Returns true/noisy fronts *and* the corresponding pareto_solutions so we can align
    the points and draw connecting lines. Each generation item matches by position in
    pareto_solutions[g] -> (pareto_true_fitnesses[g][i], pareto_fitnesses[g][i]).
    """
    row = df.iloc[run_index]
    true_all = row.get("pareto_true_fitnesses", None)
    noisy_all = row.get("pareto_fitnesses", None)
    sols_all = row.get("pareto_solutions", None)

    if true_all is None or noisy_all is None or sols_all is None:
        raise ValueError("Expected columns 'pareto_true_fitnesses', 'pareto_fitnesses', and 'pareto_solutions'.")

    # Helper to normalize a single generation item
    def _norm(front):
        if front is None:
            return []
        return [tuple(t) for t in front]

    # Resolve which generations to return (reuse your logic)
    if generation == "last":
        idxs = [len(sols_all) - 1]
    elif isinstance(generation, int):
        if generation < 0 or generation >= len(sols_all):
            raise IndexError(f"generation={generation} out of range [0, {len(sols_all)-1}].")
        idxs = [generation]
    elif generation == "all":
        idxs = list(range(0, len(sols_all), max(1, int(every_n))))
    elif isinstance(generation, (list, tuple, range)):
        idxs = list(generation)
        for g in idxs:
            if g < 0 or g >= len(sols_all):
                raise IndexError(f"generation={g} out of range [0, {len(sols_all)-1}].")
    else:
        raise TypeError("`generation` must be int, 'last', 'all', or a sequence of ints.")

    packs = []
    for g in idxs:
        sols_g  = _norm(sols_all[g])
        true_g  = _norm(true_all[g])
        noisy_g = _norm(noisy_all[g])

        # Defensive: if lengths disagree, try to truncate to the shortest to keep alignment.
        L = min(len(sols_g), len(true_g), len(noisy_g))
        sols_g, true_g, noisy_g = sols_g[:L], true_g[:L], noisy_g[:L]

        packs.append((true_g, noisy_g, sols_g))

    if len(packs) == 1 and not include_indices:
        return packs[0]
    if include_indices:
        return [(idxs[i],) + packs[i] for i in range(len(packs))]
    return packs


def plot_true_vs_noisy_fronts(
    data: Union[
        Tuple[Front, Front, List[Tuple]],
        List[Tuple[Front, Front, List[Tuple]]],
        List[Tuple[int, Front, Front, List[Tuple]]],
    ],
    run_index: int = 0,
    generation: Union[str, int] = "last",
    save_dir: str = "plots",
    filename: str = "pareto_true_vs_noisy.png",
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    title_prefix: str = "True vs Noisy Pareto Front",
    marker_size: int = 36,
    colour: str = "red",
) -> str:
    """
    Overlays TRUE (circles) and NOISY (triangles) fronts and draws a line
    connecting each (true_i -> noisy_i) for the same Pareto solution.
    Supports single or multiple generations (older gens faded).
    """
    # Normalize into a list of (maybe_gen_idx, true, noisy, sols)
    if isinstance(data, tuple) and len(data) == 3 and isinstance(data[0], list):
        items = [(None, data[0], data[1], data[2])]
        multi = False
    else:
        # Either [(true,noisy,sols), ...] or [(gen_idx,true,noisy,sols), ...]
        first = data[0]
        if len(first) == 3:
            items = [(None, *x) for x in data]  # type: ignore
        elif len(first) == 4:
            items = list(data)  # type: ignore
        else:
            raise ValueError("Unrecognized data format for plot_true_vs_noisy_fronts.")
        multi = len(items) > 1

    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, filename)

    plt.figure(figsize=(7, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    n = len(items)
    alphas = [0.2 + 0.8 * (i / max(1, n - 1)) for i in range(n)]

    # Legend flags so we only label once
    did_true_label = False
    did_noisy_label = False

    for i, item in enumerate(items):
        gi, true_fr, noisy_fr, sols = item
        alpha = alphas[i] if multi else 1.0
        is_last = (i == n - 1)

        if len(true_fr) and len(noisy_fr):
            # scatter TRUE (circles)
            tx, ty = zip(*true_fr)
            plt.scatter(tx, ty, s=marker_size * (1.2 if is_last else 1.0),
                        marker="o", facecolors="none", edgecolors=colour, alpha=alpha,
                        label=("True" if not did_true_label else None))
            did_true_label = True

            # scatter NOISY (triangles)
            nx_, ny_ = zip(*noisy_fr)
            plt.scatter(nx_, ny_, s=marker_size * (1.2 if is_last else 1.0),
                        marker="^", color=colour, alpha=alpha,
                        label=("Noisy" if not did_noisy_label else None))
            did_noisy_label = True

            # lines connecting corresponding points
            for (x1, y1), (x2, y2) in zip(true_fr, noisy_fr):
                plt.plot([x1, x2], [y1, y2], color=colour, alpha=alpha, linewidth=0.8)

        # If one is empty, still plot the available one
        elif len(true_fr):
            tx, ty = zip(*true_fr)
            plt.scatter(tx, ty, s=marker_size, marker="o", facecolors="none",
                        edgecolors=colour, alpha=alpha,
                        label=("True" if not did_true_label else None))
            did_true_label = True
        elif len(noisy_fr):
            nx_, ny_ = zip(*noisy_fr)
            plt.scatter(nx_, ny_, s=marker_size, marker="^", color=colour, alpha=alpha,
                        label=("Noisy" if not did_noisy_label else None))
            did_noisy_label = True

    if not multi:
        plt.title(f"{title_prefix} (Run {run_index}, Gen {generation})")
    else:
        first_idx = items[0][0] if items[0][0] is not None else 0
        last_idx = items[-1][0] if items[-1][0] is not None else len(items) - 1
        plt.title(f"{title_prefix} (Run {run_index}, Gens {first_idx}→{last_idx})")

    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return outpath

df = load_results_csv("data/temp/results.csv")
packs = get_true_noisy_fronts_with_solutions(
    df, run_index=0, generation="last", every_n=1, include_indices=True
)
plot_true_vs_noisy_fronts(
    packs,
    run_index=0,
    generation="last",
    save_dir="plots",
    filename="pareto_true_vs_noisy_over_time.png",
    xlabel="Value",
    ylabel="Weight",
    title_prefix="True vs Noisy Pareto Fronts Over Time",
    colour="red",
)