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

df = load_results_csv("data/temp/results.csv")
fronts_all = get_pareto_front(df, run_index=0, generation="all", use_true_fitness=False, every_n=1)
plot_pareto_fronts(fronts_all, run_index=0, generation="all",
                   save_dir="plots", filename="pareto_front_over_time.png",
                   xlabel="Value", ylabel="Weight", line=False)