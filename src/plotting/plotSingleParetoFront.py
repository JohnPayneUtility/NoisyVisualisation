import ast
import pandas as pd
from typing import List, Tuple, Optional, Union
import os
import matplotlib.pyplot as plt

# --- Helpers ---
def _maybe_eval(x):
    """Safely parse Python-literal-like strings (e.g., lists/tuples/dicts) from CSV."""
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
    # Columns that may contain list/tuple data stored as strings
    candidate_list_cols = [
        "unique_sols", "unique_fits", "noisy_fits",
        "sol_iterations", "sol_transitions",
        "pareto_solutions", "pareto_fitnesses", "pareto_true_fitnesses",
        "hypervolumes"
    ]
    for col in candidate_list_cols:
        if col in df.columns:
            df[col] = df[col].map(_maybe_eval)
    return df

def get_pareto_front(
    df: pd.DataFrame,
    run_index: int = 0,
    generation: Union[int, str] = "last",
    use_true_fitness: bool = True
) -> List[Tuple]:
    """
    Extract a single Pareto front (list of objective tuples) from one run.
    
    Parameters
    ----------
    df : DataFrame loaded by `load_results_csv`.
    run_index : which row/run to take the front from.
    generation : integer generation or "last" for the final generation.
    use_true_fitness : if True, prefer `pareto_true_fitnesses`, else use `pareto_fitnesses`.
    
    Returns
    -------
    List[Tuple] : list of objective tuples for the selected generation.
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

    # Each element is the Pareto front at a generation: a list of objective tuples
    if generation == "last":
        front = fronts_over_time[-1]
    else:
        if not isinstance(generation, int):
            raise TypeError("`generation` must be int or 'last'.")
        if generation < 0 or generation >= len(fronts_over_time):
            raise IndexError(f"generation={generation} out of range [0, {len(fronts_over_time)-1}].")
        front = fronts_over_time[generation]

    if front is None or len(front) == 0:
        raise ValueError("Selected generation has an empty Pareto front.")
    if not all(isinstance(t, (list, tuple)) for t in front):
        raise TypeError("Pareto front is not a list of objective tuples.")
    return [tuple(obj) for obj in front]

def plot_pareto_front(
    front,
    run_index: int = 0,
    generation: str = "last",
    save_dir: str = "plots",
    filename: str = "pareto_front.png"
):
    """
    Plot a 2D Pareto front and save to file.
    
    Parameters
    ----------
    front : list of (obj1, obj2) tuples
        The Pareto front data to plot.
    run_index : int
        Which run the data came from (for the title).
    generation : str or int
        Which generation (for the title).
    save_dir : str
        Directory where the plot will be saved.
    filename : str
        Name of the output PNG file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Assume front is list of 2D tuples
    x = [pt[0] for pt in front]
    y = [pt[1] for pt in front]

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, c="red", marker="o", label="Pareto solutions")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"Pareto Front (Run {run_index}, Gen {generation})")
    plt.legend()
    plt.grid(True)

    outpath = os.path.join(save_dir, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Pareto front saved to {outpath}")

# --- Example usage ---
if __name__ == "__main__":
    df = load_results_csv("data/temp/results.csv")
    # Grab the last-generation Pareto front from the first run,
    # preferring true (noise-free) fitnesses if available.
    front = get_pareto_front(df, run_index=0, generation="last", use_true_fitness=True)
    print(f"Selected Pareto front has {len(front)} points. First 5:\n{front[:5]}")
    plot_pareto_front(front, run_index=0, generation="last")