"""
Analysis plots for Pareto fronts.

This module provides visualizations for analyzing hypervolume, IGD, and
movement metrics over time or distance.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import MDS as MDS_sklearn, TSNE, Isomap

from ..base import (
    DEFAULT_TEMPLATE,
    PARETO_COLORSCALE,
    front_distance,
    get_series_info,
    create_empty_figure,
)


def plot_ind_vs_dist(frontdata, series_labels, distance_method='cumulative', nruns=1):
    """
    Plot hypervolume vs front distance using various distance methods.

    Shows how hypervolume changes as the Pareto front moves in decision space.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        distance_method: Method for computing x-axis distance
            - 'cumulative': Cumulative distance between successive fronts
            - 'mds': MDS 1D embedding of pairwise distances
            - 'tsne': t-SNE 1D embedding
            - 'isomap': Isomap 1D embedding
        nruns: Number of runs to include

    Returns:
        go.Figure: Plotly figure with hypervolume vs distance
    """
    fig = go.Figure()

    solution_set = "algo_front_solutions"
    metric = "algo_front_noisy_hypervolume"
    num_runs = nruns

    if not frontdata:
        fig.update_layout(title="No multi-objective data")
        return fig

    group_idx = 0
    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    max_run = min(num_runs, len(runs_full))
    run_indices = range(0, max_run)

    all_gens = []
    run_data = []

    # First pass: collect data for each run
    for run_idx in run_indices:
        gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
        if not gen_entries:
            continue

        gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))

        fronts = []
        hvs = []
        gens = []
        for entry in gen_entries:
            fronts.append(entry.get(solution_set) or [])
            hvs.append(float(entry.get(metric, 0.0)))
            gens.append(entry.get("gen_idx", 0))

        K = len(fronts)
        if K == 0:
            continue

        all_gens.extend(gens)
        run_data.append({
            "run_idx": run_idx,
            "fronts": fronts,
            "hvs": hvs,
            "gens": gens,
            "xs": None,
        })

    if not run_data:
        fig.update_layout(title=f"No usable runs for {series_name}")
        return fig

    # Compute x-coordinates based on method
    if distance_method == "cumulative":
        for rd in run_data:
            fronts = rd["fronts"]
            xs = [0.0]
            cum_dist = 0.0
            prev_front = fronts[0]

            for cur_front in fronts[1:]:
                d = front_distance(prev_front, cur_front)
                cum_dist += d
                xs.append(cum_dist)
                prev_front = cur_front

            rd["xs"] = xs

        xaxis_title = "Cumulative distance between successive fronts\n(avg min Hamming)"
        plot_title = f"Hypervolume vs cumulative front distance - {series_name}"

    elif distance_method == "mds":
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        mds = MDS_sklearn(
            n_components=1,
            dissimilarity='precomputed',
            n_init=4,
            max_iter=1000,
            random_state=42,
        )
        X_coords = mds.fit_transform(D)
        xs_raw = X_coords[:, 0]
        xs_global = xs_raw - xs_raw.min()

        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "MDS 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title = f"Hypervolume vs MDS front distance - {series_name}"

    elif distance_method == "tsne":
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        if N == 1:
            xs_global = np.array([0.0], dtype=float)
        elif N == 2:
            xs_global = np.array([0.0, 1.0], dtype=float)
        else:
            perplexity = min(30.0, max(5.0, (N - 1) / 3.0))
            if perplexity >= N:
                perplexity = N - 1.0

            tsne = TSNE(
                n_components=1,
                metric="precomputed",
                perplexity=perplexity,
                max_iter=1000,
                random_state=42,
                init="random",
            )
            X_coords = tsne.fit_transform(D)
            xs_raw = X_coords[:, 0]
            xs_global = xs_raw - xs_raw.min()

        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "t-SNE 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title = f"Hypervolume vs t-SNE front distance - {series_name}"

    elif distance_method == "isomap":
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        if N == 1:
            xs_global = np.array([0.0], dtype=float)
        elif N == 2:
            xs_global = np.array([0.0, 1.0], dtype=float)
        else:
            n_neighbors = min(10, N - 1)
            if n_neighbors < 2:
                n_neighbors = 2

            iso = Isomap(
                n_neighbors=n_neighbors,
                n_components=1,
                metric="precomputed",
            )
            X_coords = iso.fit_transform(D)
            xs_raw = X_coords[:, 0]
            xs_global = xs_raw - xs_raw.min()

        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "Isomap 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title = f"Hypervolume vs Isomap front distance - {series_name}"

    else:
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = list(range(K))

        xaxis_title = "Index (fallback)"
        plot_title = f"Hypervolume vs index (fallback) - {series_name}"

    # Plot all runs
    gmin, gmax = min(all_gens), max(all_gens)

    for k, rd in enumerate(run_data):
        show_cb = (k == 0)
        fig.add_trace(
            go.Scatter(
                x=rd["xs"],
                y=rd["hvs"],
                mode="lines+markers",
                name=f"{series_name}, Run {rd['run_idx']}",
                marker=dict(
                    size=8,
                    color=rd["gens"],
                    colorscale=PARETO_COLORSCALE,
                    colorbar=dict(title="Generation") if show_cb else None,
                    showscale=show_cb,
                    cmin=gmin,
                    cmax=gmax,
                ),
            )
        )

    fig.update_layout(
        title=plot_title,
        xaxis_title=xaxis_title,
        yaxis_title=f"Hypervolume ({metric})",
        template=DEFAULT_TEMPLATE,
    )

    return fig


def plot_igd_vs_dist(frontdata, series_labels, distance_method='cumulative', nruns=1):
    """
    Plot IGD (Inverted Generational Distance) vs front distance.

    Shows how IGD changes as the Pareto front moves in decision space.
    IGD is computed relative to the final generation's clean front.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        distance_method: Method for computing x-axis distance
            - 'cumulative': Cumulative distance between successive fronts
            - 'raw': Raw per-step distance to previous
            - 'mds': MDS 1D embedding
            - 'tsne': t-SNE 1D embedding
            - 'isomap': Isomap 1D embedding
        nruns: Number of runs to include

    Returns:
        go.Figure: Plotly figure with IGD vs distance
    """
    fig = go.Figure()

    solution_set = "algo_front_solutions"
    approx_fits_key = "algo_front_noisy_fitnesses"
    ref_fits_key = "clean_front_fitnesses"
    num_runs = nruns

    def igd(approx_front, ref_front):
        """IGD(P, R) = (1/|R|) * sum_{r in R} min_{p in P} d(r, p)"""
        if not ref_front or not approx_front:
            return np.nan

        A = [np.asarray(p, dtype=float) for p in approx_front]
        R = [np.asarray(r, dtype=float) for r in ref_front]

        mins = []
        for r in R:
            mins.append(min(float(np.linalg.norm(r - p)) for p in A))
        return float(np.mean(mins)) if mins else np.nan

    if not frontdata:
        fig.update_layout(title="No multi-objective data")
        return fig

    group_idx = 0
    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    max_run = min(num_runs, len(runs_full))
    run_indices = range(0, max_run)

    all_gens = []
    run_data = []

    # First pass: collect data for each run
    for run_idx in run_indices:
        gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
        if not gen_entries:
            continue

        gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))
        last_entry = gen_entries[-1] if gen_entries else {}

        ref_front = last_entry.get(ref_fits_key) or []

        fronts = []
        igds = []
        gens = []

        for entry in gen_entries:
            fronts.append(entry.get(solution_set) or [])
            approx_front = entry.get(approx_fits_key) or []
            igds.append(igd(approx_front, ref_front))
            gens.append(entry.get("gen_idx", 0))

        K = len(fronts)
        if K == 0:
            continue

        all_gens.extend(gens)
        run_data.append({
            "run_idx": run_idx,
            "fronts": fronts,
            "igds": igds,
            "gens": gens,
            "xs": None,
        })

    if not run_data:
        fig.update_layout(title=f"No usable runs for {series_name}")
        return fig

    # Compute x-coordinates based on method
    if distance_method == "cumulative":
        for rd in run_data:
            fronts = rd["fronts"]
            xs = [0.0]
            cum_dist = 0.0
            prev_front = fronts[0]

            for cur_front in fronts[1:]:
                d = front_distance(prev_front, cur_front)
                cum_dist += float(d)
                xs.append(cum_dist)
                prev_front = cur_front

            rd["xs"] = xs

        xaxis_title = "Cumulative distance between successive fronts\n(avg min Hamming)"
        plot_title = f"IGD vs cumulative front distance - {series_name}"

    elif distance_method == "raw":
        for rd in run_data:
            fronts = rd["fronts"]
            xs = []
            if not fronts:
                rd["xs"] = xs
                continue

            xs.append(0.0)
            prev_front = fronts[0]
            for cur_front in fronts[1:]:
                d = front_distance(prev_front, cur_front)
                xs.append(float(d))
                prev_front = cur_front

            rd["xs"] = xs

        xaxis_title = "Raw distance to previous front\n(avg min Hamming)"
        plot_title = f"IGD vs raw front distance - {series_name}"

    elif distance_method == "mds":
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        mds = MDS_sklearn(
            n_components=1,
            dissimilarity='precomputed',
            n_init=4,
            max_iter=1000,
            random_state=42,
        )
        X_coords = mds.fit_transform(D)
        xs_raw = X_coords[:, 0]
        xs_global = xs_raw - xs_raw.min()

        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "MDS 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title = f"IGD vs MDS front distance - {series_name}"

    elif distance_method == "tsne":
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        if N == 1:
            xs_global = np.array([0.0], dtype=float)
        elif N == 2:
            xs_global = np.array([0.0, 1.0], dtype=float)
        else:
            perplexity = min(30.0, max(5.0, (N - 1) / 3.0))
            if perplexity >= N:
                perplexity = N - 1.0

            tsne = TSNE(
                n_components=1,
                metric="precomputed",
                perplexity=perplexity,
                max_iter=1000,
                random_state=42,
                init="random",
            )
            X_coords = tsne.fit_transform(D)
            xs_raw = X_coords[:, 0]
            xs_global = xs_raw - xs_raw.min()

        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "t-SNE 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title = f"IGD vs t-SNE front distance - {series_name}"

    elif distance_method == "isomap":
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        if N == 1:
            xs_global = np.array([0.0], dtype=float)
        elif N == 2:
            xs_global = np.array([0.0, 1.0], dtype=float)
        else:
            n_neighbors = min(10, N - 1)
            if n_neighbors < 2:
                n_neighbors = 2

            iso = Isomap(
                n_neighbors=n_neighbors,
                n_components=1,
                metric="precomputed",
            )
            X_coords = iso.fit_transform(D)
            xs_raw = X_coords[:, 0]
            xs_global = xs_raw - xs_raw.min()

        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "Isomap 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title = f"IGD vs Isomap front distance - {series_name}"

    else:
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = list(range(K))

        xaxis_title = "Index (fallback)"
        plot_title = f"IGD vs index (fallback) - {series_name}"

    # Plot all runs
    gmin, gmax = min(all_gens), max(all_gens)

    for k, rd in enumerate(run_data):
        show_cb = (k == 0)
        fig.add_trace(
            go.Scatter(
                x=rd["xs"],
                y=rd["igds"],
                mode="lines+markers",
                name=f"{series_name}, Run {rd['run_idx']}",
                marker=dict(
                    size=8,
                    color=rd["gens"],
                    colorscale=PARETO_COLORSCALE,
                    colorbar=dict(title="Generation") if show_cb else None,
                    showscale=show_cb,
                    cmin=gmin,
                    cmax=gmax,
                ),
            )
        )

    fig.update_layout(
        title=plot_title,
        xaxis_title=xaxis_title,
        yaxis_title="IGD (from ref front)",
        template=DEFAULT_TEMPLATE,
    )

    return fig


def plot_progress_per_movement(
    frontdata,
    series_labels,
    group_idx=0,
    run_idx=0,
    solution_key="algo_front_solutions",
    hv_key="algo_front_clean_hypervolume",
    eps=1e-12,
    k_patience=10,
    use_ratio=True,
    show_deltas=True,
):
    """
    Plot progress per movement ratio across generations.

    Shows the ratio of objective improvement (delta HV) to decision space
    movement (front distance). Useful for identifying when optimization
    is making efficient progress.

    Args:
        frontdata: Data for the Pareto fronts
        series_labels: Labels for each series
        group_idx: Index of the group to use
        run_idx: Index of the run to use
        solution_key: Key for solution data
        hv_key: Key for hypervolume data
        eps: Small value to prevent division by zero
        k_patience: Window size for stop detection
        use_ratio: Whether to include ratio in stop condition
        show_deltas: Whether to show delta traces

    Returns:
        go.Figure: Plotly figure with progress per movement ratio
    """
    fig = go.Figure()

    if not frontdata:
        fig.update_layout(title="No MO_data_PPP provided")
        return fig

    runs_full, series_name = get_series_info(frontdata, series_labels, group_idx)

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    if len(runs_full) <= run_idx:
        fig.update_layout(title=f"Run {run_idx} not available for {series_name}")
        return fig

    gen_entries = runs_full[run_idx]
    if not gen_entries or len(gen_entries) < 2:
        fig.update_layout(title=f"Not enough generations for {series_name} (run {run_idx})")
        return fig

    # Compute per-gen series
    gens = []
    delta_D = []
    delta_O = []
    rho = []

    sols = [e.get(solution_key) or [] for e in gen_entries]
    hvs = [e.get(hv_key, None) for e in gen_entries]
    gen_ids = [e.get("gen_idx", i) for i, e in enumerate(gen_entries)]

    for t in range(1, len(gen_entries)):
        prev_s, cur_s = sols[t-1], sols[t]
        dD = float(front_distance(prev_s, cur_s))

        prev_hv = hvs[t-1]
        cur_hv = hvs[t]
        if prev_hv is None or cur_hv is None:
            dO = 0.0
        else:
            dO = float(cur_hv - prev_hv)

        gens.append(int(gen_ids[t]))
        delta_D.append(dD)
        delta_O.append(dO)
        rho.append(dO / (dD + eps))

    gens = np.asarray(gens)
    delta_D = np.asarray(delta_D, dtype=float)
    delta_O = np.asarray(delta_O, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # Find stop point
    tauD = float(np.percentile(delta_D, 10))
    posO = delta_O[delta_O > 0]
    tauO = float(np.percentile(posO, 10)) if len(posO) else 0.0
    tauR = float(np.percentile(rho, 90)) if len(rho) else 0.0

    stop_gen = None
    if len(gens) >= k_patience:
        for i in range(k_patience - 1, len(gens)):
            w = slice(i - k_patience + 1, i + 1)
            cond_D = np.all(delta_D[w] < tauD)
            cond_O = np.all(delta_O[w] < tauO)
            if use_ratio:
                cond_R = np.all(rho[w] > tauR)
                if cond_D and cond_O and cond_R:
                    stop_gen = int(gens[i])
                    break
            else:
                if cond_D and cond_O:
                    stop_gen = int(gens[i])
                    break

    # Plot
    fig.add_trace(go.Scatter(
        x=gens, y=rho, mode="lines+markers",
        name="dO/dD"
    ))

    if show_deltas:
        fig.add_trace(go.Scatter(
            x=gens, y=delta_D, mode="lines",
            name="dD (decision change)", yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=gens, y=delta_O, mode="lines",
            name="dO (HV improvement)", yaxis="y3"
        ))

    fig.update_layout(
        title=f"Progress movement ratio - {series_name} (run {run_idx})",
        xaxis=dict(title="Generation"),
        yaxis=dict(title="dO/dD"),
        yaxis2=dict(
            title="dD", overlaying="y", side="right",
            showgrid=False
        ),
        yaxis3=dict(
            title="dO", overlaying="y", side="right",
            anchor="free", position=1.0,
            showgrid=False
        ),
        legend=dict(orientation="h"),
    )

    if stop_gen is not None:
        fig.add_vline(
            x=stop_gen,
            line_dash="dash",
            annotation_text=f"stop @ {stop_gen}",
            annotation_position="top left"
        )

    return fig
