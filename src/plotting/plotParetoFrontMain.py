import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
import numpy as np
from sklearn.manifold import MDS as MDS_sklearn, TSNE, Isomap

# import plotly.io as pio
# import imageio.v2 as imageio
# from io import BytesIO

# ---- Distance helpers ----
def canonical_solution(sol):
    # Force every element to be an int
    return tuple(int(x) for x in sol)

def avg_min_hamming_A_to_B(frontA, frontB):
    """Asymmetric: average over a∈A of min_b Hamming(a,b); normalised by length."""
    A = [ canonical_solution(a) for a in frontA ]
    B = [ canonical_solution(b) for b in frontB ]
    L = len(A[0]) if A else 1
    norm = float(L) if L > 0 else 1.0
    total = 0.0
    for a in A:
        # min Hamming(a, b) over b ∈ B
        m = min(sum(aa != bb for aa, bb in zip(a, b)) for b in B) / norm
        total += m
    return total / len(A)

def front_distance(frontA, frontB):
    """Symmetric average-min Hamming distance between two fronts."""
    d1 = avg_min_hamming_A_to_B(frontA, frontB)
    d2 = avg_min_hamming_A_to_B(frontB, frontA)
    return 0.5*(d1 + d2)


def plotParetoFront(frontdata, series_labels):
    """
    Function to generate pareto front plots based on various settings.
    """
    fig = go.Figure()

    if not frontdata:
        fig.update_layout(title="No multi-objective data")
        return fig

    # Pick first series and first run for a simple baseline
    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = str(series_labels[group_idx]) if series_labels and len(series_labels) > group_idx else f"Series {group_idx}"

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    run_idx = 0
    gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []

    for entry in gen_entries:
        pts = entry.get('algo_front_noisy_fitnesses') or []
        g   = entry.get('gen_idx', None)
        if not pts:
            continue

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                name=f"G{g}"
            )
        )

    fig.update_layout(
        title=f"True Fitness of Noisy PF — {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template="plotly_white",
        legend_title="Generation"
    )
    return fig

def plotParetoFrontSubplots(frontdata, series_labels):
    """
    Plot each generation's Pareto front in its own subplot.
    """
    if not frontdata:
        fig = go.Figure()
        fig.update_layout(title="No multi-objective data")
        return fig

    # Pick first series and first run
    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig = go.Figure()
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    run_idx = 2
    gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []

    if not gen_entries:
        fig = go.Figure()
        fig.update_layout(title=f"No generations for {series_name}, Run {run_idx}")
        return fig

    # --- Determine how many subplots we need ---
    # Assume one entry per generation
    n_gens = len(gen_entries)

    # Simple auto-grid: as square as possible
    cols = math.ceil(math.sqrt(n_gens))
    rows = math.ceil(n_gens / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"G{entry.get('gen_idx', i)}" for i, entry in enumerate(gen_entries)],
        shared_xaxes=True,
        shared_yaxes=True
    )

    # --- Add each generation's front to its own subplot ---
    for i, entry in enumerate(gen_entries):
        pts = entry.get('algo_front_noisy_fitnesses') or []
        g   = entry.get('gen_idx', i)

        if not pts:
            continue

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                showlegend=False  # legend would be repetitive here
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title=f"True Fitness of Noisy PF — {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template="plotly_white",
        height=300 * rows,  # scale figure height with number of rows
    )

    # Optional: tighten subplot spacing a bit
    fig.update_layout(margin=dict(l=40, r=20, t=80, b=40))

    return fig

def plotParetoFrontSubplotsMulti(frontdata, series_labels, nruns=1):
    """
    """
    if not frontdata:
        fig = go.Figure()
        fig.update_layout(title="No multi-objective data")
        return fig

    # Pick first series
    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig = go.Figure()
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    # Decide which runs to plot
    max_runs_available = len(runs_full)
    num_runs = min(nruns, max_runs_available)
    run_indices = list(range(num_runs))

    # --- Determine how many generations (subplots) we need ---
    # Use the maximum number of generations across the selected runs
    n_gens = max(len(runs_full[r]) for r in run_indices)

    if n_gens == 0:
        fig = go.Figure()
        fig.update_layout(title=f"No generations for {series_name}")
        return fig

    # Simple auto-grid: as square as possible
    cols = math.ceil(math.sqrt(n_gens))
    rows = math.ceil(n_gens / cols)

    # Titles: use gen_idx from first run where available, else index
    subplot_titles = []
    for gi in range(n_gens):
        # Try to get a gen_idx from the first run that has this generation
        gen_label = gi
        for r in run_indices:
            if gi < len(runs_full[r]):
                gen_label = runs_full[r][gi].get("gen_idx", gi)
                break
        subplot_titles.append(f"G{gen_label}")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True
    )

    # Colour palette for runs
    colors = px.colors.qualitative.Plotly
    # --- Add each generation's fronts for each run to its subplot ---
    for gi in range(n_gens):
        row = gi // cols + 1
        col = gi % cols + 1

        for r_idx, run_id in enumerate(run_indices):
            run = runs_full[run_id]
            if gi >= len(run):
                continue  # this run has fewer generations

            entry = run[gi]
            pts = entry.get("algo_front_noisy_fitnesses") or []
            if not pts:
                continue

            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]

            color = colors[r_idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=f"Run {run_id}",
                    showlegend=(gi == 0),  # only show in first subplot
                    line=dict(color=color),
                    marker=dict(color=color)
                ),
                row=row,
                col=col
            )

    fig.update_layout(
        title=f"True Fitness of Noisy PF — {series_name}, first {num_runs} runs",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template="plotly_white",
        height=300 * rows,
        margin=dict(l=40, r=20, t=80, b=40),
    )

    return fig

def PlotparetoFrontSubplotsHighlightedOld(frontdata, series_labels):
    """
    Like plotParetoFrontSubplots, but:
      - Shows previous generation's solutions in grey in each subplot.
      - Highlights new solutions (not present in previous generation) with star markers.
    """
    if not frontdata:
        fig = go.Figure()
        fig.update_layout(title="No multi-objective data")
        return fig

    # Pick first series and first run
    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig = go.Figure()
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    run_idx = 0
    gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []

    if not gen_entries:
        fig = go.Figure()
        fig.update_layout(title=f"No generations for {series_name}, Run {run_idx}")
        return fig

    # --- Determine how many subplots we need ---
    n_gens = len(gen_entries)

    # Simple auto-grid: as square as possible
    cols = math.ceil(math.sqrt(n_gens))
    rows = math.ceil(n_gens / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"G{entry.get('gen_idx', i)}" for i, entry in enumerate(gen_entries)
        ],
        shared_xaxes=True,
        shared_yaxes=True
    )

    # --- Add each generation's front to its own subplot ---
    for i, entry in enumerate(gen_entries):
        pts = entry.get('algo_front_noisy_fitnesses') or []
        g   = entry.get('gen_idx', i)

        if not pts:
            continue

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        # Determine subplot position
        row = i // cols + 1
        col = i % cols + 1

        # --- Previous generation points (grey) ---
        if i > 0:
            prev_entry = gen_entries[i - 1]
            prev_pts = prev_entry.get('algo_front_noisy_fitnesses') or []

            if prev_pts:
                prev_xs = [float(p[0]) for p in prev_pts]
                prev_ys = [float(p[1]) for p in prev_pts]

                fig.add_trace(
                    go.Scatter(
                        x=prev_xs,
                        y=prev_ys,
                        mode="markers",
                        marker=dict(color="lightgrey"),
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

            # Build a set of previous points for quick membership checks
            prev_set = { (float(p[0]), float(p[1])) for p in prev_pts }
        else:
            prev_set = set()

        # --- Current generation points with highlighting ---
        # Mark each point as star if new, circle if seen before
        marker_symbols = []
        for x_val, y_val in zip(xs, ys):
            if (x_val, y_val) in prev_set:
                marker_symbols.append("circle")
            else:
                marker_symbols.append("star")

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                marker=dict(symbol=marker_symbols),
                showlegend=False  # legend would be repetitive here
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title=f"True Fitness of Noisy PF — {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template="plotly_white",
        height=300 * rows,  # scale figure height with number of rows
    )

    # Optional: tighten subplot spacing a bit
    fig.update_layout(margin=dict(l=40, r=20, t=80, b=40))

    return fig

def PlotparetoFrontSubplotsHighlighted(
    frontdata,
    series_labels,
    solutions_key="algo_front_solutions",
    fits_key="algo_front_noisy_fitnesses",
    dist_decimals=3,
):
    """
    Like plotParetoFrontSubplots, but:
      - Shows previous generation's solutions in grey in each subplot.
      - Highlights new solutions (not present in previous generation) with star markers.
      - Adds (d=...) to each subplot title where d is front_distance(cur, prev) in decision space.
    """
    if not frontdata:
        fig = go.Figure()
        fig.update_layout(title="No multi-objective data")
        return fig

    # Pick first series and first run
    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig = go.Figure()
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    run_idx = 0
    gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
    if not gen_entries:
        fig = go.Figure()
        fig.update_layout(title=f"No generations for {series_name}, Run {run_idx}")
        return fig

    # --- Build subplot titles including distance-to-previous ---
    subplot_titles = []
    for i, entry in enumerate(gen_entries):
        g = entry.get("gen_idx", i)

        if i == 0:
            subplot_titles.append(f"G{g} (d=—)")
            continue

        prev_entry = gen_entries[i - 1]
        cur_front_sols  = entry.get(solutions_key) or []
        prev_front_sols = prev_entry.get(solutions_key) or []

        if not cur_front_sols or not prev_front_sols:
            d_txt = "—"
        else:
            try:
                d = front_distance(cur_front_sols, prev_front_sols)
                d_txt = f"{d:.{dist_decimals}f}"
            except Exception:
                d_txt = "—"

        subplot_titles.append(f"G{g} (d={d_txt})")

    # --- Determine grid ---
    n_gens = len(gen_entries)
    cols = math.ceil(math.sqrt(n_gens))
    rows = math.ceil(n_gens / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # --- Add each generation's front to its own subplot ---
    for i, entry in enumerate(gen_entries):
        pts = entry.get(fits_key) or []
        if not pts:
            continue

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        row = i // cols + 1
        col = i % cols + 1

        # --- Previous generation points (grey) ---
        if i > 0:
            prev_entry = gen_entries[i - 1]
            prev_pts = prev_entry.get(fits_key) or []

            if prev_pts:
                prev_xs = [float(p[0]) for p in prev_pts]
                prev_ys = [float(p[1]) for p in prev_pts]

                fig.add_trace(
                    go.Scatter(
                        x=prev_xs,
                        y=prev_ys,
                        mode="markers",
                        marker=dict(color="lightgrey"),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            prev_set = {(float(p[0]), float(p[1])) for p in prev_pts}
        else:
            prev_set = set()

        # --- Current generation points with highlighting ---
        marker_symbols = []
        for x_val, y_val in zip(xs, ys):
            marker_symbols.append("circle" if (x_val, y_val) in prev_set else "star")

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                marker=dict(symbol=marker_symbols),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"True Fitness of Noisy PF — {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template="plotly_white",
        height=300 * rows,
        margin=dict(l=40, r=20, t=80, b=40),
    )

    return fig


def plotParetoFrontAnimation(frontdata, series_labels):
    """
    Make an animated plot where each frame shows one generation's Pareto front.
    """
    if not frontdata:
        fig = go.Figure()
        fig.update_layout(title="No multi-objective data")
        return fig

    # Pick first series and first run
    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig = go.Figure()
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    run_idx = 0
    gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []

    if not gen_entries:
        fig = go.Figure()
        fig.update_layout(title=f"No generations for {series_name}, Run {run_idx}")
        return fig

    # ---- Collect data & axis ranges across all generations ----
    all_x, all_y = [], []

    gens_data = []  # list of (gen_label, xs, ys)
    for i, entry in enumerate(gen_entries):
        pts = entry.get('algo_front_noisy_fitnesses') or []
        g   = entry.get('gen_idx', i)

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        gens_data.append((g, xs, ys))
        all_x.extend(xs)
        all_y.extend(ys)

    if not all_x or not all_y:
        fig = go.Figure()
        fig.update_layout(title=f"No points to plot for {series_name}, Run {run_idx}")
        return fig

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # ---- Initial frame: first generation with points ----
    # Find first generation that actually has data
    init_idx = next((idx for idx, (_, xs, ys) in enumerate(gens_data) if xs and ys), 0)
    init_gen, init_xs, init_ys = gens_data[init_idx]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=init_xs,
                y=init_ys,
                mode="lines+markers",
                name=f"G{init_gen}",
                showlegend=False,
            )
        ]
    )

    # ---- Build frames ----
    frames = []
    for g, xs, ys in gens_data:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        showlegend=False,
                    )
                ],
                name=str(g),
                layout=go.Layout(title_text=f"True Fitness of Noisy PF — {series_name}, Run {run_idx}, Gen {g}")
            )
        )

    fig.frames = frames

    # ---- Animation controls (Play/Pause + slider) ----
    steps = []
    for g, _, _ in gens_data:
        steps.append(
            dict(
                method="animate",
                label=f"G{g}",
                args=[
                    [str(g)],
                    dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=0)),
                ],
            )
        )

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Generation: "},
            pad={"t": 30},
            steps=steps,
        )
    ]

    fig.update_layout(
        title=f"True Fitness of Noisy PF — {series_name}, Run {run_idx}",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
        template="plotly_white",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.15,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        sliders=sliders,
        height=600,
    )
    return fig

# def saveParetoFrontGIF(frontdata, series_labels):
#     fig = plotParetoFrontAnimation(frontdata, series_labels)

#     images = []
#     for frame in fig.frames:
#         # Update figure to this frame's data
#         fig.update(data=frame.data)
#         # Render to PNG bytes
#         png_bytes = pio.to_image(fig, format="png", width=800, height=600)
#         images.append(imageio.imread(BytesIO(png_bytes)))
#     gif_path = "../../plots/pareto_front.gif"
#     imageio.mimsave(gif_path, images, duration=0.4)

# def plotParetoFrontNoisy(frontdata, series_labels):
#     """
#     Plot true (clean) fitness of noisy PF + noisy fitness as triangles.
#     Colours represent generation (continuous scale).
#     """
#     import numpy as np
#     fig = go.Figure()

#     if not frontdata:
#         fig.update_layout(title="No multi-objective data")
#         return fig

#     # Select first series/run as before
#     group_idx = 0
#     runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
#     series_name = str(series_labels[group_idx]) if series_labels and len(series_labels) > group_idx else f"Series {group_idx}"

#     if not runs_full:
#         fig.update_layout(title=f"No runs for {series_name}")
#         return fig

#     run_idx = 0
#     gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
#     if not gen_entries:
#         fig.update_layout(title=f"No generations for {series_name}")
#         return fig

#     # Extract generation range for normalisation
#     gens = [entry.get('gen_idx', 0) for entry in gen_entries]
#     gen_min, gen_max = min(gens), max(gens)

#     # Aggregate all points to one trace (better for colorbar)
#     clean_x, clean_y, clean_c = [], [], []
#     noisy_x, noisy_y, noisy_c = [], [], []

#     for entry in gen_entries:
#         g = entry.get('gen_idx', None)
#         clean_pts = entry.get('algo_front_clean_fitnesses') or []
#         noisy_pts = entry.get('algo_front_noisy_fitnesses') or []

#         for p in clean_pts:
#             clean_x.append(float(p[0]))
#             clean_y.append(float(p[1]))
#             clean_c.append(g)

#         for p in noisy_pts:
#             noisy_x.append(float(p[0]))
#             noisy_y.append(float(p[1]))
#             noisy_c.append(g)

#     # Clean points (circles)
#     fig.add_trace(
#         go.Scatter(
#             x=clean_x,
#             y=clean_y,
#             mode="markers",
#             marker=dict(
#                 symbol="circle",
#                 size=8,
#                 color=clean_c,
#                 colorscale="Viridis",
#                 colorbar=dict(title="Generation"),
#                 cmin=gen_min,
#                 cmax=gen_max
#             ),
#             name="Clean fitness"
#         )
#     )

#     # Noisy points (triangles)
#     fig.add_trace(
#         go.Scatter(
#             x=noisy_x,
#             y=noisy_y,
#             mode="markers",
#             marker=dict(
#                 symbol="triangle-up",
#                 size=8,
#                 color=noisy_c,
#                 colorscale="Viridis",
#                 showscale=False,
#                 cmin=gen_min,
#                 cmax=gen_max
#             ),
#             name="Noisy fitness"
#         )
#     )

#     fig.update_layout(
#         title=f"True vs Noisy Fitness of Noisy PF — {series_name}, Run {run_idx}",
#         xaxis_title="Objective 1",
#         yaxis_title="Objective 2",
#         template="plotly_white",
#         legend_title="Type"
#     )

#     return fig

# def plotParetoFrontNoisy(frontdata, series_labels):
#     import plotly.graph_objects as go

#     fig = go.Figure()
#     if not frontdata:
#         fig.update_layout(title="No multi-objective data")
#         return fig

#     group_idx, run_idx = 0, 0
#     runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
#     series_name = str(series_labels[group_idx]) if series_labels and len(series_labels) > group_idx else f"Series {group_idx}"
#     gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
#     if not gen_entries:
#         fig.update_layout(title=f"No generations for {series_name}")
#         return fig

#     gens = [e.get("gen_idx", 0) for e in gen_entries]
#     gen_min, gen_max = min(gens), max(gens)

#     clean_x, clean_y, clean_c = [], [], []
#     noisy_x, noisy_y, noisy_c = [], [], []

#     for entry in gen_entries:
#         g = entry.get("gen_idx", 0)
#         clean_pts = entry.get("algo_front_clean_fitnesses") or []
#         noisy_pts = entry.get("algo_front_noisy_fitnesses") or []

#         # Draw dotted connectors (colour by generation)
#         for c_fit, n_fit in zip(clean_pts, noisy_pts):
#             fig.add_trace(go.Scatter(
#                 x=[float(c_fit[0]), float(n_fit[0])],
#                 y=[float(c_fit[1]), float(n_fit[1])],
#                 mode="lines",
#                 line=dict(dash="dot", width=1),
#                 marker=dict(color=g, colorscale="agsunset", cmin=gen_min, cmax=gen_max),
#                 hoverinfo="skip",
#                 showlegend=False
#             ))

#             clean_x.append(float(c_fit[0])); clean_y.append(float(c_fit[1])); clean_c.append(g)
#             noisy_x.append(float(n_fit[0])); noisy_y.append(float(n_fit[1])); noisy_c.append(g)

#     # Clean points (circles) with colorbar
#     fig.add_trace(go.Scatter(
#         x=clean_x, y=clean_y, mode="markers", name="Clean fitness",
#         marker=dict(symbol="circle", size=8, color=clean_c,
#                     colorscale="agsunset", colorbar=dict(title="Generation"),
#                     cmin=gen_min, cmax=gen_max)
#     ))

#     # Noisy points (triangles) same scale
#     fig.add_trace(go.Scatter(
#         x=noisy_x, y=noisy_y, mode="markers", name="Noisy fitness",
#         marker=dict(symbol="triangle-up", size=8, color=noisy_c,
#                     colorscale="agsunset", showscale=False,
#                     cmin=gen_min, cmax=gen_max)
#     ))

#     fig.update_layout(
#         title=f"True vs Noisy Fitness — {series_name}, Run {run_idx}",
#         xaxis_title="Objective 1", yaxis_title="Objective 2",
#         template="plotly_white", legend_title="Type"
#     )
#     return fig

def _gen_color(g, gmin, gmax, scale="sunsetdark"):
    t = 0.5 if gmax == gmin else (g - gmin) / (gmax - gmin)
    return sample_colorscale(scale, [t])[0]  # returns "rgb(r,g,b)"

def plotParetoFrontNoisy(frontdata, series_labels):

    fig = go.Figure()
    if not frontdata:
        fig.update_layout(title="No multi-objective data"); return fig

    group_idx = run_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = str(series_labels[group_idx]) if series_labels and len(series_labels) > group_idx else f"Series {group_idx}"
    gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
    if not gen_entries:
        fig.update_layout(title=f"No generations for {series_name}"); return fig

    gens = [e.get("gen_idx", 0) for e in gen_entries]
    gmin, gmax = min(gens), max(gens)

    clean_x, clean_y, clean_c = [], [], []
    noisy_x, noisy_y, noisy_c = [], [], []

    for entry in gen_entries:
        g = entry.get("gen_idx", 0)
        clean_pts = entry.get("algo_front_clean_fitnesses") or []
        noisy_pts = entry.get("algo_front_noisy_fitnesses") or []
        col = _gen_color(g, gmin, gmax)

        for c_fit, n_fit in zip(clean_pts, noisy_pts):
            fig.add_trace(go.Scatter(
                x=[float(c_fit[0]), float(n_fit[0])],
                y=[float(c_fit[1]), float(n_fit[1])],
                mode="lines",
                line=dict(dash="dot", width=0.4, color=col),  # <-- use line.color
                hoverinfo="skip",
                showlegend=False
            ))
            clean_x.append(float(c_fit[0])); clean_y.append(float(c_fit[1])); clean_c.append(g)
            noisy_x.append(float(n_fit[0])); noisy_y.append(float(n_fit[1])); noisy_c.append(g)

    # points (same colorscale + shared range)
    fig.add_trace(go.Scatter(
        x=clean_x, y=clean_y, mode="markers", name="Clean fitness",
        marker=dict(symbol="circle", size=8, color=clean_c,
                    colorscale="sunsetdark", colorbar=dict(title="Generation"),
                    cmin=gmin, cmax=gmax)
    ))
    fig.add_trace(go.Scatter(
        x=noisy_x, y=noisy_y, mode="markers", name="Noisy fitness",
        marker=dict(symbol="triangle-up", size=4, color=noisy_c,
                    colorscale="sunsetdark", showscale=False,
                    cmin=gmin, cmax=gmax)
    ))

    fig.update_layout(
        title=f"True vs Noisy Fitness — {series_name}, Run {run_idx}",
        xaxis_title="Objective 1", yaxis_title="Objective 2",
        template="plotly_white", legend_title="Type"
    )
    return fig

# def plotParetoFrontIndVsDist(frontdata, series_labels):
#     import plotly.graph_objects as go

#     fig = go.Figure()

#     # ---- Basic guards ----
#     if not frontdata:
#         fig.update_layout(title="No multi-objective data")
#         return fig

#     group_idx = 0
#     run_idx = 0

#     runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
#     series_name = (
#         str(series_labels[group_idx])
#         if series_labels and len(series_labels) > group_idx
#         else f"Series {group_idx}"
#     )

#     if not runs_full:
#         fig.update_layout(title=f"No runs for {series_name}")
#         return fig

#     gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
#     if not gen_entries:
#         fig.update_layout(title=f"No generations for {series_name}")
#         return fig

#     # ---- Ensure entries are in generation order ----
#     gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))

#     xs = []
#     ys = []
#     gens = []

#     # cumulative distance, first front at 0
#     cum_dist = 0.0

#     # Handle the first front
#     first = gen_entries[0]
#     prev_front = first.get("algo_front_solutions") or []

#     xs.append(cum_dist)  # 0.0
#     ys.append(float(first.get("algo_front_noisy_hypervolume", 0.0)))
#     gens.append(first.get("gen_idx", 0))

#     # Remaining fronts: add incremental distance
#     for entry in gen_entries[1:]:
#         cur_front = entry.get("algo_front_solutions") or []
#         d = front_distance(prev_front, cur_front)
#         cum_dist += d

#         xs.append(cum_dist)
#         ys.append(float(entry.get("algo_front_noisy_hypervolume", 0.0)))
#         gens.append(entry.get("gen_idx", 0))

#         prev_front = cur_front

#     # ---- Plot (colour by generation) ----
#     gmin, gmax = min(gens), max(gens)
#     fig.add_trace(
#         go.Scatter(
#             x=xs,
#             y=ys,
#             mode="lines+markers",
#             name=f"{series_name}, Run {run_idx}",
#             marker=dict(
#                 size=8,
#                 color=gens,
#                 colorscale="sunsetdark",
#                 colorbar=dict(title="Generation"),
#                 cmin=gmin,
#                 cmax=gmax,
#             ),
#         )
#     )

#     fig.update_layout(
#         title=f"hypervolume vs front distance — {series_name}, Run {run_idx}",
#         xaxis_title="Cumulative distance between successive fronts\n(avg min Hamming, A→B)",
#         yaxis_title="Noisy PF hypervolume (algo_front_noisy_hypervolume)",
#         template="plotly_white",
#     )

#     return fig

# def plotParetoFrontIndVsDist(frontdata, series_labels):
#     fig = go.Figure()

#     # ----- choose distance behaviour here -----
#     distance_method = "cumulative"   # or "mds"
#     # distance_method = "mds"
#     solution_set = "algo_front_solutions"
#     metric = "algo_front_noisy_hypervolume"
#     # ------------------------------------------

#     # ---- Basic guards ----
#     if not frontdata:
#         fig.update_layout(title="No multi-objective data")
#         return fig

#     group_idx = 0
#     run_idx = 0

#     runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
#     series_name = (
#         str(series_labels[group_idx])
#         if series_labels and len(series_labels) > group_idx
#         else f"Series {group_idx}"
#     )

#     if not runs_full:
#         fig.update_layout(title=f"No runs for {series_name}")
#         return fig

#     gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
#     if not gen_entries:
#         fig.update_layout(title=f"No generations for {series_name}")
#         return fig

#     # ---- Ensure entries are in generation order ----
#     gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))

#     # Extract fronts, hypervolumes, generations in order once
#     fronts = []
#     hvs = []
#     gens = []
#     for entry in gen_entries:
#         fronts.append(entry.get(solution_set) or [])
#         hvs.append(float(entry.get(metric, 0.0)))
#         gens.append(entry.get("gen_idx", 0))

#     K = len(fronts)
#     if K == 0:
#         fig.update_layout(title=f"No fronts for {series_name}")
#         return fig

#     # ---- Compute x-coordinates depending on method ----
#     if distance_method == "cumulative":
#         xs = []
#         cum_dist = 0.0
#         xs.append(cum_dist)  # first front at 0

#         prev_front = fronts[0]
#         for cur_front in fronts[1:]:
#             d = front_distance(prev_front, cur_front)
#             cum_dist += d
#             xs.append(cum_dist)
#             prev_front = cur_front

#         xaxis_title = "Cumulative distance between successive fronts\n(avg min Hamming)"
#         plot_title = f"Hypervolume vs cumulative front distance — {series_name}, Run {run_idx}"

#     elif distance_method == "mds":
#         # Build full symmetric distance matrix
#         D = np.zeros((K, K), dtype=float)
#         for i in range(K):
#             for j in range(i + 1, K):
#                 d = front_distance(fronts[i], fronts[j])
#                 D[i, j] = D[j, i] = float(d)

#         # 1D MDS embedding from the pairwise distance matrix
#         mds = MDS_sklearn(
#             n_components=1,
#             dissimilarity='precomputed',
#             random_state=42
#         )
#         X_coords = mds.fit_transform(D)  # shape (K, 1)
#         xs_raw = X_coords[:, 0]

#         # Shift so minimum is at 0 for nicer plotting
#         xs = (xs_raw - xs_raw.min()).tolist()

#         xaxis_title = "MDS 1D embedding of front distances\n(avg min Hamming)"
#         plot_title = f"Hypervolume vs MDS front distance — {series_name}, Run {run_idx}"

#     else:
#         # Fallback: default to cumulative if someone sets a weird value
#         xs = list(range(K))
#         xaxis_title = "Index (fallback)"
#         plot_title = f"Hypervolume vs index (fallback) — {series_name}, Run {run_idx}"

#     # ---- Plot (colour by generation) ----
#     gmin, gmax = min(gens), max(gens)
#     fig.add_trace(
#         go.Scatter(
#             x=xs,
#             y=hvs,
#             mode="lines+markers",
#             name=f"{series_name}, Run {run_idx}",
#             marker=dict(
#                 size=8,
#                 color=gens,
#                 colorscale="sunsetdark",
#                 colorbar=dict(title="Generation"),
#                 cmin=gmin,
#                 cmax=gmax,
#             ),
#         )
#     )

#     fig.update_layout(
#         title=plot_title,
#         xaxis_title=xaxis_title,
#         yaxis_title=f"Hypervolume ({metric})",
#         template="plotly_white",
#     )

#     return fig

# def plotParetoFrontIndVsDist(frontdata, series_labels):
#     fig = go.Figure()

#     # ----- options -----
#     distance_method = "cumulative"   # or "mds"
#     # distance_method = "mds"
#     solution_set    = "algo_front_solutions"
#     metric          = "algo_front_noisy_hypervolume"
#     num_runs        = 3
#     # -------------------

#     # ---- Basic guards ----
#     if not frontdata:
#         fig.update_layout(title="No multi-objective data")
#         return fig

#     group_idx = 0
#     runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
#     series_name = (
#         str(series_labels[group_idx])
#         if series_labels and len(series_labels) > group_idx
#         else f"Series {group_idx}"
#     )

#     if not runs_full:
#         fig.update_layout(title=f"No runs for {series_name}")
#         return fig

#     # Decide which run indices to plot
#     max_run = min(num_runs, len(runs_full))
#     run_indices = range(0, max_run)

#     all_gens = []   # for global colour range
#     run_data = []   # store per-run x, y, gens for plotting

#     # ---- Process each run separately ----
#     for run_idx in run_indices:
#         gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
#         if not gen_entries:
#             continue

#         # Ensure entries are in generation order
#         gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))

#         # Extract fronts, hypervolumes, generations in order
#         fronts = []
#         hvs    = []
#         gens   = []
#         for entry in gen_entries:
#             fronts.append(entry.get(solution_set) or [])
#             hvs.append(float(entry.get(metric, 0.0)))
#             gens.append(entry.get("gen_idx", 0))

#         K = len(fronts)
#         if K == 0:
#             continue

#         # ---- Compute x-coordinates depending on method for THIS run ----
#         if distance_method == "cumulative":
#             xs = []
#             cum_dist = 0.0
#             xs.append(cum_dist)  # first front at 0

#             prev_front = fronts[0]
#             for cur_front in fronts[1:]:
#                 d = front_distance(prev_front, cur_front)
#                 cum_dist += d
#                 xs.append(cum_dist)
#                 prev_front = cur_front

#             xaxis_title = "Cumulative distance between successive fronts\n(avg min Hamming)"
#             plot_title  = f"Hypervolume vs cumulative front distance — {series_name}"

#         elif distance_method == "mds":
#             # Build full symmetric distance matrix for this run
#             D = np.zeros((K, K), dtype=float)
#             for i in range(K):
#                 for j in range(i + 1, K):
#                     d = front_distance(fronts[i], fronts[j])
#                     D[i, j] = D[j, i] = float(d)

#             # 1D MDS embedding from the pairwise distance matrix
#             mds = MDS_sklearn(
#                 n_components=1,
#                 dissimilarity='precomputed',
#                 n_init=8,
#                 max_iter=1500,
#                 random_state=42
#             )
#             X_coords = mds.fit_transform(D)  # shape (K, 1)
#             xs_raw = X_coords[:, 0]

#             # Shift so minimum is at 0 for nicer plotting
#             xs = (xs_raw - xs_raw.min()).tolist()

#             xaxis_title = "MDS 1D embedding of front distances\n(avg min Hamming)"
#             plot_title  = f"Hypervolume vs MDS front distance — {series_name}"

#         else:
#             # Fallback: index
#             xs = list(range(K))
#             xaxis_title = "Index (fallback)"
#             plot_title  = f"Hypervolume vs index (fallback) — {series_name}"

#         # Store for plotting later
#         all_gens.extend(gens)
#         run_data.append({
#             "run_idx": run_idx,
#             "xs": xs,
#             "hvs": hvs,
#             "gens": gens,
#         })

#     # If nothing valid was collected
#     if not run_data:
#         fig.update_layout(title=f"No usable runs for {series_name}")
#         return fig

#     # ---- Plot all runs, sharing colour range for generations ----
#     gmin, gmax = min(all_gens), max(all_gens)

#     for k, rd in enumerate(run_data):
#         show_cb = (k == 0)  # only first run shows the colorbar
#         fig.add_trace(
#             go.Scatter(
#                 x=rd["xs"],
#                 y=rd["hvs"],
#                 mode="lines+markers",
#                 name=f"{series_name}, Run {rd['run_idx']}",
#                 marker=dict(
#                     size=8,
#                     color=rd["gens"],
#                     colorscale="sunsetdark",
#                     colorbar=dict(title="Generation") if show_cb else None,
#                     showscale=show_cb,
#                     cmin=gmin,
#                     cmax=gmax,
#                 ),
#             )
#         )

#     fig.update_layout(
#         title=plot_title,
#         xaxis_title=xaxis_title,
#         yaxis_title=f"Hypervolume ({metric})",
#         template="plotly_white",
#     )

#     return fig

def plotParetoFrontIndVsDist(frontdata, series_labels, distance_method='cumulative', nruns=1):
    fig = go.Figure()

    # ----- options -----
    # distance_method = "cumulative"   # or "mds" or "tsne" or "isomap"
    # distance_method = "cumulative"
    solution_set    = "algo_front_solutions"
    metric          = "algo_front_noisy_hypervolume"
    num_runs        = nruns
    # -------------------

    # ---- Basic guards ----
    if not frontdata:
        fig.update_layout(title="No multi-objective data")
        return fig

    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    # Decide which run indices to plot
    max_run = min(num_runs, len(runs_full))
    run_indices = range(0, max_run)

    all_gens = []   # for global colour range
    run_data = []   # store per-run fronts, hvs, gens (and later xs)

    # ---- First pass: collect data for each run ----
    for run_idx in run_indices:
        gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
        if not gen_entries:
            continue

        # Ensure entries are in generation order
        gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))

        fronts = []
        hvs    = []
        gens   = []
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
            "xs": None,   # will fill later
        })

    # If nothing valid was collected
    if not run_data:
        fig.update_layout(title=f"No usable runs for {series_name}")
        return fig

    # ---- Compute x-coordinates depending on method ----
    if distance_method == "cumulative":
        for rd in run_data:
            fronts = rd["fronts"]
            K = len(fronts)

            xs = []
            cum_dist = 0.0
            xs.append(cum_dist)  # first front at 0
            prev_front = fronts[0]

            for cur_front in fronts[1:]:
                d = front_distance(prev_front, cur_front)
                cum_dist += d
                xs.append(cum_dist)
                prev_front = cur_front

            rd["xs"] = xs

        xaxis_title = "Cumulative distance between successive fronts\n(avg min Hamming)"
        plot_title  = f"Hypervolume vs cumulative front distance — {series_name}"

    elif distance_method == "mds":
        # --- Global MDS over ALL fronts from ALL runs ---
        # Flatten fronts in (run order, then gen order)
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        # Build full symmetric distance matrix over all fronts
        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        # 1D MDS embedding from the global pairwise distance matrix
        mds = MDS_sklearn(
            n_components=1,
            dissimilarity='precomputed',
            n_init=4,
            max_iter=1000,
            random_state=42,
        )
        X_coords = mds.fit_transform(D)  # shape (N, 1)
        xs_raw = X_coords[:, 0]

        # Shift so minimum is at 0 for nicer plotting
        xs_global = xs_raw - xs_raw.min()

        # Slice global xs back into per-run sequences
        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "MDS 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title  = f"Hypervolume vs MDS front distance — {series_name}"

    elif distance_method == "tsne":
        # --- Global t-SNE over ALL fronts from ALL runs ---
        # Flatten fronts in (run order, then gen order)
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        # Build full symmetric distance matrix over all fronts
        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        # Handle tiny N explicitly so t-SNE doesn't complain about perplexity
        if N == 1:
            xs_global = np.array([0.0], dtype=float)
        elif N == 2:
            xs_global = np.array([0.0, 1.0], dtype=float)
        else:
            # Choose a reasonable perplexity given N
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
            X_coords = tsne.fit_transform(D)   # shape (N, 1)
            xs_raw = X_coords[:, 0]

            # Shift so minimum is at 0 for nicer plotting
            xs_global = xs_raw - xs_raw.min()

        # Slice global xs back into per-run sequences
        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "t-SNE 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title  = f"Hypervolume vs t-SNE front distance — {series_name}"
    
    elif distance_method == "isomap":
        # --- Global Isomap over ALL fronts from ALL runs ---
        # Flatten fronts in (run order, then gen order)
        flat_fronts = [front for rd in run_data for front in rd["fronts"]]
        N = len(flat_fronts)

        # Build full symmetric distance matrix over all fronts
        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                d = front_distance(flat_fronts[i], flat_fronts[j])
                D[i, j] = D[j, i] = float(d)

        # Handle tiny N explicitly so neighbors / embedding don’t blow up
        if N == 1:
            xs_global = np.array([0.0], dtype=float)
        elif N == 2:
            xs_global = np.array([0.0, 1.0], dtype=float)
        else:
            # Choose a sensible number of neighbors
            # (must be < N)
            n_neighbors = min(10, N - 1)
            if n_neighbors < 2:
                n_neighbors = 2

            iso = Isomap(
                n_neighbors=n_neighbors,
                n_components=1,
                metric="precomputed",
            )
            X_coords = iso.fit_transform(D)  # shape (N, 1)
            xs_raw = X_coords[:, 0]

            # Shift so minimum is at 0 for nicer plotting
            xs_global = xs_raw - xs_raw.min()

        # Slice global xs back into per-run sequences
        idx = 0
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = xs_global[idx:idx + K].tolist()
            idx += K

        xaxis_title = "Isomap 1D embedding of front distances\n(avg min Hamming, all runs)"
        plot_title  = f"Hypervolume vs Isomap front distance — {series_name}"

    else:
        # Fallback: simple index
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = list(range(K))

        xaxis_title = "Index (fallback)"
        plot_title  = f"Hypervolume vs index (fallback) — {series_name}"

    # ---- Plot all runs, sharing colour range for generations ----
    gmin, gmax = min(all_gens), max(all_gens)

    for k, rd in enumerate(run_data):
        show_cb = (k == 0)  # only first run shows the colorbar
        fig.add_trace(
            go.Scatter(
                x=rd["xs"],
                y=rd["hvs"],
                mode="lines+markers",
                name=f"{series_name}, Run {rd['run_idx']}",
                marker=dict(
                    size=8,
                    color=rd["gens"],
                    colorscale="sunsetdark",
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
        template="plotly_white",
    )

    return fig

# def plotParetoFrontIGDVsDist(frontdata, series_labels, distance_method='cumulative', nruns=1):
#     fig = go.Figure()

#     # ----- options -----
#     solution_set      = "algo_front_solutions"          # used for x-axis distances (decision-space fronts)
#     approx_fits_key   = "algo_front_noisy_fitnesses"    # used for IGD (approx front in objective space)
#     ref_fits_key      = "clean_front_fitnesses"         # final generation used as IGD reference front
#     num_runs          = nruns
#     # -------------------

#     # ---- helpers ----
#     def euclid(a, b):
#         a = np.asarray(a, dtype=float)
#         b = np.asarray(b, dtype=float)
#         return float(np.linalg.norm(a - b))

#     def igd(approx_front, ref_front):
#         """
#         IGD(P, R) = (1/|R|) * sum_{r in R} min_{p in P} d(r, p)
#         """
#         if not ref_front:
#             return np.nan
#         if not approx_front:
#             return np.nan

#         # Convert once (robust to tuples/lists)
#         A = [np.asarray(p, dtype=float) for p in approx_front]
#         R = [np.asarray(r, dtype=float) for r in ref_front]

#         mins = []
#         for r in R:
#             mins.append(min(float(np.linalg.norm(r - p)) for p in A))
#         return float(np.mean(mins)) if mins else np.nan

#     # ---- Basic guards ----
#     if not frontdata:
#         fig.update_layout(title="No multi-objective data")
#         return fig

#     group_idx = 0
#     runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
#     series_name = (
#         str(series_labels[group_idx])
#         if series_labels and len(series_labels) > group_idx
#         else f"Series {group_idx}"
#     )

#     if not runs_full:
#         fig.update_layout(title=f"No runs for {series_name}")
#         return fig

#     # Decide which run indices to plot
#     max_run = min(num_runs, len(runs_full))
#     run_indices = range(0, max_run)

#     all_gens = []   # for global colour range
#     run_data = []   # store per-run fronts (decision-space), igds, gens (and later xs)

#     # ---- First pass: collect data for each run ----
#     for run_idx in run_indices:
#         gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
#         if not gen_entries:
#             continue

#         # Ensure entries are in generation order
#         gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))
#         last_entry = gen_entries[-1] if gen_entries else {}

#         # Reference front for IGD = final generation clean front fitnesses
#         ref_front = last_entry.get(ref_fits_key) or []

#         fronts = []  # decision-space fronts for x-axis
#         igds  = []
#         gens  = []

#         for entry in gen_entries:
#             fronts.append(entry.get(solution_set) or [])
#             approx_front = entry.get(approx_fits_key) or []
#             igds.append(igd(approx_front, ref_front))
#             gens.append(entry.get("gen_idx", 0))

#         K = len(fronts)
#         if K == 0:
#             continue

#         all_gens.extend(gens)
#         run_data.append({
#             "run_idx": run_idx,
#             "fronts": fronts,
#             "igds": igds,
#             "gens": gens,
#             "xs": None,   # will fill later
#         })

#     # If nothing valid was collected
#     if not run_data:
#         fig.update_layout(title=f"No usable runs for {series_name}")
#         return fig

#     # ---- Compute x-coordinates depending on method ----
#     if distance_method == "cumulative":
#         for rd in run_data:
#             fronts = rd["fronts"]
#             xs = []
#             cum_dist = 0.0
#             xs.append(cum_dist)
#             prev_front = fronts[0]

#             for cur_front in fronts[1:]:
#                 d = front_distance(prev_front, cur_front)
#                 cum_dist += float(d)
#                 xs.append(cum_dist)
#                 prev_front = cur_front

#             rd["xs"] = xs

#         xaxis_title = "Cumulative distance between successive fronts\n(avg min Hamming)"
#         plot_title  = f"IGD vs cumulative front distance — {series_name}"

#     elif distance_method == "mds":
#         flat_fronts = [front for rd in run_data for front in rd["fronts"]]
#         N = len(flat_fronts)

#         D = np.zeros((N, N), dtype=float)
#         for i in range(N):
#             for j in range(i + 1, N):
#                 d = front_distance(flat_fronts[i], flat_fronts[j])
#                 D[i, j] = D[j, i] = float(d)

#         mds = MDS_sklearn(
#             n_components=1,
#             dissimilarity='precomputed',
#             n_init=4,
#             max_iter=1000,
#             random_state=42,
#         )
#         X_coords = mds.fit_transform(D)
#         xs_raw = X_coords[:, 0]
#         xs_global = xs_raw - xs_raw.min()

#         idx = 0
#         for rd in run_data:
#             K = len(rd["fronts"])
#             rd["xs"] = xs_global[idx:idx + K].tolist()
#             idx += K

#         xaxis_title = "MDS 1D embedding of front distances\n(avg min Hamming, all runs)"
#         plot_title  = f"IGD vs MDS front distance — {series_name}"

#     elif distance_method == "tsne":
#         flat_fronts = [front for rd in run_data for front in rd["fronts"]]
#         N = len(flat_fronts)

#         D = np.zeros((N, N), dtype=float)
#         for i in range(N):
#             for j in range(i + 1, N):
#                 d = front_distance(flat_fronts[i], flat_fronts[j])
#                 D[i, j] = D[j, i] = float(d)

#         if N == 1:
#             xs_global = np.array([0.0], dtype=float)
#         elif N == 2:
#             xs_global = np.array([0.0, 1.0], dtype=float)
#         else:
#             perplexity = min(30.0, max(5.0, (N - 1) / 3.0))
#             if perplexity >= N:
#                 perplexity = N - 1.0

#             tsne = TSNE(
#                 n_components=1,
#                 metric="precomputed",
#                 perplexity=perplexity,
#                 max_iter=1000,
#                 random_state=42,
#                 init="random",
#             )
#             X_coords = tsne.fit_transform(D)
#             xs_raw = X_coords[:, 0]
#             xs_global = xs_raw - xs_raw.min()

#         idx = 0
#         for rd in run_data:
#             K = len(rd["fronts"])
#             rd["xs"] = xs_global[idx:idx + K].tolist()
#             idx += K

#         xaxis_title = "t-SNE 1D embedding of front distances\n(avg min Hamming, all runs)"
#         plot_title  = f"IGD vs t-SNE front distance — {series_name}"

#     elif distance_method == "isomap":
#         flat_fronts = [front for rd in run_data for front in rd["fronts"]]
#         N = len(flat_fronts)

#         D = np.zeros((N, N), dtype=float)
#         for i in range(N):
#             for j in range(i + 1, N):
#                 d = front_distance(flat_fronts[i], flat_fronts[j])
#                 D[i, j] = D[j, i] = float(d)

#         if N == 1:
#             xs_global = np.array([0.0], dtype=float)
#         elif N == 2:
#             xs_global = np.array([0.0, 1.0], dtype=float)
#         else:
#             n_neighbors = min(10, N - 1)
#             if n_neighbors < 2:
#                 n_neighbors = 2

#             iso = Isomap(
#                 n_neighbors=n_neighbors,
#                 n_components=1,
#                 metric="precomputed",
#             )
#             X_coords = iso.fit_transform(D)
#             xs_raw = X_coords[:, 0]
#             xs_global = xs_raw - xs_raw.min()

#         idx = 0
#         for rd in run_data:
#             K = len(rd["fronts"])
#             rd["xs"] = xs_global[idx:idx + K].tolist()
#             idx += K

#         xaxis_title = "Isomap 1D embedding of front distances\n(avg min Hamming, all runs)"
#         plot_title  = f"IGD vs Isomap front distance — {series_name}"

#     else:
#         for rd in run_data:
#             K = len(rd["fronts"])
#             rd["xs"] = list(range(K))

#         xaxis_title = "Index (fallback)"
#         plot_title  = f"IGD vs index (fallback) — {series_name}"

#     # ---- Plot all runs, sharing colour range for generations ----
#     gmin, gmax = min(all_gens), max(all_gens)

#     for k, rd in enumerate(run_data):
#         show_cb = (k == 0)
#         fig.add_trace(
#             go.Scatter(
#                 x=rd["xs"],
#                 y=rd["igds"],
#                 mode="lines+markers",
#                 name=f"{series_name}, Run {rd['run_idx']}",
#                 marker=dict(
#                     size=8,
#                     color=rd["gens"],
#                     colorscale="sunsetdark",
#                     colorbar=dict(title="Generation") if show_cb else None,
#                     showscale=show_cb,
#                     cmin=gmin,
#                     cmax=gmax,
#                 ),
#             )
#         )

#     fig.update_layout(
#         title=plot_title,
#         xaxis_title=xaxis_title,
#         yaxis_title=f"IGD (from ref front)",
#         template="plotly_white",
#     )
#     return fig

def plotParetoFrontIGDVsDist(frontdata, series_labels, distance_method='cumulative', nruns=1):
    fig = go.Figure()

    # ----- options -----
    solution_set      = "algo_front_solutions"          # used for x-axis distances (decision-space fronts)
    approx_fits_key   = "algo_front_noisy_fitnesses"    # used for IGD (approx front in objective space)
    ref_fits_key      = "clean_front_fitnesses"         # final generation used as IGD reference front
    num_runs          = nruns
    # -------------------

    # ---- helpers ----
    def euclid(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return float(np.linalg.norm(a - b))

    def igd(approx_front, ref_front):
        """
        IGD(P, R) = (1/|R|) * sum_{r in R} min_{p in P} d(r, p)
        """
        if not ref_front:
            return np.nan
        if not approx_front:
            return np.nan

        # Convert once (robust to tuples/lists)
        A = [np.asarray(p, dtype=float) for p in approx_front]
        R = [np.asarray(r, dtype=float) for r in ref_front]

        mins = []
        for r in R:
            mins.append(min(float(np.linalg.norm(r - p)) for p in A))
        return float(np.mean(mins)) if mins else np.nan

    # ---- Basic guards ----
    if not frontdata:
        fig.update_layout(title="No multi-objective data")
        return fig

    group_idx = 0
    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig

    # Decide which run indices to plot
    max_run = min(num_runs, len(runs_full))
    run_indices = range(0, max_run)

    all_gens = []   # for global colour range
    run_data = []   # store per-run fronts (decision-space), igds, gens (and later xs)

    # ---- First pass: collect data for each run ----
    for run_idx in run_indices:
        gen_entries = runs_full[run_idx] if len(runs_full) > run_idx else []
        if not gen_entries:
            continue

        # Ensure entries are in generation order
        gen_entries = sorted(gen_entries, key=lambda e: e.get("gen_idx", 0))
        last_entry = gen_entries[-1] if gen_entries else {}

        # Reference front for IGD = final generation clean front fitnesses
        ref_front = last_entry.get(ref_fits_key) or []

        fronts = []  # decision-space fronts for x-axis
        igds  = []
        gens  = []

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
            "xs": None,   # will fill later
        })

    # If nothing valid was collected
    if not run_data:
        fig.update_layout(title=f"No usable runs for {series_name}")
        return fig

    # ---- Compute x-coordinates depending on method ----
    if distance_method == "cumulative":
        for rd in run_data:
            fronts = rd["fronts"]
            xs = []
            cum_dist = 0.0
            xs.append(cum_dist)
            prev_front = fronts[0]

            for cur_front in fronts[1:]:
                d = front_distance(prev_front, cur_front)
                cum_dist += float(d)
                xs.append(cum_dist)
                prev_front = cur_front

            rd["xs"] = xs

        xaxis_title = "Cumulative distance between successive fronts\n(avg min Hamming)"
        plot_title  = f"IGD vs cumulative front distance — {series_name}"

    elif distance_method == "raw":
        # Per-step distance to previous front (not cumulative)
        for rd in run_data:
            fronts = rd["fronts"]
            xs = []
            if not fronts:
                rd["xs"] = xs
                continue

            xs.append(0.0)  # first gen has no previous front
            prev_front = fronts[0]
            for cur_front in fronts[1:]:
                d = front_distance(prev_front, cur_front)
                xs.append(float(d))
                prev_front = cur_front

            rd["xs"] = xs

        xaxis_title = "Raw distance to previous front\n(avg min Hamming)"
        plot_title  = f"IGD vs raw front distance — {series_name}"

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
        plot_title  = f"IGD vs MDS front distance — {series_name}"

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
        plot_title  = f"IGD vs t-SNE front distance — {series_name}"

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
        plot_title  = f"IGD vs Isomap front distance — {series_name}"

    else:
        for rd in run_data:
            K = len(rd["fronts"])
            rd["xs"] = list(range(K))

        xaxis_title = "Index (fallback)"
        plot_title  = f"IGD vs index (fallback) — {series_name}"

    # ---- Plot all runs, sharing colour range for generations ----
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
                    colorscale="sunsetdark",
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
        yaxis_title=f"IGD (from ref front)",
        template="plotly_white",
    )

    return fig

# --------------------------------------------------
# Objective space vs decision space change
# --------------------------------------------------

def plotProgressPerMovementRatio(
    frontdata,
    series_labels,
    group_idx=0,
    run_idx=0,
    solution_key="algo_front_solutions",
    # hv_key="algo_front_noisy_hypervolume",
    hv_key="algo_front_clean_hypervolume",
    eps=1e-12,
    k_patience=10,
    use_ratio=True,
    show_deltas=True,
    ):
    """
    """

    # ---------------- basic guards ----------------
    fig = go.Figure()
    if not frontdata:
        fig.update_layout(title="No MO_data_PPP provided")
        return fig, None, {}

    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig, None, {}

    if len(runs_full) <= run_idx:
        fig.update_layout(title=f"Run {run_idx} not available for {series_name}")
        return fig, None, {}

    gen_entries = runs_full[run_idx]
    if not gen_entries or len(gen_entries) < 2:
        fig.update_layout(title=f"Not enough generations for {series_name} (run {run_idx})")
        return fig, None, {}

    # -------------- compute per-gen series --------------
    gens = []
    delta_D = []
    delta_O = []
    rho = []

    # pull per-gen fronts and HV
    sols = [e.get(solution_key) or [] for e in gen_entries]
    hvs  = [e.get(hv_key, None) for e in gen_entries]
    gen_ids = [e.get("gen_idx", i) for i, e in enumerate(gen_entries)]

    for t in range(1, len(gen_entries)):
        prev_s, cur_s = sols[t-1], sols[t]
        dD = float(front_distance(prev_s, cur_s))

        # Objective "progress": positive improvement in noisy HV
        prev_hv = hvs[t-1]
        cur_hv  = hvs[t]
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
    rho     = np.asarray(rho, dtype=float)

    # -------------- find stop point --------------
    # percentile thresholds (adaptive)
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

    # -------------- plot --------------
    fig.add_trace(go.Scatter(
        x=gens, y=rho, mode="lines+markers",
        name="ΔO/ΔD"
    ))

    if show_deltas:
        fig.add_trace(go.Scatter(
            x=gens, y=delta_D, mode="lines",
            name="ΔD (decision change)", yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=gens, y=delta_O, mode="lines",
            name="ΔO (HV improvement)", yaxis="y3"
        ))

    fig.update_layout(
        title=f"Progress movement ratio — {series_name} (run {run_idx})",
        xaxis=dict(title="Generation"),
        yaxis=dict(title="ΔO/ΔD"),
        yaxis2=dict(
            title="ΔD", overlaying="y", side="right",
            showgrid=False
        ),
        yaxis3=dict(
            title="ΔO", overlaying="y", side="right",
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

    debug = dict(
        tauD=tauD, tauO=tauO, tauR=tauR,
        gens=gens, delta_D=delta_D, delta_O=delta_O, rho=rho
    )
    # return fig, stop_gen, debug
    return fig


def plotMovementCorrelation(
    frontdata,
    series_labels,
    group_idx=0,
    run_idx=0,
    solution_key="algo_front_solutions",
    IndVsDist_IndType="NoisyHV",
    # hv_key="algo_front_clean_hypervolume",
    # hv_key="algo_front_noisy_hypervolume",
    window=25,                                # sliding window size
    corr_method="pearson",                  # "spearman" or "pearson"
    show_deltas=True,
    ):
    """
    Plot sliding-window correlation between decision movement (ΔD) and
    objective progress (ΔO, e.g., ΔHV) across generations.

    Returns:
        fig (plotly.graph_objects.Figure)
    """
    def symmetric_range(arr, pad=0.05):
        m = np.nanmax(np.abs(arr))
        if not np.isfinite(m) or m == 0:
            m = 1.0
        m *= (1 + pad)
        return [-m, m]

    # ---------------- basic guards ----------------
    fig = go.Figure()
    if not frontdata:
        fig.update_layout(title="No MO_data provided")
        return fig

    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )

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

    # -------------- compute per-gen series --------------
    if IndVsDist_IndType == 'NoisyHV':
        hv_key = 'algo_front_noisy_hypervolume'
    if IndVsDist_IndType == 'CleanHV':
        hv_key = 'algo_front_clean_hypervolume'
    
    gens = []
    delta_D = []
    delta_O = []

    sols = [e.get(solution_key) or [] for e in gen_entries]
    hvs  = [e.get(hv_key, None) for e in gen_entries]
    gen_ids = [e.get("gen_idx", i) for i, e in enumerate(gen_entries)]

    for t in range(1, len(gen_entries)):
        prev_s, cur_s = sols[t-1], sols[t]
        dD = float(front_distance(prev_s, cur_s))

        prev_hv = hvs[t-1]
        cur_hv  = hvs[t]
        if prev_hv is None or cur_hv is None:
            dO = 0.0
        else:
            dO = float(cur_hv - prev_hv)

        gens.append(int(gen_ids[t]))
        delta_D.append(dD)
        delta_O.append(dO)

    gens = np.asarray(gens)
    delta_D = np.asarray(delta_D, dtype=float)
    delta_O = np.asarray(delta_O, dtype=float)

    corr_range = [-1.1, 1.1]
    dD_range   = symmetric_range(delta_D)
    dO_range   = symmetric_range(delta_O)

    # -------------- sliding correlation --------------
    def _corr(x, y, method="spearman"):
        # guard: constant vectors -> correlation undefined
        if np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return np.nan

        if method == "pearson":
            # np.corrcoef returns 2x2 matrix
            return float(np.corrcoef(x, y)[0, 1])

        # spearman: rank-transform then pearson
        # (avoid scipy dependency, works fine)
        rx = x.argsort().argsort().astype(float)
        ry = y.argsort().argsort().astype(float)
        if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
            return np.nan
        return float(np.corrcoef(rx, ry)[0, 1])

    corr = np.full_like(delta_D, fill_value=np.nan, dtype=float)

    w = int(window)
    if w < 2:
        w = 2

    for i in range(len(delta_D)):
        if i < w - 1:
            continue
        xs = delta_D[i-w+1:i+1]
        ys = delta_O[i-w+1:i+1]
        corr[i] = _corr(xs, ys, method=corr_method)

    # -------------- plot --------------
    fig.add_trace(go.Scatter(
        x=gens, y=corr, mode="lines+markers",
        name=f"{corr_method.title()} corr(ΔD, ΔO), window={w}"
    ))

    if show_deltas:
        fig.add_trace(go.Scatter(
            x=gens, y=delta_D, mode="lines",
            name="ΔD (decision change)", yaxis="y2"
        ))
        fig.add_trace(go.Scatter(
            x=gens, y=delta_O, mode="lines",
            name="ΔO (HV change)", yaxis="y3"
        ))

    fig.update_layout(
        title=f"Movement correlation — {series_name} (run {run_idx})",
        xaxis=dict(title="Generation"),

        yaxis=dict(
            title="Correlation (-1 to 1)",
            range=corr_range,
            zeroline=True,
            zerolinewidth=2
        ),

        yaxis2=dict(
            title="ΔD",
            overlaying="y",
            side="right",
            range=dD_range,
            zeroline=True,
            zerolinewidth=2,
            showgrid=False
        ),

        yaxis3=dict(
            title="ΔO",
            overlaying="y",
            side="right",
            anchor="free",
            position=1.0,
            range=dO_range,
            zeroline=True,
            zerolinewidth=2,
            showgrid=False
        ),

        legend=dict(orientation="h"),
    )

    return fig


def plotMoveDeltaHistograms(
    frontdata,
    series_labels,
    group_idx=0,
    solution_key="algo_front_solutions",
    IndVsDist_IndType="NoisyHV",
    bins_decision=50,
    bins_objective=50,
    include_zero_moves=False,                 # include gens where front didn't change (ΔD=0, ΔO=0)
):
    """
    Pools per-generation move deltas across ALL runs for one series/group, then plots:
      - histogram of ΔD (decision-space front distance)
      - histogram of ΔO (objective-space change, e.g. ΔHV)

    Returns:
        fig (plotly Figure)
        debug (dict) with pooled arrays and counts
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("ΔD (decision-space front distance)", f"ΔO (objective change)")
    )

    if IndVsDist_IndType == 'NoisyHV':
        hv_key = 'algo_front_noisy_hypervolume'
    if IndVsDist_IndType == 'CleanHV':
        hv_key = 'algo_front_clean_hypervolume'

    # ---------------- guards ----------------
    if not frontdata:
        fig.update_layout(title="No data provided")
        return fig, {"delta_D": np.array([]), "delta_O": np.array([]), "n_runs": 0, "n_moves": 0}

    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )
    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig, {"delta_D": np.array([]), "delta_O": np.array([]), "n_runs": 0, "n_moves": 0}

    # ---------------- pool deltas across runs ----------------
    all_delta_D = []
    all_delta_O = []

    for run_idx, gen_entries in enumerate(runs_full):
        if not gen_entries or len(gen_entries) < 2:
            continue

        sols = [e.get(solution_key) or [] for e in gen_entries]
        hvs  = [e.get(hv_key, None) for e in gen_entries]

        for t in range(1, len(gen_entries)):
            prev_s, cur_s = sols[t-1], sols[t]
            dD = float(front_distance(prev_s, cur_s))

            prev_hv = hvs[t-1]
            cur_hv  = hvs[t]
            if prev_hv is None or cur_hv is None:
                dO = 0.0
            else:
                dO = float(cur_hv - prev_hv)

            if include_zero_moves or (dD != 0.0 or dO != 0.0):
                all_delta_D.append(dD)
                all_delta_O.append(dO)

    all_delta_D = np.asarray(all_delta_D, dtype=float)
    all_delta_O = np.asarray(all_delta_O, dtype=float)

    # ---------------- plot histograms ----------------
    fig.add_trace(
        go.Histogram(x=all_delta_D, nbinsx=bins_decision, name="ΔD"),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=all_delta_O, nbinsx=bins_objective, name="ΔO"),
        row=1, col=2
    )

    fig.update_layout(
        title=f"Move delta histograms (pooled across runs) — {series_name}",
        bargap=0.05,
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="ΔD", row=1, col=1)
    fig.update_xaxes(title_text="ΔO", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    debug = {
        "series_name": series_name,
        "n_runs": len(runs_full),
        "n_moves": int(len(all_delta_D)),
        "delta_D": all_delta_D,
        "delta_O": all_delta_O,
        "include_zero_moves": include_zero_moves,
    }
    return fig

def plotObjectiveVsDecisionScatter(
    frontdata,
    series_labels,
    group_idx=0,
    run_idx=None,                            # if None -> pool across all runs
    solution_key="algo_front_solutions",
    IndVsDist_IndType="NoisyHV",   # or "algo_front_noisy_hypervolume"
    include_zero_moves=True,
    color_by="gen",                          # "gen" or "run" or None
    marker_size=6,
):
    """
    """
    fig = go.Figure()

    if IndVsDist_IndType == 'NoisyHV':
        hv_key = 'algo_front_noisy_hypervolume'
    if IndVsDist_IndType == 'CleanHV':
        hv_key = 'algo_front_clean_hypervolume'

    if not frontdata:
        fig.update_layout(title="No data provided")
        return fig, {}

    runs_full = frontdata[group_idx] if len(frontdata) > group_idx else []
    series_name = (
        str(series_labels[group_idx])
        if series_labels and len(series_labels) > group_idx
        else f"Series {group_idx}"
    )
    if not runs_full:
        fig.update_layout(title=f"No runs for {series_name}")
        return fig, {}

    # Decide which runs to use
    if run_idx is None:
        run_indices = list(range(len(runs_full)))
    else:
        if run_idx >= len(runs_full):
            fig.update_layout(title=f"Run {run_idx} not available for {series_name}")
            return fig, {}
        run_indices = [run_idx]

    all_dD, all_dO, all_gen, all_run = [], [], [], []

    for r in run_indices:
        gen_entries = runs_full[r]
        if not gen_entries or len(gen_entries) < 2:
            continue

        sols = [e.get(solution_key) or [] for e in gen_entries]
        hvs  = [e.get(hv_key, None) for e in gen_entries]
        gen_ids = [e.get("gen_idx", i) for i, e in enumerate(gen_entries)]

        for t in range(1, len(gen_entries)):
            prev_s, cur_s = sols[t-1], sols[t]
            dD = float(front_distance(prev_s, cur_s))

            prev_hv = hvs[t-1]
            cur_hv  = hvs[t]
            dO = 0.0 if (prev_hv is None or cur_hv is None) else float(cur_hv - prev_hv)

            if include_zero_moves or (dD != 0.0 or dO != 0.0):
                all_dD.append(dD)
                all_dO.append(dO)
                all_gen.append(int(gen_ids[t]))
                all_run.append(int(r))

    dD = np.asarray(all_dD, dtype=float)
    dO = np.asarray(all_dO, dtype=float)
    gen = np.asarray(all_gen, dtype=int) if all_gen else np.array([], dtype=int)
    run = np.asarray(all_run, dtype=int) if all_run else np.array([], dtype=int)

    if len(dD) == 0:
        fig.update_layout(title=f"No moves to plot — {series_name}")
        return fig, {"delta_D": dD, "delta_O": dO}

    # Choose coloring
    if color_by is None:
        marker = dict(size=marker_size, opacity=0.75)
        fig.add_trace(go.Scatter(
            x=dD, y=dO, mode="markers",
            marker=marker,
            name="Moves"
        ))
    else:
        if (run_idx is None and color_by == "gen"):
            # pooled + gen doesn't make much sense -> default to run coloring
            color_by = "run"

        c = gen if color_by == "gen" else run
        color_title = "Generation" if color_by == "gen" else "Run"

        fig.add_trace(go.Scatter(
            x=dD, y=dO, mode="markers",
            marker=dict(
                size=marker_size,
                opacity=0.75,
                color=c,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_title)
            ),
            name="Moves"
        ))

    # Add reference lines
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        title=f"Objective vs decision movement — {series_name}" + ("" if run_idx is None else f" (run {run_idx})"),
        xaxis_title="ΔD (decision-space front distance)",
        yaxis_title=f"ΔO (objective change: {hv_key})",
        legend=dict(orientation="h"),
    )

    debug = {"delta_D": dD, "delta_O": dO, "gen": gen, "run": run}
    return fig
