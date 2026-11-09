"""
Pareto rank layers plot.

This module draws one run's candidate pool (its population or a passive archive, at one generation or
combined across generations) in objective space, coloured by each member's non-dominated rank
(dashboard.mo_frames.rank_layer_data prepares the layers).
"""

import math

import plotly.graph_objects as go

from ..base import (
    DEFAULT_TEMPLATE,
    DEFAULT_MARKER_SIZE,
    create_empty_figure,
    front_line,
    generation_color,
)

# Rank 0 has its own colour, outside the colourscale of the other ranks
RANK0_COLOR = '#d62728'
RANK_COLORSCALE = 'Viridis'
RANK0_MARKER_SIZE = 11
MAX_MARKER_SIZE = 22
# With more ranks than this, only the first ones get legend entries and a colourbar shows the rest
LEGEND_MAX_RANKS = 10
# Identity lists in the hover are cut after this many ids (the counts stay exact)
HOVER_ID_LIMIT = 10

SPACE_LABELS = {'noisy': 'noisy y', 'true': 'true f(x)'}


def rank_color(rank, n_ranks):
    """The colour of a rank: rank 0's own colour, else ranks 1..n_ranks-1 along the colourscale (the
    midpoint when rank 1 is the only one). Fixed by the whole population's rank count, so showing fewer
    ranks never recolours one."""
    if rank == 0:
        return RANK0_COLOR
    return generation_color(rank, 1, max(n_ranks - 1, 1), RANK_COLORSCALE)


def _marker_size(rank, count):
    base = RANK0_MARKER_SIZE if rank == 0 else DEFAULT_MARKER_SIZE
    return min(base + 2 * math.sqrt(count - 1), MAX_MARKER_SIZE)


def _plural(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _number(value):
    return f"{float(value):.10g}"


def _vector(values):
    return "(" + ", ".join(_number(v) for v in values) + ")"


def _ids(ids):
    shown = ", ".join(str(i) for i in ids[:HOVER_ID_LIMIT])
    more = len(ids) - HOVER_ID_LIMIT
    return shown + (f" … and {more} more" if more > 0 else "")


def _hover(point, basis, pool='population'):
    """One point's hover text: its rank, both spaces' values, the other basis's ranks and its members (and,
    for an archive's members, when they entered it and how many remain in the final archive)."""
    other = 'true' if basis == 'noisy' else 'noisy'
    others = "; ".join(f"{_vector(v)} ×{n}" for v, n in point['other_values'])
    other_ranks = ", ".join(f"{r} ×{n}" for r, n in point['other_ranks'])
    labels = dict(SPACE_LABELS)
    if pool == 'clean_archive':  # a clean-archive genotype's y is that of its entry evaluation
        labels['noisy'] = f"{SPACE_LABELS['noisy']} (entry evaluation)"
    lines = [
        f"<b>Rank {point['rank']}</b> ({basis} objectives)",
        f"{labels[basis]}: {_vector((point['x'], point['y']))}",
        f"{labels[other]}: {others}",
        f"{other} ranks: {other_ranks}",
        f"×{_plural(point['count'], 'member')} · {_plural(len(set(point['eval_ids'])), 'evaluation')} · "
        f"{_plural(len(set(point['geno_ids'])), 'genotype')} · {_plural(len(set(point['obs_ids'])), 'observation')}",
        f"eval ids: {_ids(point['eval_ids'])}",
        f"genotype ids: {_ids(point['geno_ids'])}",
        f"observation ids: {_ids(point['obs_ids'])}",
    ]
    if 'entry_generations' in point:
        lines.append(f"entered the archive at gen: {_ids(sorted(set(point['entry_generations'])))}")
    if 'in_final' in point:
        lines.append(f"in the final archive: {point['in_final']} of {point['count']}")
    return "<br>".join(lines)


def _axis_range(extent):
    low, high = extent
    pad = (high - low) * 0.05 or 0.5
    return [low - pad, high + pad]


def plot_rank_layers(layers, series_label=None, staircase=False, uirevision=None):
    """
    Plot a candidate pool's rank layers: one marker trace per shown rank, coloured by rank_color.

    Unlike the front plots, the data is one pool (a population or a passive archive, at one generation or
    combined across generations), not a run's front entries.

    Args:
        layers: dashboard.mo_frames.rank_layer_data (or {'unavailable': message})
        series_label: the run's label, for the title
        staircase: If True, also join each shown rank by its attainment staircase
        uirevision: kept zoom and legend state while it stays the same (e.g. the run and basis)

    Returns:
        go.Figure: Plotly figure with the rank layers
    """
    if 'unavailable' in layers:
        return create_empty_figure(layers['unavailable'])
    gen, basis = layers['generation'], layers['basis']
    pool, combined, source = layers.get('pool', 'population'), layers.get('combined', False), layers.get('pool_label')
    if not layers['points']:
        if combined:
            return create_empty_figure(f"The combined {source} is empty")
        return create_empty_figure(f"The {source or 'population'} at generation {gen} is empty")

    n_ranks, weights = layers['n_ranks'], layers['weights']
    by_rank = {}
    for point in layers['points']:
        by_rank.setdefault(point['rank'], []).append(point)

    fig = go.Figure()
    if staircase:
        for rank, points in sorted(by_rank.items(), reverse=True):
            if len(points) < 2:
                continue
            xs, ys, shape = front_line([(p['x'], p['y']) for p in points], weights, attainment=True)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines', line=dict(color=rank_color(rank, n_ranks), width=1), line_shape=shape,
                opacity=0.6, hoverinfo='skip', showlegend=False, legendgroup=f"rank{rank}"))

    # highest rank first, so rank 0 is drawn last (on top); legendrank keeps the legend in rank order
    for rank, points in sorted(by_rank.items(), reverse=True):
        members = sum(p['count'] for p in points)
        fig.add_trace(go.Scatter(
            x=[p['x'] for p in points], y=[p['y'] for p in points], mode='markers',
            name=f"Rank {rank} ({_plural(members, 'member')})", legendgroup=f"rank{rank}", legendrank=rank,
            showlegend=n_ranks <= LEGEND_MAX_RANKS or rank < LEGEND_MAX_RANKS,
            marker=dict(color=rank_color(rank, n_ranks), size=[_marker_size(rank, p['count']) for p in points],
                        opacity=1.0 if rank == 0 else 0.85,
                        line=dict(color='white', width=1.0 if rank == 0 else 0.5)),
            text=[_hover(p, basis, pool) for p in points], hovertemplate='%{text}<extra></extra>'))

    if n_ranks > LEGEND_MAX_RANKS:
        # a colourbar for ranks 1..n_ranks-1, on a trace with no points
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', showlegend=False, hoverinfo='skip',
            marker=dict(color=[1], colorscale=RANK_COLORSCALE, cmin=1, cmax=n_ranks - 1, showscale=True,
                        colorbar=dict(title='Rank', thickness=15, len=0.7))))

    space = SPACE_LABELS[basis]
    axis_titles = [f"Objective {i + 1} ({space}, {'maximised' if w > 0 else 'minimised'})"
                   for i, w in enumerate(weights)]
    shown = f"{layers['n_shown']} of {layers['n_members']} members shown"
    ranks = f"ranks 0–{layers['max_rank_shown']} of {n_ranks}"
    if combined:
        final = f" · {layers['n_final']} in the final archive" if 'n_final' in layers else ""
        status = f"{shown} · {layers['n_genotypes']} genotypes{final} · {ranks}"
        scope = f", {source}, all generations combined"
    else:
        start, end = layers['step_range']
        unchanged = f"gen {start}" if start == end else f"gens {start}–{end}"
        what = 'population' if source is None else 'archive'
        status = f"Step {layers['step'] + 1} of {layers['n_steps']} · {what} unchanged over {unchanged} · {shown}"
        status += f" · {ranks}" if source is None else f" · {layers['n_genotypes']} genotypes · {ranks}"
        scope = f", Gen {gen}" if source is None else f", {source}, Gen {gen}"
    run = f" - {series_label}" if series_label else ""
    fig.update_layout(
        title=f"Pareto rank layers ({basis} objectives){run}{scope}<br><sup>{status}</sup>",
        xaxis_title=axis_titles[0],
        yaxis_title=axis_titles[1],
        xaxis=dict(range=_axis_range(layers['axis_extent'][0])),
        yaxis=dict(range=_axis_range(layers['axis_extent'][1])),
        template=DEFAULT_TEMPLATE,
        legend_title=f"Rank ({basis})",
        uirevision=uirevision,
        height=600,
    )
    return fig
