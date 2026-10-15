"""Node sizes of the multiobjective STN plot.

Each multiobjective STN step adds a first-front node (type STN_MO) and, when the mode shows a second front, a
second-front node (type STN_MO_Noise). Both are sized by front size on one shared scale within
[stn_min, stn_max]. Previously the range came from the first-front nodes alone, so a second front smaller
than every first front got a negative size, and Plotly refused to build the trace ('both pareto (algo pov)':
the second front is the clean front). Other node types are sized as before.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import networkx as nx
import pytest

from noisyvis.viz.config import PlotConfig
from noisyvis.viz.graph.stn import add_mo_fronts
from noisyvis.viz.styling import apply_node_sizes
from noisyvis.viz.traces import create_node_traces

STN_MIN, STN_MAX = 5.0, 15.0


def config():
    cfg = PlotConfig()
    cfg.stn_plot_type = 'multiobjective'
    cfg.node_size.stn_min, cfg.node_size.stn_max = STN_MIN, STN_MAX
    return cfg


def front(n):
    return [[i % 2, (i // 2) % 2, i] for i in range(n)]


def mo_graph(first_sizes, second_sizes=None):
    """One run of STN entries: first fronts of the given sizes, optionally a second front at every step."""
    steps = []
    for g, n1 in enumerate(first_sizes):
        n2 = None if second_sizes is None else second_sizes[g]
        steps.append({'front1': front(n1), 'front2': None if n2 is None else front(n2),
                      'metric1': float(g), 'metric2': None if n2 is None else float(g) + 0.5, 'gen_idx': g})
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [steps], 'blue', 0)
    return G


def mo_sizes(G):
    return {n: (d['type'], d['front_size'], d['size']) for n, d in G.nodes(data=True) if 'STN_MO' in d['type']}


def positions(G):
    return {n: (float(i), 0.0) for i, n in enumerate(G.nodes())}


@pytest.mark.parametrize("first,second", [
    ([15, 20, 27], [8, 12, 24]),     # algo POV: a second front smaller than every first front
    ([3, 4, 5], [9, 12, 14]),        # both pareto: second fronts larger than every first front
])
def test_every_mo_node_stays_within_the_configured_range(first, second):
    G = mo_graph(first, second)
    apply_node_sizes(G, config())
    sizes = mo_sizes(G)
    assert {t for t, _, _ in sizes.values()} == {'STN_MO', 'STN_MO_Noise'}
    assert all(STN_MIN <= s <= STN_MAX for _, _, s in sizes.values()), sizes
    lon, stn, noisy = create_node_traces(G, positions(G), config())     # Plotly validates sizes here
    assert len(noisy.marker.size) == len(second) and min(noisy.marker.size) >= STN_MIN


def test_one_shared_scale_keeps_relative_sizes():
    """Front size maps linearly onto [stn_min, stn_max] across both node types: the smallest and largest
    fronts (here a second front and a first front) take the ends, and sizes increase with front size."""
    G = mo_graph([15, 20, 27], [8, 12, 24])
    apply_node_sizes(G, config())
    by_front = sorted((f, s) for _, f, s in mo_sizes(G).values())
    assert by_front[0] == (8, STN_MIN) and by_front[-1] == (27, STN_MAX)
    assert [s for _, s in by_front] == sorted(s for _, s in by_front)
    for f, s in by_front:
        assert s == pytest.approx(STN_MIN + (f - 8) / (27 - 8) * (STN_MAX - STN_MIN))


def test_equal_front_sizes_take_the_midpoint():
    G = mo_graph([6, 6], [6, 6])
    apply_node_sizes(G, config())
    assert {s for _, _, s in mo_sizes(G).values()} == {STN_MIN + 0.5 * (STN_MAX - STN_MIN)}


def test_first_fronts_alone_are_sized_as_before():
    """Without second-front nodes the range is the first fronts' range, exactly as before."""
    G = mo_graph([10, 14, 20])
    apply_node_sizes(G, config())
    assert sorted(s for _, _, s in mo_sizes(G).values()) == [STN_MIN, STN_MIN + 0.4 * (STN_MAX - STN_MIN), STN_MAX]


def test_other_node_types_are_sized_as_before():
    """Single-objective STN, STN_ALT, LON and other nodes keep their own rules, unaffected by MO nodes."""
    G = mo_graph([15, 20], [8, 30])
    G.add_node('STN_S0_R0_a', type='STN', iterations=2)
    G.add_node('STN_S0_R0_b', type='STN', iterations=6)
    G.add_node('alt_node', type='STN_ALT')
    G.add_node('Local Optimum 1')
    G.add_node('Local Optimum 2')
    G.add_edge('Local Optimum 1', 'Local Optimum 2', weight=0.25)
    G.add_node('guide', type='guide')
    cfg = config()
    apply_node_sizes(G, cfg)
    size = {n: d['size'] for n, d in G.nodes(data=True)}
    assert (size['STN_S0_R0_a'], size['STN_S0_R0_b']) == (STN_MIN, STN_MAX)
    assert size['alt_node'] == STN_MIN
    lon_min, lon_max = cfg.node_size.lon_min, cfg.node_size.lon_max
    assert (size['Local Optimum 1'], size['Local Optimum 2']) == (lon_min, lon_min + 0.25 * (lon_max - lon_min))
    assert size['guide'] == 1
