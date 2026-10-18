"""Node sizes of the multiobjective STN plot.

Each multiobjective STN state adds a first-front node (type STN_MO) and, when the view shows a second front,
a second-front node (type STN_MO_Noise). Both are sized by front size on one shared scale within
[stn_min, stn_max]. Previously the range came from the first-front nodes alone, so a second front smaller
than every first front got a negative size, and Plotly refused to build the trace ('both pareto (algo pov)':
the second front is the clean front). Other node types are sized as before.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import networkx as nx
import pytest

from noisyvis.dashboard.mo_frames import STNFrontState, stn_state_items
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


def state(n, hv):
    return STNFrontState(tuple(range(n)), front(n), hv, n)


def mo_graph(first_sizes, second_sizes=None):
    """One run of front states: first fronts of the given sizes, optionally a second front at every step (each
    step a new state of both)."""
    steps = []
    for g, n1 in enumerate(first_sizes):
        n2 = None if second_sizes is None else second_sizes[g]
        steps.append((g, state(n1, float(g)), None if n2 is None else state(n2, float(g) + 0.5)))
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [stn_state_items(steps)], 'blue', 0)
    return G


def mo_sizes(G):
    return {n: (d['type'], d['front_size'], d['size']) for n, d in G.nodes(data=True) if 'STN_MO' in d['type']}


def positions(G):
    return {n: (float(i), 0.0) for i, n in enumerate(G.nodes())}


@pytest.mark.parametrize("first,second", [
    ([15, 20, 27], [8, 12, 24]),     # algo POV: a second front smaller than every first front
    ([3, 4, 5], [9, 12, 14]),        # second fronts larger than every first front (e.g. True POV)
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


# ------------------------------------------------------------------------------ node size basis

def counted_graph():
    """g0: a noisy primary of 2 unique solutions listed 6 times (N > U) with a clean satellite of 3 (N = U);
    g1: a primary of 4 unique solutions (N = U) with the same satellite."""
    satellite = STNFrontState((0, 1, 2), front(3), 2.0, 3)
    steps = [(0, STNFrontState((0, 1), front(2), 1.0, 6), satellite),
             (1, STNFrontState((0, 1, 2, 3), front(4), 1.5, 4), satellite)]
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [stn_state_items(steps)], 'blue', 0)
    return G


def sized(G, mode):
    cfg = config()
    cfg.mo_node_size = mode
    apply_node_sizes(G, cfg)
    return {n: d['size'] for n, d in G.nodes(data=True)}


def test_all_solutions_is_the_default_basis():
    assert PlotConfig().mo_node_size == 'all'
    G = counted_graph()
    apply_node_sizes(G, config())                        # the mode left at its default
    default = {n: d['size'] for n, d in G.nodes(data=True)}
    assert default == sized(counted_graph(), 'all') != sized(counted_graph(), 'unique')


@pytest.mark.parametrize("mode,expected", [
    # all: counts with repeats 6, 4, 3 -> range 3..6, over primaries and the satellite together
    ('all', {'MO_S0_R0_P0': STN_MAX, 'MO_S0_R0_P1': STN_MIN + (4 - 3) / (6 - 3) * (STN_MAX - STN_MIN),
             'MO_S0_R0_Q0': STN_MIN}),
    # unique: 2, 4, 3 -> range 2..4
    ('unique', {'MO_S0_R0_P0': STN_MIN, 'MO_S0_R0_P1': STN_MAX,
                'MO_S0_R0_Q0': STN_MIN + (3 - 2) / (4 - 2) * (STN_MAX - STN_MIN)}),
])
def test_node_size_follows_the_selected_count_on_one_shared_range(mode, expected):
    G = counted_graph()
    assert sized(G, mode) == pytest.approx(expected)
    lon, stn, noisy = create_node_traces(G, positions(G), config())     # Plotly validates sizes here
    assert sorted(list(stn.marker.size) + list(noisy.marker.size)) == pytest.approx(sorted(expected.values()))


def test_an_unchanged_satellite_rescales_with_the_range():
    """The satellite's counts (N = U = 3) are the same in both modes, but the shared range is not, so its
    displayed size differs: the expected behaviour."""
    G = counted_graph()
    assert (G.nodes['MO_S0_R0_Q0']['n_observations'], G.nodes['MO_S0_R0_Q0']['front_size']) == (3, 3)
    assert sized(counted_graph(), 'all')['MO_S0_R0_Q0'] != sized(counted_graph(), 'unique')['MO_S0_R0_Q0']


@pytest.mark.parametrize("mode", ['all', 'unique'])
def test_other_node_types_ignore_the_basis(mode):
    G = counted_graph()
    G.add_node('STN_S0_R0_a', type='STN', iterations=2)
    G.add_node('STN_S0_R0_b', type='STN', iterations=6)
    G.add_node('Local Optimum 1')
    size = sized(G, mode)
    assert (size['STN_S0_R0_a'], size['STN_S0_R0_b'], size['Local Optimum 1']) == (STN_MIN, STN_MAX, config().node_size.lon_min)
