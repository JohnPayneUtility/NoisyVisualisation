"""Multi-objective STN positions (viz.layout.calculate_positions_mo).

A node's position is that of its distinct genotype set: every distinct set among the STN_MO and
STN_MO_Noise nodes is embedded once (front_distance, the selected layout method), in first-appearance order
with primaries before satellites, and every node takes its set's coordinates. Role, hypervolume, generation,
run, series and edges play no part, so nodes with the same set share exactly one position and a change of
HV alone moves a node only along z (checked for every layout method; the edge, satellite and repeated-genotype
checks, which do not depend on the method, skip the slow t-SNE). t-SNE's perplexity is lowered to K - 1 below 31 sets (sklearn requires
perplexity < n_samples), and a single set is placed at the origin.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import TSNE

from noisyvis.common import front_distance, sol_tuple_ints
from noisyvis.common.embedding import compute_distance_matrix
from noisyvis.dashboard.mo_frames import STNFrontState, stn_state_items
from noisyvis.viz import layout
from noisyvis.viz.graph.stn import add_mo_fronts

LAYOUTS = ('mds', 'r_lmds', 'fps_lmds', 'tsne', 'kamada_kawai', 'kamada_kawai_weighted', 'spring', 'raw')


def bits(value):
    return [int(c) for c in format(value, "010b")]


def front(*keys):
    return [bits(k) for k in keys]


def state(solutions, hv):
    return STNFrontState(tuple(range(len(solutions))), solutions, hv, len(solutions))


def add_run(G, sequence, series_idx=0, color='blue'):
    """One run of (gen, primary, satellite) states through the Stage 1 item builder and Stage 2 graph."""
    add_mo_fronts(G, [stn_state_items(sequence)], color, series_idx, 'grey')


def key(G, node):
    return frozenset(sol_tuple_ints(s) for s in G.nodes[node]['front_solutions'])


# Genotype sets: SA, SB, SC primaries; SX, SY satellites; SA also appears as a satellite, and SD only in
# series 1, once with repeated genotypes
SA, SB, SC = front(1, 2, 3), front(40, 41), front(700, 701, 702, 703)
SX, SY, SD = front(5, 900), front(300, 301, 302), front(512, 513)


def mixed_graph():
    """Two series; series 0 has two runs over the same states. Shared satellite (A and B -> X), several
    satellites of one primary (A -> X, Y, and SA itself), an HV-only primary change (SA at HV 1 then 9), a
    satellite with the primary's genotype set, and a node whose front repeats genotypes."""
    G = nx.MultiDiGraph()
    sequence = [(0, state(SA, 1.0), state(SX, 5.0)), (1, state(SB, 2.0), state(SX, 5.0)),
                (2, state(SA, 1.0), state(SY, 6.0)), (3, state(SA, 9.0), state(SY, 6.0)),
                (4, state(SC, 3.0), state(SA, 7.0))]
    add_mo_fronts(G, [stn_state_items(sequence)] * 2, 'blue', 0, 'grey')   # runs 0 and 1
    add_run(G, [(0, state(SD + [SD[0], SD[0]], 4.0), state(SB, 8.0)), (5, state(SD, 4.5), state(SB, 8.0))],
            series_idx=1, color='red')
    return G


def expected_items(G):
    """The distinct sets in first-appearance order, primaries before satellites, each once and deduplicated."""
    seen, items = set(), []
    for node_type in ('STN_MO', 'STN_MO_Noise'):
        for n, d in G.nodes(data=True):
            if d.get('type') == node_type and key(G, n) not in seen:
                seen.add(key(G, n))
                distinct = []
                for s in d['front_solutions']:
                    if sol_tuple_ints(s) not in {sol_tuple_ints(t) for t in distinct}:
                        distinct.append(s)
                items.append(distinct)
    return items


@pytest.fixture
def embedded(monkeypatch):
    """Records the items every embedding receives (the matrix paths and landmark MDS)."""
    seen = []
    real_matrix, real_lmds = layout.compute_distance_matrix, layout.landmark_mds

    def matrix(distance_fn, items):
        seen.append([list(map(sol_tuple_ints, f)) for f in items])
        return real_matrix(distance_fn, items)

    def lmds(distance_fn, items, **kwargs):
        seen.append([list(map(sol_tuple_ints, f)) for f in items])
        return real_lmds(distance_fn, items, **kwargs)

    monkeypatch.setattr(layout, 'compute_distance_matrix', matrix)
    monkeypatch.setattr(layout, 'landmark_mds', lmds)
    return seen


def reference_positions(items, layout_type):
    """The positions of a graph holding one node per set, in the same order: the embedding of exactly those
    items, read back per set."""
    R = nx.MultiDiGraph()
    for i, f in enumerate(items):
        R.add_node(f"set{i}", type='STN_MO', front_solutions=f)
    pos = layout.calculate_positions_mo(R, layout_type)
    return [pos[f"set{i}"] for i in range(len(items))]


# ------------------------------------------------------------------------------ T11

@pytest.mark.parametrize("layout_type", LAYOUTS)
def test_equal_genotype_sets_share_exactly_one_position(layout_type, embedded):
    """A, B: every node takes the coordinates its own set receives from one embedding of the distinct sets,
    so nodes with the same set coincide exactly across role, HV, generation, run and series."""
    G = mixed_graph()
    pos = layout.calculate_positions_mo(G, layout_type)
    items = expected_items(G)

    # one embedding input per distinct set, first appearance order, no repeated genotypes
    assert [[list(map(sol_tuple_ints, f)) for f in items]] == embedded[:1]
    assert len(items) == len({key(G, n) for n in G.nodes}) == 6
    assert all(len(set(map(sol_tuple_ints, f))) == len(f) for f in items)

    # each node reads its own set's coordinates
    index = {frozenset(map(sol_tuple_ints, f)): i for i, f in enumerate(items)}
    reference = reference_positions(items, layout_type)
    assert set(pos) == set(G.nodes)
    for n in G.nodes:
        assert pos[n] == reference[index[key(G, n)]], n
        assert all(math.isfinite(c) for c in pos[n])

    # equality classes: SA appears as primary (HV 1 and HV 9) and as a satellite, in runs 0 and 1
    sa = [n for n in G.nodes if key(G, n) == frozenset(map(sol_tuple_ints, SA))]
    assert {G.nodes[n]['type'] for n in sa} == {'STN_MO', 'STN_MO_Noise'} and len(sa) == 6
    assert {G.nodes[n]['run_idx'] for n in sa} == {0, 1} and {G.nodes[n]['fitness'] for n in sa} == {1.0, 9.0, 7.0}
    assert len({pos[n] for n in sa}) == 1
    # SB is a primary in series 0 and a satellite in series 1
    sb = [n for n in G.nodes if key(G, n) == frozenset(map(sol_tuple_ints, SB))]
    assert {n.split('_')[1] for n in sb} == {'S0', 'S1'} and len({pos[n] for n in sb}) == 1


@pytest.mark.parametrize("layout_type", ('mds', 'r_lmds', 'kamada_kawai'))
def test_satellites_take_their_own_sets_coordinates(layout_type):
    """C, D: one satellite shared by two primaries has one position, its own set's; several satellites of
    one primary each have their own set's."""
    G = mixed_graph()
    pos = layout.calculate_positions_mo(G, layout_type)
    items = expected_items(G)
    reference = dict(zip((frozenset(map(sol_tuple_ints, f)) for f in items), reference_positions(items, layout_type)))

    shared = 'MO_S0_R0_Q0'                                   # SX, linked from A and B
    assert sorted(u for u, _, d in G.in_edges(shared, data=True) if d['edge_type'] == 'Noise_MO') == \
        ['MO_S0_R0_P0', 'MO_S0_R0_P1']
    assert pos[shared] == reference[key(G, shared)]
    satellites = sorted(v for _, v, d in G.out_edges('MO_S0_R0_P0', data=True) if d['edge_type'] == 'Noise_MO')
    assert len(satellites) == 2
    for s in satellites:
        assert pos[s] == reference[key(G, s)]


def _rebuilt(G, edges):
    """G's nodes in the same insertion order with the given edges."""
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in edges:
        H.add_edge(u, v, **d)
    return H


@pytest.mark.parametrize("layout_type", ('mds', 'fps_lmds', 'kamada_kawai'))
def test_edges_do_not_move_nodes(layout_type):
    """E: with the nodes and their insertion order fixed, removing, reordering or adding satellite and
    temporal edges changes no position."""
    G = mixed_graph()
    edges = list(G.edges(data=True))
    noise = [e for e in edges if e[2]['edge_type'] == 'Noise_MO']
    temporal = [e for e in edges if e[2]['edge_type'] == 'STN_MO']
    extra = [('MO_S0_R0_P2', 'MO_S0_R0_Q0', {'edge_type': 'Noise_MO'}),
             ('MO_S0_R1_P2', 'MO_S0_R0_P0', {'edge_type': 'STN_MO'})]
    pos = layout.calculate_positions_mo(G, layout_type)
    for variant in ([], temporal, noise, edges[::-1], edges + extra):
        assert layout.calculate_positions_mo(_rebuilt(G, variant), layout_type) == pos


@pytest.mark.parametrize("layout_type", ('mds', 'r_lmds', 'kamada_kawai'))
def test_repeated_genotypes_change_nothing(layout_type, embedded):
    """F: a front listing a genotype several times is the same set as the distinct list, contributes one
    embedding item without repeats, and leaves every position as the distinct graph gives it."""
    G = mixed_graph()
    pos = layout.calculate_positions_mo(G, layout_type)
    repeated = 'MO_S1_R0_P0'
    assert len(G.nodes[repeated]['front_solutions']) == 4 and len(key(G, repeated)) == 2
    assert pos[repeated] == pos['MO_S1_R0_P1']                 # SD at HV 4.5, listed without repeats

    distinct = _rebuilt(G, G.edges(data=True))
    distinct.nodes[repeated]['front_solutions'] = SD
    assert layout.calculate_positions_mo(distinct, layout_type) == pos
    assert embedded[0] == embedded[-1] and embedded[0].count([sol_tuple_ints(s) for s in SD]) == 1
    # the stored HVs are untouched
    assert (G.nodes[repeated]['fitness'], G.nodes['MO_S1_R0_P1']['fitness']) == (4.0, 4.5)


@pytest.mark.parametrize("layout_type", LAYOUTS)
def test_hv_only_states_differ_only_along_z(layout_type):
    """G: two states with the same genotypes and different HVs are separate nodes at one XY; their 3D points
    (x, y, fitness as z) differ only in z."""
    G = mixed_graph()
    pos = layout.calculate_positions_mo(G, layout_type)
    low, high = 'MO_S0_R0_P0', 'MO_S0_R0_P2'                   # SA at HV 1, then HV 9 at g3
    assert G.nodes[low]['front_solutions'] == G.nodes[high]['front_solutions'] == SA
    assert (*pos[low], G.nodes[low]['fitness']) == (*pos[high], 1.0) and G.nodes[high]['fitness'] == 9.0
    assert G.has_edge(low, high)                               # the vertical transition


def test_non_mo_nodes_are_not_positioned_here():
    G = mixed_graph()
    G.add_node('Local Optimum 1', solution=bits(3), fitness=1.0)
    G.add_node('guide_0', type='guide', solution=bits(4))
    pos = layout.calculate_positions_mo(G, 'mds')
    assert 'Local Optimum 1' not in pos and 'guide_0' not in pos


# ------------------------------------------------------------------------------ T13: adaptive t-SNE

def k_set_graph(k):
    """k distinct sets over 2k nodes: each set once as a primary and once as a satellite at another HV."""
    G = nx.MultiDiGraph()
    if k:
        add_run(G, [(g, state(front(3 * g, 3 * g + 1), float(g)), state(front(3 * g, 3 * g + 1), g + 0.5))
                    for g in range(k)])
    return G


@pytest.fixture
def tsne_calls(monkeypatch):
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return TSNE(**kwargs)

    monkeypatch.setattr(layout, 'TSNE', spy)
    return calls


@pytest.mark.parametrize("k", [0, 1, 2, 10, 30, 31])
def test_tsne_perplexity_adapts_to_the_number_of_sets(k, tsne_calls):
    G = k_set_graph(k)
    assert len({key(G, n) for n in G.nodes}) == k and len(G) == 2 * k
    pos = layout.calculate_positions_mo(G, 'tsne')
    assert set(pos) == set(G.nodes)
    if k == 0:
        assert pos == {} and tsne_calls == []
    elif k == 1:
        assert set(pos.values()) == {(0.0, 0.0)} and tsne_calls == []
    else:
        [call] = tsne_calls
        assert call == {'n_components': 2, 'metric': 'precomputed', 'random_state': 42, 'init': 'random',
                        'perplexity': float(min(30, k - 1))}
        assert all(math.isfinite(c) for xy in pos.values() for c in xy)
    # the primary and the satellite of each set share its position
    for g in range(k):
        assert pos[f"MO_S0_R0_P{g}"] == pos[f"MO_S0_R0_Q{g}"]


def test_tsne_above_30_sets_is_the_previous_configuration():
    """31 sets: perplexity 30 is sklearn's default, so the result is bitwise the previous call's."""
    G = k_set_graph(31)
    items = expected_items(G)
    D = compute_distance_matrix(front_distance, items)
    previous = TSNE(n_components=2, metric='precomputed', random_state=42, init='random').fit_transform(D)
    pos = layout.calculate_positions_mo(G, 'tsne')
    index = {frozenset(map(sol_tuple_ints, f)): i for i, f in enumerate(items)}
    for n in G.nodes:
        assert pos[n] == (float(previous[index[key(G, n)], 0]), float(previous[index[key(G, n)], 1]))


@pytest.mark.parametrize("k", [2, 10, 31])
def test_mds_and_kamada_kawai_are_unchanged(k):
    """The same embedding of the distinct sets as a direct call of the unchanged methods."""
    G = k_set_graph(k)
    items = expected_items(G)
    index = {frozenset(map(sol_tuple_ints, f)): i for i, f in enumerate(items)}
    D = compute_distance_matrix(front_distance, items)
    mds = MDS_sklearn(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(D)
    H = nx.complete_graph(k)
    for i in range(k):
        for j in range(i + 1, k):
            H[i][j]['weight'] = max(D[i, j], 1e-6)
    raw = nx.kamada_kawai_layout(H, weight='weight', dim=2)
    for layout_type, XY in (('mds', mds), ('kamada_kawai', np.array([raw[i] for i in range(k)]))):
        pos = layout.calculate_positions_mo(G, layout_type)
        for n in G.nodes:
            i = index[key(G, n)]
            assert pos[n] == (float(XY[i, 0]), float(XY[i, 1])), (layout_type, n)
