"""Multi-objective STN graph construction from front states (viz.graph.stn.add_mo_fronts).

Items from dashboard.mo_frames (stn_state_items / stn_view_entries) carry per-role state ids. A node is a
front state, reused within its run whenever its id recurs: primary `MO_S{s}_R{r}_P{state1}` (STN_MO),
satellite `MO_S{s}_R{r}_Q{state2}` (STN_MO_Noise). STN_MO edges join consecutive different primary states;
Noise_MO edges join each primary to the satellite states paired with it. Each edge is added once, and a
repeat records its generation. Positions are checked in test_mo_stn_positions.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import math
import re

import networkx as nx
import pytest

from noisyvis.dashboard.mo_frames import STNFrontState, stn_state_items, stn_view_entries
from noisyvis.viz.config import PlotConfig
from noisyvis.viz.graph.stn import add_mo_fronts
from noisyvis.viz.styling import apply_node_sizes
from noisyvis.viz.traces import create_node_traces
from test_mo_stn_views import crafted


def state(genotypes, hv, n=None):
    genotypes = tuple(genotypes)
    return STNFrontState(genotypes, [[g, 0] for g in genotypes], hv, len(genotypes) if n is None else n)


def build(sequence, series_idx=0):
    """The graph of one run from (gen, primary, satellite) states, through the real Stage 1 item builder."""
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [stn_state_items(sequence)], 'blue', series_idx, 'grey')
    return G


def P(k, s=0, r=0):
    return f"MO_S{s}_R{r}_P{k}"


def Q(k, s=0, r=0):
    return f"MO_S{s}_R{r}_Q{k}"


def edges(G, edge_type):
    """{(u, v): generations} for one edge type; every pair must appear once (no parallel edges)."""
    found = [(u, v, d['generations']) for u, v, d in G.edges(data=True) if d['edge_type'] == edge_type]
    pairs = {(u, v): gens for u, v, gens in found}
    assert len(pairs) == len(found), f"parallel {edge_type} edges: {found}"
    return pairs


def nodes(G, node_type):
    return {n: (d['generations'], d['visits'], d['gen_idx']) for n, d in G.nodes(data=True) if d['type'] == node_type}


A, B, C, S = state((1, 2), 1.0), state((1, 3), 2.0), state((4,), 3.0), state((1, 2), 4.0)
X, Y, Z = state((7,), 10.0), state((8,), 11.0), state((9,), 12.0)


# ------------------------------------------------------------------------------ primary states

def test_primary_revisit_reuses_its_node():
    """g0 A, g1 A, g2 B, g3 B, g4 C, g5 A: three nodes; A -> B, B -> C, C -> A."""
    G = build([(0, A, None), (1, A, None), (2, B, None), (3, B, None), (4, C, None), (5, A, None)])
    assert nodes(G, 'STN_MO') == {P(0): ([0, 5], 2, 0), P(1): ([2], 1, 2), P(2): ([4], 1, 4)}
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [2], (P(1), P(2)): [4], (P(2), P(0)): [5]}
    assert not nodes(G, 'STN_MO_Noise') and not edges(G, 'Noise_MO')


def test_repeated_transitions_keep_one_directed_edge_each():
    """A -> B -> A -> B: two nodes, A -> B (both occurrences recorded) and B -> A, no self-loops."""
    G = build([(0, A, None), (1, B, None), (2, A, None), (3, B, None)])
    assert set(G.nodes) == {P(0), P(1)}
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [1, 3], (P(1), P(0)): [2]}
    assert G.number_of_edges() == 2 and not list(nx.selfloop_edges(G))
    assert nodes(G, 'STN_MO') == {P(0): ([0, 2], 2, 0), P(1): ([1, 3], 2, 1)}


def test_same_genotypes_with_another_hv_are_another_state():
    G = build([(0, A, None), (1, S, None)])
    assert G.nodes[P(0)]['front_solutions'] == G.nodes[P(1)]['front_solutions']
    assert (G.nodes[P(0)]['fitness'], G.nodes[P(1)]['fitness']) == (1.0, 4.0)
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [1]}


# ------------------------------------------------------------------------------ satellites

def test_case_a_primary_changes_and_shares_the_satellite():
    G = build([(0, A, X), (1, B, X)])
    assert set(nodes(G, 'STN_MO')) == {P(0), P(1)} and nodes(G, 'STN_MO_Noise') == {Q(0): ([0], 1, 0)}
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [1]}
    assert edges(G, 'Noise_MO') == {(P(0), Q(0)): [0], (P(1), Q(0)): [1]}


def test_case_b_satellite_changes_and_keeps_the_primary():
    G = build([(0, A, X), (1, A, Y), (2, B, Y)])
    assert nodes(G, 'STN_MO') == {P(0): ([0], 1, 0), P(1): ([2], 1, 2)}
    assert nodes(G, 'STN_MO_Noise') == {Q(0): ([0], 1, 0), Q(1): ([1], 1, 1)}
    # the satellite-only change neither adds a temporal edge nor interrupts the primary: A -> B at g2
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [2]}
    assert edges(G, 'Noise_MO') == {(P(0), Q(0)): [0], (P(0), Q(1)): [1], (P(1), Q(1)): [2]}


def test_case_c_both_change_and_reuse_earlier_states():
    G = build([(0, A, X), (1, B, Y), (2, A, X)])
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [1], (P(1), P(0)): [2]}
    assert edges(G, 'Noise_MO') == {(P(0), Q(0)): [0, 2], (P(1), Q(1)): [1]}
    assert G.number_of_nodes() == 4


def test_visits_count_state_entries_per_role():
    """g10 A/X, g11 A/Y, g12 A/Z, g13 B/Z, g14 A/Z: A is entered twice (satellite-only changes do not
    re-enter it); Z is entered once (it stays active while the primary moves)."""
    G = build([(10, A, X), (11, A, Y), (12, A, Z), (13, B, Z), (14, A, Z)])
    assert nodes(G, 'STN_MO') == {P(0): ([10, 14], 2, 10), P(1): ([13], 1, 13)}
    assert nodes(G, 'STN_MO_Noise') == {Q(0): ([10], 1, 10), Q(1): ([11], 1, 11), Q(2): ([12], 1, 12)}
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [13], (P(1), P(0)): [14]}
    assert edges(G, 'Noise_MO') == {(P(0), Q(0)): [10], (P(0), Q(1)): [11], (P(0), Q(2)): [12, 14],
                                    (P(1), Q(2)): [13]}


def test_a_returning_satellite_is_a_visit():
    G = build([(0, A, X), (1, A, Y), (2, A, X)])
    assert nodes(G, 'STN_MO_Noise') == {Q(0): ([0, 2], 2, 0), Q(1): ([1], 1, 1)}
    assert edges(G, 'Noise_MO') == {(P(0), Q(0)): [0, 2], (P(0), Q(1)): [1]}
    assert nodes(G, 'STN_MO') == {P(0): ([0], 1, 0)}


# ------------------------------------------------------------------------------ attributes

def test_node_attributes():
    G = build([(3, state((1, 2, 5), 1.5, n=7), state((1, 9), 2.5, n=2))])
    primary, satellite = G.nodes[P(0)], G.nodes[Q(0)]
    assert primary == {'type': 'STN_MO', 'is_noisy': False, 'front_solutions': [[1, 0], [2, 0], [5, 0]],
                       'front_size': 3, 'hypervolume': 1.5, 'fitness': 1.5, 'run_idx': 0, 'gen_idx': 3,
                       'color': 'blue', 'n_observations': 7, 'first_evals': None, 'generations': [3], 'visits': 1}
    assert (satellite['type'], satellite['is_noisy'], satellite['front_size'], satellite['n_observations']) == \
        ('STN_MO_Noise', True, 2, 2)
    assert satellite['fitness'] == satellite['hypervolume'] == 2.5
    edge = G.get_edge_data(P(0), Q(0))[0]
    assert edge == {'weight': 0.5, 'color': 'grey', 'edge_type': 'Noise_MO', 'is_noisy': True, 'generations': [3]}


def test_observation_count_is_kept_from_the_first_appearance():
    """The observation count is not part of a state, so a recurring state with another count is the same
    node, which keeps its first-appearance count."""
    G = build([(0, state((1, 2), 1.0, n=4), None), (1, B, None), (2, state((1, 2), 1.0, n=9), None)])
    assert set(G.nodes) == {P(0), P(1)} and G.nodes[P(0)]['n_observations'] == 4


def with_evals(items, evals):
    """Items as stn_view_entries gives them: each carrying the evaluations completed by its generation."""
    return [{**item, 'evals': evals[item['gen_idx']]} for item in items]


def test_first_evals_is_kept_from_the_first_appearance():
    """A -> B -> A with X, Y, X: every node's first_evals (and gen_idx) is that of its state's first item."""
    G = nx.MultiDiGraph()
    items = with_evals(stn_state_items([(0, A, X), (2, B, Y), (5, A, X)]), {0: 12, 2: 30, 5: 61})
    add_mo_fronts(G, [items], 'blue', 0)
    assert {n: (d['gen_idx'], d['first_evals']) for n, d in G.nodes(data=True)} == \
        {P(0): (0, 12), P(1): (2, 30), Q(0): (0, 12), Q(1): (2, 30)}
    assert G.nodes[P(0)]['generations'] == [0, 5]


def test_node_ids_avoid_the_names_styling_and_traces_match():
    G = build([(0, A, X), (1, B, Y)], series_idx=3)
    for node in G.nodes:
        assert re.fullmatch(r"MO_S\d+_R\d+_[PQ]\d+", node), node
        assert not any(s in node for s in ("STN", "Local Optimum", "Noisy")), node


def test_nan_hv_states_are_reused():
    nan = float('nan')
    G = build([(0, state((1,), nan), state((7,), nan)), (1, state((2,), nan), state((7,), nan)),
               (2, state((1,), nan), state((7,), nan))])
    assert set(G.nodes) == {P(0), P(1), Q(0)}
    assert all(math.isnan(d['fitness']) and math.isnan(d['hypervolume']) for _, d in G.nodes(data=True))
    assert edges(G, 'Noise_MO') == {(P(0), Q(0)): [0, 2], (P(1), Q(0)): [1]}


def test_single_front_items_add_no_satellites():
    items = stn_state_items([(0, A, None), (1, B, None)])
    assert all(i['front2'] is None and i['metric2'] is None and i['state2'] is None for i in items)
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [items], 'blue', 0)
    assert {d['type'] for _, d in G.nodes(data=True)} == {'STN_MO'}
    assert {d['edge_type'] for _, _, d in G.edges(data=True)} == {'STN_MO'}


def test_reuse_is_within_a_run_and_series():
    G = nx.MultiDiGraph()
    items = stn_state_items([(0, A, X), (1, B, X)])
    add_mo_fronts(G, [items, items], 'blue', 0)
    add_mo_fronts(G, [items], 'red', 1)
    assert set(G.nodes) == {P(k, s, r) for k in (0, 1) for s, r in ((0, 0), (0, 1), (1, 0))} | \
        {Q(0, s, r) for s, r in ((0, 0), (0, 1), (1, 0))}
    assert G.nodes[P(0, 0, 1)]['run_idx'] == 1 and G.nodes[P(0, 1, 0)]['color'] == 'red'


def test_a_state_id_given_two_front_states_is_refused():
    items = stn_state_items([(0, A, None), (1, B, None)])
    items[1] = {**items[1], 'state1': 0}
    with pytest.raises(ValueError, match="two different front states"):
        add_mo_fronts(nx.MultiDiGraph(), [items], 'blue', 0)


def test_state_graph_node_sizes_stay_on_the_shared_scale():
    G = build([(0, state((1, 2, 3), 1.0), state((7,), 2.0)), (1, state((1,), 3.0), state((7, 8, 9, 10, 11), 2.5))])
    cfg = PlotConfig()
    cfg.stn_plot_type = 'multiobjective'
    cfg.node_size.stn_min, cfg.node_size.stn_max = 5.0, 15.0
    apply_node_sizes(G, cfg)
    sizes = {n: d['size'] for n, d in G.nodes(data=True)}
    assert min(sizes.values()) == 5.0 and max(sizes.values()) == 15.0
    assert sizes[P(1)] == sizes[Q(0)] == 5.0 and sizes[Q(1)] == 15.0
    create_node_traces(G, {n: (float(i), 0.0) for i, n in enumerate(G.nodes)}, cfg)   # Plotly validates sizes


# ------------------------------------------------------------------------------ real Stage 1 entries

@pytest.mark.parametrize("view_id,pov", [("population_noisy_vs_clean", "algo"), ("population_noisy_vs_clean", "true"),
                                         ("population_noisy_vs_corrected", "algo"), ("archive_noisy_vs_clean", "algo"),
                                         ("population_noisy", "algo")])
def test_crafted_run_graph_matches_its_items(view_id, pov):
    """The crafted Stage 1 run: one node per distinct state id and role, holding that state's front and HV;
    one STN_MO edge per distinct transition between consecutive primary states; one Noise_MO edge per
    distinct (primary, satellite) pairing."""
    _, view = crafted()
    items = stn_view_entries(view, view_id, pov)
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [items], 'blue', 0)

    for item in items:
        primary = G.nodes[P(item['state1'])]
        assert (primary['front_solutions'], primary['fitness'], primary['type']) == \
            (item['front1'], item['metric1'], 'STN_MO')
        if item['state2'] is not None:
            satellite = G.nodes[Q(item['state2'])]
            assert (satellite['front_solutions'], satellite['fitness'], satellite['type']) == \
                (item['front2'], item['metric2'], 'STN_MO_Noise')
    transitions = {}
    for prev, item in zip(items, items[1:]):
        if item['state1'] != prev['state1']:
            transitions.setdefault((P(prev['state1']), P(item['state1'])), []).append(item['gen_idx'])
    assert edges(G, 'STN_MO') == transitions
    pairs = {}
    for item in items:
        if item['state2'] is not None:
            pairs.setdefault((P(item['state1']), Q(item['state2'])), []).append(item['gen_idx'])
    assert edges(G, 'Noise_MO') == pairs
    assert len(nodes(G, 'STN_MO')) == len({i['state1'] for i in items})


def test_crafted_noisy_vs_clean_graph():
    """Items (gen, primary, satellite): g0 0/0, g2 1/0, g4 1/1, g5 1/0, g6 2/0, g7 1/2 (test_mo_stn_views)."""
    _, view = crafted()
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [stn_view_entries(view, 'population_noisy_vs_clean')], 'blue', 0)
    assert nodes(G, 'STN_MO') == {P(0): ([0], 1, 0), P(1): ([2, 7], 2, 2), P(2): ([6], 1, 6)}
    assert nodes(G, 'STN_MO_Noise') == {Q(0): ([0, 5], 2, 0), Q(1): ([4], 1, 4), Q(2): ([7], 1, 7)}
    assert edges(G, 'STN_MO') == {(P(0), P(1)): [2], (P(1), P(2)): [6], (P(2), P(1)): [7]}
    assert edges(G, 'Noise_MO') == {(P(0), Q(0)): [0], (P(1), Q(0)): [2, 5], (P(1), Q(1)): [4], (P(2), Q(0)): [6],
                                    (P(1), Q(2)): [7]}
    assert [G.nodes[P(k)]['n_observations'] for k in range(3)] == [4, 4, 3]


# ------------------------------------------------------------------------------ hover text

@pytest.mark.parametrize("square", [True, False])
def test_mo_hover_shows_discovery_counts_and_satellite_connections(square):
    """Every MO node's hover text: its HV, the generation and evaluation count at which its state was first
    seen, its solutions counted with repeats (observations) and unique (distinct genotypes), and its satellite
    connections (a primary's edges to satellites, a satellite's edges to primaries), on whichever trace it is
    drawn. Run 0: g0 A/X, g1 B/X, g2 B/Y, so A has 1 satellite, B 2, X 2 primaries, Y 1. Series 1 is a
    single-front run: 0. Other nodes' hover text is unchanged."""
    G = nx.MultiDiGraph()
    add_mo_fronts(G, [with_evals(stn_state_items([(0, state((1, 2), 1.5, n=5), state((1, 2, 3), 2.5, n=3)),
                                                  (1, state((4,), 2.0, n=2), state((1, 2, 3), 2.5, n=3)),
                                                  (2, state((4,), 2.0, n=2), state((7,), 3.5, n=1))]),
                                 {0: 10, 1: 20, 2: 30})], 'blue', 0)
    add_mo_fronts(G, [with_evals(stn_state_items([(4, state((9,), 0.5, n=4), None)]), {4: 44})], 'red', 1)
    G.add_node('STN_S0_R0_Sol0_True', type='STN', fitness=4.0, iterations=2, evals=7)
    G.add_node('Local Optimum 1', fitness=3.0)
    cfg = PlotConfig()
    cfg.stn_plot_type = 'multiobjective'
    cfg.stn.noisy_nodes_square = square
    lon, stn, square_trace = create_node_traces(G, {n: (float(i), 0.0) for i, n in enumerate(G.nodes)}, cfg)
    texts = sorted(list(stn.text) + list(square_trace.text))
    mo = lambda hv, g, e, n, u, links: (f"Fitness: {hv}<br>Gens: {g}<br>Evals: {e}<br>Solutions (incl. repeats): {n}"
                                        f"<br>Unique solutions: {u}<br>Satellite connections: {links}")
    assert texts == sorted([
        mo(1.5, 0, 10, 5, 2, 1),          # A
        mo(2.5, 0, 10, 3, 3, 2),          # X, shared by A and B
        mo(2.0, 1, 20, 2, 1, 2),          # B, with X and Y
        mo(3.5, 2, 30, 1, 1, 1),          # Y
        mo(0.5, 4, 44, 4, 1, 0),          # the single-front primary
        "Fitness: 4.0<br>Gens: 2<br>Evals: 7",
    ])
    assert list(lon.text) == ["Fitness: 3.0<br>Gens: <br>Evals: "]
    assert (len(square_trace.text) == 2) == square
