"""The multi-objective STN views in the dashboard: controls, callbacks and the path to the plotted graph.

    controls    mo_plot_type lists the five views (default population_noisy, not clearable); mo_stn_pov the
                two points of view (default algo), disabled for the single-front views
    callbacks   process_STN_data builds STN_MO_data with mo_frames.stn_view_entries for the selected view and
                POV, and MO_data_PPP (the Pareto tab's data) with front_entries, whatever the view or POV
    pipeline    STN_MO_data, through the browser store's JSON, into add_mo_fronts, calculate_positions_mo
                and the node and edge traces

The two callbacks run from their own source in graph_data.py (the module itself loads the warehouse when
imported) against MORunView records. The live app's contract and HTTP responses are pinned in
test_dashboard_package; the parts are covered in test_mo_stn_views, _graph and _positions.

Runs in-process and writes nothing.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
from collections import defaultdict

import networkx as nx
import pandas as pd
import pytest
from plotly.io.json import to_json_plotly

from harness.mo_runs import build, step_run
from noisyvis.common import sol_tuple_ints
from noisyvis.dashboard import mo_frames
from noisyvis.dashboard.layout.components import create_mo_stn_options_section, create_multiobjective_options_section
from noisyvis.dashboard.layout.main_layout import create_layout
from noisyvis.dashboard.mo_frames import (
    DEFAULT_STN_POV,
    DEFAULT_STN_VIEW,
    STN_VIEWS,
    front_entries,
    row_view,
    stn_view_entries,
    stn_view_fronts,
)
from noisyvis.results.mo_view import MORunView
from noisyvis.tracking.logger import clear_active_logger
from noisyvis.viz.config import PlotConfig, parse_callback_inputs
from noisyvis.viz.graph.stn import add_mo_fronts
from noisyvis.viz.layout import calculate_positions
from noisyvis.viz.styling import apply_node_sizes, style_nodes
from noisyvis.viz.traces import create_edge_traces, create_node_traces
from test_mo_stn_views import crafted

GRAPH_DATA = mo_frames.__file__.replace("mo_frames.py", "callbacks/graph_data.py")
PAIRED = ('population_noisy_vs_clean', 'population_noisy_vs_corrected', 'archive_noisy_vs_clean')
SINGLE = ('population_noisy', 'archive_noisy')


def callback(name, namespace):
    """A graph_data.py callback compiled from its source, without its decorator, into `namespace`."""
    tree = ast.parse(open(GRAPH_DATA).read())
    [func] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    func.decorator_list = []
    exec(compile(ast.Module(body=[func], type_ignores=[]), GRAPH_DATA, "exec"), namespace)
    return namespace[name]


def find(component, component_id):
    """The component with that id in a component tree (or a list of trees)."""
    if getattr(component, 'id', None) == component_id:
        return component
    children = component if isinstance(component, (list, tuple)) else getattr(component, 'children', None)
    for child in children if isinstance(children, (list, tuple)) else [children] if children is not None else []:
        found = find(child, component_id)
        if found is not None:
            return found
    return None


# ------------------------------------------------------------------------------ records

@pytest.fixture(scope="module")
def frame():
    """The server-side frame: three NSGA-II rows (two noise levels) and two SEMO rows, one without a
    reference point (its HVs are NaN), plus the crafted run."""
    clear_active_logger()
    rows = []
    for algo, prob, noise, params in (("NSGA2", "kp1", 1.0, {}), ("NSGA2", "prior", 1.0, {"seed": 2}),
                                      ("NSGA2", "kp1", 2.0, {"seed": 3}), ("SEMO", "kp1", 1.0, {}),
                                      ("SEMO", "kp1", 1.0, {"seed": 4, "ref_point": None})):
        algo_run = build(algo, prob=prob, **params)
        step_run(algo_run)
        rows.append({"algo_name": algo, "noise": noise, "mo_record": algo_run.eval_log.finalise()})
    lab, _ = crafted()
    rows.append({"algo_name": "Crafted", "noise": 1.0, "mo_record": lab.log.finalise()})
    clear_active_logger()
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def process(frame):
    return callback("process_STN_data", {
        "pd": pd, "df": frame, "row_view": row_view, "front_entries": front_entries,
        "stn_view_entries": stn_view_entries, "DEFAULT_STN_VIEW": DEFAULT_STN_VIEW, "DEFAULT_STN_POV": DEFAULT_STN_POV})


def stn_rows(frame):
    """What the STN_data store holds: each row's metadata and its row id (no mo_record)."""
    return frame.drop(columns=['mo_record']).assign(_row=frame.index).to_dict('records')


# ------------------------------------------------------------------------------ controls

def test_view_and_point_of_view_controls():
    section = create_multiobjective_options_section()
    views = find(section, 'mo_plot_type')
    assert [o['value'] for o in views.options] == list(STN_VIEWS)
    assert [o['label'] for o in views.options] == [
        'Population front: noisy only', 'Population front: noisy vs clean',
        'Population front: noisy vs corrected noisy', 'Historical archive: noisy only',
        'Historical archive: noisy vs clean']
    assert views.value == 'population_noisy' and views.clearable is False
    pov = find(section, 'mo_stn_pov')
    assert [(o['value'], o['label']) for o in pov.options] == [('algo', 'Algorithm POV'), ('true', 'True POV')]
    assert pov.value == 'algo' and pov.clearable is False


def preorder(component):
    """A component tree's components in document order."""
    yield component
    children = getattr(component, 'children', None)
    for child in children if isinstance(children, (list, tuple)) else [children] if children is not None else []:
        if hasattr(child, 'to_plotly_json'):
            yield from preorder(child)


def test_node_size_control_in_its_own_section_after_the_stn_run_options():
    size = find(create_mo_stn_options_section(), 'mo-stn-node-size')
    assert [(o['value'], o['label']) for o in size.options] == [('unique', 'unique solutions'), ('all', 'all solutions')]
    assert size.value == 'all' and size.clearable is False

    empty = pd.DataFrame()
    order = []
    for c in preorder(create_layout(empty, [], empty, empty, [], [])):
        label = getattr(c, 'children', None) if type(c).__name__ == 'Label' else None
        if label in ("STN run options:", "Multiobjective STN options:", "LON options:"):
            order.append(label)
        elif getattr(c, 'id', None) == 'mo-stn-node-size':
            order.append('node size')
        elif type(c).__name__ == 'Hr' and order:
            order.append('hr')
    i = order.index("Multiobjective STN options:")
    assert order[i - 2:i + 4] == ["STN run options:", "hr", "Multiobjective STN options:", "node size", "hr", "LON options:"]


def test_node_size_basis_defaults_to_all_solutions():
    params = inspect.signature(parse_callback_inputs).parameters
    required = {name: ({} if name == 'axis_values' else None) for name, p in params.items() if p.default is p.empty}
    assert parse_callback_inputs(**required).mo_node_size == 'all'
    assert parse_callback_inputs(**required, mo_node_size=None).mo_node_size == 'all'
    assert parse_callback_inputs(**required, mo_node_size='unique').mo_node_size == 'unique'


def test_pov_is_disabled_exactly_for_single_front_views():
    toggle = callback("toggle_mo_stn_pov", {"stn_view_fronts": stn_view_fronts, "DEFAULT_STN_VIEW": DEFAULT_STN_VIEW})
    assert {view: toggle(view) for view in STN_VIEWS} == {**{v: True for v in SINGLE}, **{v: False for v in PAIRED}}
    assert toggle(None) is True                                    # the default view is single-front
    with pytest.raises(ValueError):
        toggle('bpbhv_algo_pov')


# ------------------------------------------------------------------------------ callbacks

def test_stn_data_follows_view_and_pov_and_pareto_data_does_not(frame, process):
    rows = stn_rows(frame)
    groups = [list(g.index) for _, g in frame.groupby(['algo_name', 'noise'])]
    pareto = None
    for view_id in STN_VIEWS:
        for pov in ('algo', 'true'):
            _, _, _, mo_data, mo_series, mo_ppp = process(rows, view_id, pov)
            assert mo_data == [[stn_view_entries(row_view(frame, r), view_id, pov) for r in g] for g in groups]
            assert mo_series == [k for k, _ in frame.groupby(['algo_name', 'noise'])]
            if pareto is None:
                pareto = mo_ppp
                assert mo_ppp == [[front_entries(row_view(frame, r)) for r in g] for g in groups]
            assert mo_ppp == pareto, (view_id, pov)                  # the Pareto tab's data never moves
    # single-front views ignore the POV; paired views do not
    for view_id in SINGLE:
        assert process(rows, view_id, 'true')[3] == process(rows, view_id, 'algo')[3]
    for view_id in PAIRED:
        assert process(rows, view_id, 'true')[3] != process(rows, view_id, 'algo')[3]


def test_missing_values_take_the_defaults_and_unknown_ids_are_refused(frame, process):
    rows = stn_rows(frame)
    assert process(rows, None, None)[3] == process(rows, 'population_noisy', 'algo')[3]
    assert process(rows, 'archive_noisy_vs_clean', None)[3] == process(rows, 'archive_noisy_vs_clean', 'algo')[3]
    for view_id, pov in (('npnhv', 'algo'), ('population_noisy', 'clean')):
        with pytest.raises(ValueError):
            process(rows, view_id, pov)
    assert process([], 'population_noisy', 'algo') == ([], [], [], [], [], [])


# ------------------------------------------------------------------------------ to the plotted graph

def plotted(frame, process, view_id, pov, layout_type='mds'):
    """STN_MO_data through the browser store's JSON, then the update_plot MO steps: graph, styling,
    positions, node and edge traces."""
    mo_data = json.loads(to_json_plotly(process(stn_rows(frame), view_id, pov)[3]))
    config = PlotConfig()
    config.stn_plot_type = 'multiobjective'
    G = nx.MultiDiGraph()
    for idx, runs in enumerate(mo_data):
        add_mo_fronts(G, runs, config.algo_colors[idx % len(config.algo_colors)], idx, config.noisy_node_color)
    style_nodes(G, config)
    pos = calculate_positions(G, layout_type, 'multiobjective')
    _, node_trace, square_trace = create_node_traces(G, pos, config)   # satellites may be drawn square
    drawn = list(zip(node_trace.x + square_trace.x, node_trace.y + square_trace.y, node_trace.z + square_trace.z,
                     node_trace.text + square_trace.text))
    edge_traces = create_edge_traces(G, pos, config)[0]
    return mo_data, G, pos, drawn, edge_traces


def key(d):
    return frozenset(sol_tuple_ints(s) for s in d['front_solutions'])


@pytest.mark.parametrize("view_id", list(STN_VIEWS))
@pytest.mark.parametrize("pov", ['algo', 'true'])
def test_every_view_reaches_the_plot(frame, process, view_id, pov):
    if view_id in SINGLE and pov == 'true':
        pytest.skip("single-front views ignore the POV")
    mo_data, G, pos, drawn, edge_traces = plotted(frame, process, view_id, pov)
    primary_kind, satellite_kind = stn_view_fronts(view_id, pov)

    # one node per state id and role in each run; satellites exactly for paired views
    for s, runs in enumerate(mo_data):
        for r, items in enumerate(runs):
            assert {f"MO_S{s}_R{r}_P{i['state1']}" for i in items} == \
                {n for n in G if n.startswith(f"MO_S{s}_R{r}_P")}
            assert {f"MO_S{s}_R{r}_Q{i['state2']}" for i in items if i['state2'] is not None} == \
                {n for n in G if n.startswith(f"MO_S{s}_R{r}_Q")}
    assert any(d['type'] == 'STN_MO_Noise' for _, d in G.nodes(data=True)) == (satellite_kind is not None)

    # node heights are the stored HVs (NaN for the record without a reference point), sizes distinct genotypes
    for n, d in G.nodes(data=True):
        assert d['front_size'] == len(key(d)) and d['fitness'] is d['hypervolume']
    nan_nodes = [n for n, d in G.nodes(data=True) if math.isnan(d['fitness'])]
    assert nan_nodes and all(n.startswith("MO_S3_R1_") for n in nan_nodes)   # (SEMO, 1.0), its second row

    # every node and edge is placed; equal genotype sets share one XY
    assert set(pos) == set(G.nodes) and all(math.isfinite(c) for xy in pos.values() for c in xy)
    by_set = defaultdict(set)
    for n, d in G.nodes(data=True):
        by_set[key(d)].add(pos[n])
    assert all(len(xy) == 1 for xy in by_set.values())
    assert len(edge_traces) == G.number_of_edges()
    assert len(drawn) == G.number_of_nodes()
    expected = sorted((*pos[n], d['fitness']) for n, d in G.nodes(data=True) if not math.isnan(d['fitness']))
    assert sorted(p[:3] for p in drawn if not math.isnan(p[2])) == expected
    # each node's hover text: HV; the generation and evaluation count at which its state was first seen (the
    # record's own count by the end of that generation); its solutions with repeats and unique; its satellite
    # connections (a primary's edges to satellites, a satellite's to primaries)
    groups = [list(g.index) for _, g in frame.groupby(['algo_name', 'noise'])]
    links = {}
    for n, d in G.nodes(data=True):
        s, r = (int(part[1:]) for part in n.split('_')[1:3])
        assert d['first_evals'] == row_view(frame, groups[s][r]).evals_through(d['gen_idx']), n
        edges = G.in_edges(n, data=True) if d['type'] == 'STN_MO_Noise' else G.out_edges(n, data=True)
        links[n] = sum(1 for *_, e in edges if e['edge_type'] == 'Noise_MO')
    assert sorted(text for *_, text in drawn) == sorted(
        f"Fitness: {d['fitness']}<br>Gens: {d['gen_idx']}<br>Evals: {d['first_evals']}"
        f"<br>Solutions (incl. repeats): {d['n_observations']}<br>Unique solutions: {d['front_size']}"
        f"<br>Satellite connections: {links[n]}" for n, d in G.nodes(data=True))
    assert all(d['n_observations'] >= d['front_size'] for _, d in G.nodes(data=True))
    # fronts whose members are genotypes (clean, corrected): solutions with repeats == unique solutions
    kind_of = {'STN_MO': primary_kind, 'STN_MO_Noise': satellite_kind}
    for _, d in G.nodes(data=True):
        if kind_of[d['type']] in ('corrected_noisy', 'clean_population', 'clean_archive'):
            assert d['n_observations'] == d['front_size'], (kind_of[d['type']], d['gen_idx'])
    n_noise = sum(1 for *_, e in G.edges(data=True) if e['edge_type'] == 'Noise_MO')
    by_role = lambda t: sum(links[n] for n, d in G.nodes(data=True) if d['type'] == t)
    assert by_role('STN_MO') == by_role('STN_MO_Noise') == n_noise
    assert (n_noise == 0) == (satellite_kind is None)


@pytest.mark.parametrize("view_id", ['population_noisy', 'population_noisy_vs_clean', 'archive_noisy_vs_clean'])
def test_node_sizes_follow_the_selected_count(frame, process, view_id):
    """The plot's default basis (all solutions) and unique solutions: on one range over primaries and
    satellites, a larger count is a larger marker and equal counts are equal markers."""
    _, G, _, _, _ = plotted(frame, process, view_id, 'algo')
    for mode, attr in (('all', 'n_observations'), ('unique', 'front_size')):
        if mode == 'unique':
            config = PlotConfig()
            config.stn_plot_type = 'multiobjective'
            config.mo_node_size = 'unique'
            apply_node_sizes(G, config)
        nodes = [d for _, d in G.nodes(data=True)]
        counts = [d[attr] for d in nodes]
        assert {d['size'] for d in nodes if d[attr] == min(counts)} == {PlotConfig().node_size.stn_min}
        assert {d['size'] for d in nodes if d[attr] == max(counts)} == {PlotConfig().node_size.stn_max}
        for a in nodes:
            for b in nodes:
                if a[attr] < b[attr]:
                    assert a['size'] < b['size'], (mode, a[attr], b[attr])
                elif a[attr] == b[attr]:
                    assert a['size'] == b['size'], mode


def test_point_of_view_swaps_the_trajectory(frame, process):
    for view_id in PAIRED:
        _, algo, _, _, _ = plotted(frame, process, view_id, 'algo')
        _, true, _, _, _ = plotted(frame, process, view_id, 'true')
        sets = lambda G, t: {(n.split('_R')[0], key(d), repr(d['fitness'])) for n, d in G.nodes(data=True)
                             if d['type'] == t}
        assert sets(algo, 'STN_MO') == sets(true, 'STN_MO_Noise')
        assert sets(algo, 'STN_MO_Noise') == sets(true, 'STN_MO')
        temporal = lambda G, t: all(G.nodes[u]['type'] == G.nodes[v]['type'] == t
                                    for u, v, d in G.edges(data=True) if d['edge_type'] == 'STN_MO')
        assert temporal(algo, 'STN_MO') and temporal(true, 'STN_MO')


def test_tsne_end_to_end(frame, process):
    _, G, pos, drawn, _ = plotted(frame, process, 'population_noisy_vs_clean', 'algo', layout_type='tsne')
    assert set(pos) == set(G.nodes) and all(math.isfinite(c) for xy in pos.values() for c in xy)
    assert len(drawn) == G.number_of_nodes()
