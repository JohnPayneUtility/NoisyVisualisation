"""
Reusable UI component functions for Dashboard layout.

Each function returns a Dash component or group of components
that can be assembled in the main layout.
"""
import plotly.graph_objects as go
from dash import html, dcc, dash_table

from .styles import (
    TAB_STYLE,
    TAB_SELECTED_STYLE,
    SECTION_STYLE,
    SELECTION_OUTPUT_STYLE,
    DROPDOWN_STYLE,
    DROPDOWN_STYLE_WITH_MARGIN,
    SMALL_INPUT_STYLE,
    INLINE_BLOCK_STYLE,
    INLINE_BLOCK_WITH_MARGIN,
    FLEX_ROW_STYLE,
    FLEX_WITH_GAP_STYLE,
    INLINE_DROPDOWN_WRAPPER_STYLE,
    INLINE_VERTICAL_ALIGN_STYLE,
    INLINE_VERTICAL_ALIGN_NO_MARGIN_STYLE,
    FULL_WIDTH_INLINE_STYLE,
    MONOSPACE_STYLE,
)


_SCHEMATIC_SERIES_COLOR = '#21918c'
_SCHEMATIC_SAT_COLOR = '#9e9e9e'


def _build_schematic_figure(simple_mode: bool = False, show_misjudgements: bool = False) -> go.Figure:
    """Build a Plotly 3D figure illustrating STN plot elements.

    simple_mode: label elements A/B/C/D/E inline instead of verbose annotations.
    show_misjudgements: add a 3rd base node + satellite showing a misjudgement.
    """
    series_color = _SCHEMATIC_SERIES_COLOR
    sat_color = _SCHEMATIC_SAT_COLOR
    noise_color = 'grey'
    label_size = 16 if simple_mode else 13  # larger inline labels when simple mode is on

    # Node positions: (x, y, z=fitness)
    # Base nodes carry the noisy (evaluated) position; satellites carry the true position.
    bx_a, by_a, bz_a = 0.0, 0.0, 6.8   # base node 1 — noisy fitness
    bx_b, by_b, bz_b = 3.2, 1.5, 9.8   # base node 2 — noisy fitness, slight y offset
    sx_a, sy_a, sz_a = 0.0, 0.0, 5.0   # C1: posterior noise satellite — same x-y, lower true fitness
    sx_b, sy_b, sz_b = 4.0, 0.7, 8.0   # C2: prior noise satellite — offset x-y, lower true fitness

    # Base node 3: noisy fitness above node 2 (looks attractive), true fitness much lower → misjudgement
    bx_c, by_c, bz_c = 8.2, 1.5, 10.5
    sx_c, sy_c, sz_c = 7.5, 0.4, 6.5   # satellite 3 — true fitness, much lower than noisy

    traces = []

    # STN transition edges: 1→2 (always), 2→3 (misjudgement only)
    stn_xs = [bx_a, bx_b]
    stn_ys = [by_a, by_b]
    stn_zs = [bz_a, bz_b]
    if show_misjudgements:
        stn_xs += [None, bx_b, bx_c]
        stn_ys += [None, by_b, by_c]
        stn_zs += [None, bz_b, bz_c]
    traces.append(go.Scatter3d(
        x=stn_xs, y=stn_ys, z=stn_zs,
        mode='lines',
        line=dict(width=5, color=series_color),
        hoverinfo='none', showlegend=False,
    ))

    # Noise edges (grey solid): base → satellite
    noise_pairs = [
        ((bx_a, by_a, bz_a), (sx_a, sy_a, sz_a)),
        ((bx_b, by_b, bz_b), (sx_b, sy_b, sz_b)),
    ]
    if show_misjudgements:
        noise_pairs.append(((bx_c, by_c, bz_c), (sx_c, sy_c, sz_c)))
    for (x0, y0, z0), (x1, y1, z1) in noise_pairs:
        traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(width=3, color=noise_color),
            hoverinfo='none', showlegend=False,
        ))

    # Base nodes (circles) — label A in simple mode
    base_xs = [bx_a, bx_b]
    base_ys = [by_a, by_b]
    base_zs = [bz_a, bz_b]
    base_labels = ['A', 'A']
    if show_misjudgements:
        base_xs.append(bx_c)
        base_ys.append(by_c)
        base_zs.append(bz_c)
        base_labels.append('A')
    traces.append(go.Scatter3d(
        x=base_xs, y=base_ys, z=base_zs,
        mode='markers+text' if simple_mode else 'markers',
        marker=dict(size=14, color=series_color, symbol='circle'),
        text=base_labels if simple_mode else None,
        textposition='top center',
        textfont=dict(size=label_size, color='black'),
        hoverinfo='none', showlegend=False,
    ))

    # True satellite nodes (squares, grey) — C1/C2(/C3) in simple mode
    sat_xs = [sx_a, sx_b]
    sat_ys = [sy_a, sy_b]
    sat_zs = [sz_a, sz_b]
    sat_labels = ['C1', 'C2']
    if show_misjudgements:
        sat_xs.append(sx_c)
        sat_ys.append(sy_c)
        sat_zs.append(sz_c)
        sat_labels.append('C2')
    traces.append(go.Scatter3d(
        x=sat_xs, y=sat_ys, z=sat_zs,
        mode='markers+text' if simple_mode else 'markers',
        marker=dict(size=10, color=sat_color, symbol='square'),
        text=sat_labels if simple_mode else None,
        textposition='top center',
        textfont=dict(size=13, color='black'),
        hoverinfo='none', showlegend=False,
    ))

    # Build scene annotations
    if simple_mode:
        # B label at STN edge 1→2 midpoint; D labels at noise edge midpoints
        traces.append(go.Scatter3d(
            x=[(bx_a + bx_b) / 2], y=[(by_a + by_b) / 2], z=[(bz_a + bz_b) / 2],
            mode='text', text=['B'],
            textfont=dict(size=label_size, color=series_color),
            hoverinfo='none', showlegend=False,
        ))
        d_xs = [(bx_a + sx_a) / 2, (bx_b + sx_b) / 2]
        d_ys = [(by_a + sy_a) / 2, (by_b + sy_b) / 2]
        d_zs = [(bz_a + sz_a) / 2, (bz_b + sz_b) / 2]
        d_text = ['D', 'D']
        if show_misjudgements:
            d_xs.append((bx_c + sx_c) / 2)
            d_ys.append((by_c + sy_c) / 2)
            d_zs.append((bz_c + sz_c) / 2)
            d_text.append('D')
        traces.append(go.Scatter3d(
            x=d_xs, y=d_ys, z=d_zs,
            mode='text', text=d_text,
            textfont=dict(size=label_size, color='grey'),
            hoverinfo='none', showlegend=False,
        ))
        if show_misjudgements:
            # B label at STN edge 2→3 midpoint
            traces.append(go.Scatter3d(
                x=[(bx_b + bx_c) / 2], y=[(by_b + by_c) / 2], z=[(bz_b + bz_c) / 2],
                mode='text', text=['B'],
                textfont=dict(size=label_size, color=series_color),
                hoverinfo='none', showlegend=False,
            ))
        scene_annotations = []
    else:
        scene_annotations = [
            dict(
                x=bx_a, y=by_a, z=bz_a,
                text='Base node<br>(noisy sol, noisy fit)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='black', ax=-90, ay=0,
                font=dict(size=11, color='black'),
            ),
            dict(
                x=bx_b, y=by_b, z=bz_b,
                text='Base node<br>(noisy sol, noisy fit)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='black', ax=90, ay=0,
                font=dict(size=11, color='black'),
            ),
            dict(
                x=sx_a, y=sy_a, z=sz_a,
                text='C1: Posterior noise satellite<br>(same x-y, different fitness)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='black', ax=-110, ay=30,
                font=dict(size=11, color='black'),
            ),
            dict(
                x=sx_b, y=sy_b, z=sz_b,
                text='C2: Prior noise satellite<br>(offset x-y and different fitness)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='black', ax=110, ay=30,
                font=dict(size=11, color='black'),
            ),
            dict(
                x=(bx_a + bx_b) / 2, y=(by_a + by_b) / 2, z=(bz_a + bz_b) / 2,
                text='STN transition edge<br>(algorithm movement)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=series_color, ax=0, ay=60,
                font=dict(size=11, color=series_color),
            ),
            dict(
                x=(bx_a + sx_a) / 2, y=(by_a + sy_a) / 2, z=(bz_a + sz_a) / 2,
                text='Noise edge<br>(noisy vs true sol)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='grey', ax=-90, ay=0,
                font=dict(size=11, color='grey'),
            ),
            dict(
                x=(bx_b + sx_b) / 2, y=(by_b + sy_b) / 2, z=(bz_b + sz_b) / 2,
                text='Noise edge<br>(noisy vs true sol)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='grey', ax=90, ay=0,
                font=dict(size=11, color='grey'),
            ),
        ]
        if show_misjudgements:
            scene_annotations.append(dict(
                x=(bx_c + sx_c) / 2, y=(by_c + sy_c) / 2, z=(bz_c + sz_c) / 2,
                text='Noise edge<br>(noisy vs true sol)',
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='grey', ax=90, ay=0,
                font=dict(size=11, color='grey'),
            ))

    # Misjudgement annotation (red arrow) — added in both modes when enabled
    if show_misjudgements:
        mj_text = 'E' if simple_mode else 'Misjudgement<br>(moved to worse true fitness)'
        # verbose mode uses a long ax offset so the arrow is clearly visible
        mj_ax = 20 if simple_mode else 90
        scene_annotations.append(dict(
            x=bx_c, y=by_c, z=bz_c,
            text=mj_text,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor='red', ax=mj_ax, ay=0,
            font=dict(size=label_size, color='red'),
        ))

    import math
    az_rad = math.radians(35)
    el_rad = math.radians(60)
    r = 2.0
    cam = dict(
        x=r * math.cos(el_rad) * math.cos(az_rad),
        y=r * math.cos(el_rad) * math.sin(az_rad),
        z=r * math.sin(el_rad),
    )

    axis_title_size = 24 if simple_mode else 14
    axis_tick_size = 16 if simple_mode else 12

    scene_dict = dict(
        camera=dict(eye=cam),
        xaxis=dict(
            title='Dimension 1',
            titlefont=dict(size=axis_title_size, color='black'),
            tickfont=dict(size=axis_tick_size, color='black'),
            showticklabels=False,
        ),
        yaxis=dict(
            title='Dimension 2',
            titlefont=dict(size=axis_title_size, color='black'),
            tickfont=dict(size=axis_tick_size, color='black'),
            showticklabels=False,
        ),
        zaxis=dict(
            title='Fitness',
            titlefont=dict(size=axis_title_size, color='black'),
            tickfont=dict(size=axis_tick_size, color='black'),
        ),
    )
    if scene_annotations:
        scene_dict['annotations'] = scene_annotations

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=820, height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        template='plotly_white',
        showlegend=False,
        scene=scene_dict,
    )
    return fig


def _build_schematic_legend(simple_mode: bool = False, show_misjudgements: bool = False) -> list:
    """Return the children list for the schematic legend div."""
    series_color = _SCHEMATIC_SERIES_COLOR
    sat_color = _SCHEMATIC_SAT_COLOR

    item_style = {'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px', 'fontSize': '13px'}
    sym_base = {
        'width': '20px', 'height': '20px', 'borderRadius': '50%',
        'backgroundColor': series_color, 'marginRight': '10px', 'flexShrink': '0',
    }
    sym_sat = {
        'width': '16px', 'height': '16px',
        'backgroundColor': sat_color, 'marginRight': '10px', 'flexShrink': '0',
    }
    line_stn = {
        'width': '30px', 'height': '4px',
        'backgroundColor': series_color, 'marginRight': '10px', 'flexShrink': '0',
    }
    line_noise = {
        'width': '30px', 'height': '3px',
        'backgroundColor': '#888', 'marginRight': '10px', 'flexShrink': '0',
    }
    arrow_mj = {
        'width': '30px', 'height': '3px',
        'backgroundColor': 'red', 'marginRight': '10px', 'flexShrink': '0',
    }

    prefix_a  = 'A: '  if simple_mode else ''
    prefix_b  = 'B: '  if simple_mode else ''
    prefix_c1 = 'C1: ' if simple_mode else ''
    prefix_c2 = 'C2: ' if simple_mode else ''
    prefix_d  = 'D: '  if simple_mode else ''
    prefix_e  = 'E: '  if simple_mode else ''

    items = [
        html.B("Legend", style={'display': 'block', 'marginBottom': '8px'}),
        html.Div([html.Div(style=sym_base),  html.Span(f"{prefix_a}Base node — the solution the algorithm evaluated and moved from (noisy solution, noisy fitness)")], style=item_style),
        html.Div([html.Div(style=line_stn),  html.Span(f"{prefix_b}STN transition edge — the algorithm accepted this solution and moved from one base node to another")], style=item_style),
        html.Div([html.Div(style=sym_sat),   html.Span(f"{prefix_c1}Posterior noise satellite — true solution shares the same x-y position as its base node, only fitness differs")], style=item_style),
        html.Div([html.Div(style=sym_sat),   html.Span(f"{prefix_c2}Prior noise satellite — true solution is offset in both x-y position and fitness from its base node")], style=item_style),
        html.Div([html.Div(style=line_noise),html.Span(f"{prefix_d}Noise edge — connects each base node to its true satellite (offset = difference between noisy and true solution)")], style=item_style),
    ]
    if show_misjudgements:
        items.append(
            html.Div([html.Div(style=arrow_mj), html.Span(f"{prefix_e}Misjudgement — the algorithm moved to a node with lower true fitness because the noisy evaluation made it appear better")], style=item_style)
        )
    return items


def create_schematic_section() -> html.Div:
    """
    Create the schematic section: a 3D plot illustrating STN plot elements,
    with checkboxes to toggle simple annotations and misjudgement illustration.
    """
    return html.Div([
        html.H2("Schematic", style={'textAlign': 'center', 'marginBottom': '4px'}),
        html.Div([
            dcc.Checklist(
                id='schematic-misjudgements',
                options=[{'label': ' Misjudgements', 'value': 'misjudgements'}],
                value=[],
                style={'display': 'inline-block', 'marginRight': '20px'},
            ),
            dcc.Checklist(
                id='schematic-simple-annotations',
                options=[{'label': ' Simple annotations', 'value': 'simple'}],
                value=[],
                style={'display': 'inline-block'},
            ),
        ], style={'textAlign': 'center', 'marginBottom': '8px'}),
        dcc.Graph(
            id='schematic-graph',
            figure=_build_schematic_figure(simple_mode=False, show_misjudgements=False),
            config={'displayModeBar': False},
        ),
        html.Div(
            id='schematic-legend',
            children=_build_schematic_legend(simple_mode=False, show_misjudgements=False),
            style={'padding': '10px 20px 14px'},
        ),
    ], style={'borderBottom': '2px solid #ddd', 'marginBottom': '10px', 'paddingBottom': '10px'})


def create_problem_selection_section(display1_df, df_lon, lon_display_columns, experiment_names, experiment_descriptions=None):
    """
    Create the problem selection section with experiment dropdown, Table 1, and the LON table.

    Args:
        display1_df: DataFrame for problem selection (Table 1).
        df_lon: LON results DataFrame.
        lon_display_columns: Columns to show in the LON table.
        experiment_names: Sorted list of unique experiment name strings.
        experiment_descriptions: Dict mapping experiment name to description string.

    Returns:
        html.Div: Problem selection container.
    """
    table_wrapper_style = {
        "display": "flex",
        "justifyContent": "flex-start",
        "alignItems": "center",
        "padding": "10px",
        "marginTop": "0px",
    }
    return html.Div([
        html.B("Experiment Selection", style={"padding": "10px 10px 4px", "display": "block"}),
        html.Div(
            style={"display": "flex", "alignItems": "flex-start", "padding": "0px 10px 10px", "gap": "16px"},
            children=[
                dcc.Dropdown(
                    id='experiment-selector',
                    options=[{'label': '(null — no experiment name)', 'value': '__null__'}]
                            + [{'label': name, 'value': name} for name in experiment_names],
                    value=None,
                    multi=True,
                    clearable=True,
                    style={'width': '500px'},
                ),
                html.Div(
                    id='experiment-description-display',
                    style={'flex': '1', 'fontSize': '13px', 'color': '#444', 'paddingTop': '6px'},
                ),
            ],
        ),
        html.B("Problem selection table", style={"padding": "10px 10px 0px", "display": "block"}),
        html.Div(style=table_wrapper_style, children=[
            dash_table.DataTable(
                id="table1",
                data=display1_df.to_dict("records"),
                columns=[{"name": col, "id": col} for col in display1_df.columns],
                page_size=10,
                filter_action="native",
                row_selectable="single",
                selected_rows=[],
                style_table={"overflowX": "auto"},
            )
        ]),
        html.B("LON selection table", style={"padding": "10px 10px 0px", "display": "block"}),
        html.Div(style=table_wrapper_style, children=[
            dash_table.DataTable(
                id="LON_table",
                data=df_lon[lon_display_columns].to_dict("records"),
                columns=[{"name": col, "id": col} for col in lon_display_columns],
                page_size=10,
                row_selectable="single",
                style_table={"overflowX": "auto"},
            )
        ]),
    ], style=SECTION_STYLE)


def create_2d_plot_tabs():
    """
    Create the tabbed section for 2D performance plots.

    Returns:
        html.Div: 2D plot tabs container.
    """
    return html.Div([
        html.H3("2D Performance Plotting"),
        html.Div([
            html.Label("Fitness value to plot (SO line & box):"),
            dcc.Dropdown(
                id='so-fitness-mode',
                options=[
                    {'label': 'Best fitness', 'value': 'best'},
                    {'label': 'Final fitness', 'value': 'final'},
                ],
                value='best',
                clearable=False,
                style={'width': '260px'},
            ),
            html.Label("Cap noise for plot:"),
            dcc.Input(
                id='noise-cap-input',
                type='number',
                min=0,
                step=1,
                value=0,
                style={'width': '80px'},
            ),
            html.Label("Hide series:"),
            dcc.Dropdown(
                id='hide-series-dropdown',
                options=[],
                value=[],
                multi=True,
                placeholder='Select series to hide...',
                style={'width': '320px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '8px'}),
        dcc.Tabs(id='2DPlotTabSelection', value='p1', children=[
            dcc.Tab(
                label='Line plot (SO)',
                value='p1',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Box plot (SO)',
                value='p2',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Line plot evals (SO)',
                value='p6',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Box plot evals (SO)',
                value='p7',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Box plot misjudgements (SO)',
                value='p8',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Line plot (MO)',
                value='p3',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Box plot (MO)',
                value='p4',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Data',
                value='p5',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
        ]),
        html.Div(id='2DPlotTabContent'),
    ], style=SECTION_STYLE)


def create_performance_summary_table():
    """
    Create a placeholder div for the performance summary table.

    The table is populated by a callback based on the selected problem and
    the current 2D plot data (same data as the Line plot SO).

    Returns:
        html.Div: Container for the performance summary table.
    """
    return html.Div(id='performance-summary-table', style={'padding': '0 10px'})


def create_mann_whitney_table():
    """
    Create a placeholder div for the Mann-Whitney U-test table.

    The table is populated by a callback. It shows per-noise-level tabs with
    pairwise two-sided Mann-Whitney U p-values for each algorithm pair.

    Returns:
        html.Div: Container for the Mann-Whitney U-test table.
    """
    return html.Div(id='mann-whitney-table', style={'padding': '0 10px'})


def create_evals_summary_table():
    """
    Create a placeholder div for the evaluations performance summary table.

    Mirrors the fitness summary table but uses runtime (n_evals) as the metric.

    Returns:
        html.Div: Container for the evaluations summary table.
    """
    return html.Div(id='evals-summary-table', style={'padding': '0 10px'})


def create_evals_mann_whitney_table():
    """
    Create a placeholder div for the evaluations Mann-Whitney U-test table.

    Mirrors the fitness Mann-Whitney table but uses runtime (n_evals) as the metric.

    Returns:
        html.Div: Container for the evaluations Mann-Whitney U-test table.
    """
    return html.Div(id='evals-mann-whitney-table', style={'padding': '0 10px'})


def create_algorithm_table(display2_df, display2_hidden_cols):
    """
    Create Table 2 for algorithm selection.

    Args:
        display2_df: DataFrame with algorithm data.
        display2_hidden_cols: List of column names to hide.

    Returns:
        list: List containing table header, DataTable, output div, and separator.
    """
    return [
        html.H5("Table 2: Algorithms filtered by Selections"),
        dash_table.DataTable(
            id="table2",
            data=display2_df.to_dict("records"),
            columns=[
                {"name": col, "id": col}
                for col in display2_df.columns
                if col not in display2_hidden_cols
            ],
            page_size=10,
            filter_action="native",
            sort_action='native',
            row_selectable="multi",
            selected_rows=[],
            style_table={"overflowX": "auto"},
        ),
        html.Div(id="table2-selected-output", style=SELECTION_OUTPUT_STYLE),
        html.Hr(),
    ]


def create_pareto_front_section():
    """
    Create the Pareto front plotting section with dropdowns and graph.

    Returns:
        list: List of Pareto front components.
    """
    return [
        html.H3("Pareto front plotting"),
        html.H3("Plot type:"),
        dcc.Dropdown(
            id='paretoFrontPlotType',
            options=[
                {'label': 'Basic', 'value': 'Basic'},
                {'label': 'Subplots', 'value': 'Subplots'},
                {'label': 'Subplots (multi)', 'value': 'SubplotsMulti'},
                {'label': 'Subplots (highlighted)', 'value': 'SubplotsHighlight'},
                {'label': 'pareto animation', 'value': 'paretoAnimation'},
                {'label': 'Noisy', 'value': 'Noisy'},
                {'label': 'HV Vs Distance', 'value': 'IndVsDist'},
                {'label': 'IGD Vs Distance', 'value': 'IGDVsDist'},
                {'label': 'Progress per Movement', 'value': 'PPM'},
                {'label': 'Movement Correlation', 'value': 'MoveCorr'},
                {'label': 'Histogram', 'value': 'Hist'},
                {'label': 'Scatter', 'value': 'Scatter'}
            ],
            value='Basic',
            clearable=False,
            style=DROPDOWN_STYLE
        ),
        dcc.Dropdown(
            id='IndVsDist_IndType',
            options=[
                {'label': 'noisy HV', 'value': 'NoisyHV'},
                {'label': 'Clean HV', 'value': 'CleanHV'},
                {'label': 'IGD', 'value': 'IGD'}
            ],
            value='NoisyHV',
            clearable=False,
            style=DROPDOWN_STYLE
        ),
        dcc.Dropdown(
            id='IndVsDist_DistType',
            options=[
                {'label': 'Cumulative', 'value': 'cumulative'},
                {'label': 'Raw', 'value': 'raw'},
                {'label': 'MDS', 'value': 'mds'},
                {'label': 'tSNE', 'value': 'tsne'},
                {'label': 'ISOMAP', 'value': 'isomap'}
            ],
            value='cumulative',
            clearable=False,
            style=DROPDOWN_STYLE
        ),
        html.Label(" Number of runs to show: "),
        dcc.Input(
            id='paretoPlotNumRuns',
            type='number',
            min=0,
            max=1000,
            step=1,
            value=1
        ),
        html.Label(" Window size: "),
        dcc.Input(
            id='paretoPlotWindowSize',
            type='number',
            min=0,
            max=1000,
            step=1,
            value=5
        ),
        dcc.Graph(id="plotParetoFront"),
        html.Hr(),
    ]


def create_multiobjective_options_section():
    """
    Create the STN plot type selection and multiobjective plotting options section.

    Returns:
        list: List of STN plot type and MO options components.
    """
    return [
        html.Hr(),
        html.Label("STN plot type:", style={'fontWeight': 'bold'}),
        html.Br(),
        html.Div(
            dcc.Dropdown(
                id='stn-plot-type',
                options=[
                    {'label': 'Posterior noise STN plot', 'value': 'posterior'},
                    {'label': 'Posterior noise STN (algo POV)', 'value': 'posterior_algo_pov'},
                    {'label': 'Prior noise STN V4 plot', 'value': 'prior_v4'},
                    {'label': 'Prior noise STN V5 plot', 'value': 'prior_v5'},
                    {'label': 'Prior noise STN (algo POV)', 'value': 'prior_algo_pov'},
                    {'label': 'Multiobjective STN plot', 'value': 'multiobjective'},
                ],
                value='posterior',
                clearable=False,
                style=DROPDOWN_STYLE
            ),
            style=INLINE_DROPDOWN_WRAPPER_STYLE
        ),
        html.Br(),
        html.Label("Multiobjective options:", style={'fontWeight': 'bold'}),
        html.Br(),
        html.Div(
            dcc.Dropdown(
                id='mo_plot_type',
                options=[
                    {'label': 'noisy pareto noisy hv', 'value': 'npnhv'},
                    {'label': 'noisy pareto true hv', 'value': 'npthv'},
                    {'label': 'true pareto true hv', 'value': 'tpthv'},
                    {'label': 'noisy pareto both hv', 'value': 'npbhv'},
                    {'label': 'both pareto', 'value': 'bpbhv'},
                ],
                value='npnhv',
                placeholder='multiobjective plot type',
                style=DROPDOWN_STYLE
            ),
            style=INLINE_DROPDOWN_WRAPPER_STYLE
        ),
        html.Hr(),
    ]


def create_stn_options_section():
    """
    Create the STN run options section.

    Returns:
        list: List of STN options components.
    """
    return [
        html.Label("STN run options:", style={'fontWeight': 'bold'}),
        html.Br(),
        html.Div([
            html.Label(" Select starting run: "),
            dcc.Input(
                id='run-index',
                type='number',
                min=0,
                max=1000,
                step=1,
                value=0
            ),
            html.Label(" Number of runs to show: "),
            dcc.Input(
                id='run-selector',
                type='number',
                min=0,
                max=1000,
                step=1,
                value=1
            ),
            html.Label(" Lower fitness limit: "),
            dcc.Input(
                id='STN_lower_fit_limit',
                type='number',
                min=-1000000,
                max=1000000,
                step=1,
            ),
        ], style=INLINE_BLOCK_STYLE),
        dcc.Checklist(
            id='run-options',
            options=[
                {'label': 'Show best run', 'value': 'show_best'},
                {'label': 'Show mean run', 'value': 'show_mean'},
                {'label': 'Show median run', 'value': 'show_median'},
                {'label': 'Show worst run', 'value': 'show_worst'},
                {'label': 'Hamming distance labels', 'value': 'STN-hamming'},
                {'label': 'Dedup prior noise', 'value': 'dedup-prior-noise'},
                {'label': 'Show estimated fitness (adopted)', 'value': 'show_estimated_adopted'},
                {'label': 'Show estimated fitness (discarded)', 'value': 'show_estimated_discarded'},
                {'label': 'use est fitness (discarded) as base nodes', 'value': 'use_est_discarded_as_base'},
                {'label': 'Show fitness box plots', 'value': 'show_stn_boxplots'},
                {'label': 'Colour edges by evaluations', 'value': 'colour_by_evals'},
                {'label': 'Show alt representation with fitness', 'value': 'show_alt_rep'},
                {'label': 'Show alt representation', 'value': 'show_alt_rep_no_fit'},
                {'label': 'Show noisy nodes as squares', 'value': 'noisy-nodes-square'},
                {'label': 'Show noisy path', 'value': 'show_noisy_path'},
                {'label': 'Use viridis for series', 'value': 'use-viridis'},
                {'label': 'Lock algo colours', 'value': 'lock-algo-colours'},
            ],
            value=['noisy-nodes-square', 'use-viridis'],
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        dcc.Dropdown(
            id='stn-node-size-metric',
            options=[
                {'label': 'Generations as representative sol', 'value': 'generations'},
                {'label': 'Evaluations as representative sol', 'value': 'evaluations'},
            ],
            value='evaluations',
            placeholder='Node size',
            style=DROPDOWN_STYLE_WITH_MARGIN
        ),
        html.Hr(),
    ]


def create_lon_options_section():
    """
    Create the LON options section including noisy fitness and colouring options.

    Returns:
        list: List of LON options components.
    """
    return [
        html.Div([
            html.Div(
                html.Label("LON options:", style={'fontWeight': 'bold'}),
            ),
            html.Div([
                html.Label(" Show top '%' of LON nodes: "),
                dcc.Input(
                    id='LON-fit-percent',
                    type='number',
                    min=1,
                    max=100,
                    step=1,
                    value=100
                ),
                dcc.Checklist(
                    id='LON-options',
                    options=[
                        {'label': 'Filter negative', 'value': 'LON-filter-neg'},
                        {'label': 'Hamming distance labels', 'value': 'LON-hamming'},
                        {'label': 'Represent nodes as diamonds', 'value': 'LON-node-diamond'},
                        {'label': 'Display mesh', 'value': 'LON-display-mesh'},
                        {'label': 'Display surface', 'value': 'LON-display-surface'},
                        {'label': 'Use visit proportion for size', 'value': 'LON-visit-size'},
                    ],
                    value=[],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ], style=FLEX_WITH_GAP_STYLE),
            html.Div([
                html.Label("Surface colour: ", style={'verticalAlign': 'middle', 'marginRight': '6px'}),
                dcc.Dropdown(
                    id='LON-surface-colour',
                    options=[
                        {'label': 'Fitness', 'value': 'fitness'},
                        {'label': 'Neighbourhood feasibility', 'value': 'neigh_feas'},
                    ],
                    value='fitness',
                    clearable=False,
                    style={'width': '220px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                ),
            ], style={'marginTop': '6px', 'marginBottom': '4px'}),
            html.Div([
                html.Div(
                    html.Label(" Noisy fitness function: "),
                    style=INLINE_VERTICAL_ALIGN_STYLE
                ),
                html.Div(
                    dcc.Dropdown(
                        id='NLON_fit_func',
                        options=[
                            {'label': 'KP V1 simple', 'value': 'kpv1s'},
                            {'label': 'KP V2 simple', 'value': 'kpv2s'},
                            {'label': 'KP V1 mean(w)', 'value': 'kpv1mw'},
                            {'label': 'KP V2 mean(w)', 'value': 'kpv2mw'},
                            {'label': 'KP Prior (Bit Flip)', 'value': 'kppbf'},
                            {'label': 'KP Prior (Mult Bit Flip)', 'value': 'kppmbf'},
                            {'label': 'KP Prior (p,q) bitwise', 'value': 'kpppqbw'},
                            {'label': 'KP Prior (1,q) bitwise', 'value': 'kpp1qbw'},
                        ],
                        value='',
                        placeholder='Noisy Fit Func',
                        style=DROPDOWN_STYLE
                    ),
                    style=INLINE_DROPDOWN_WRAPPER_STYLE
                ),
                html.Div(
                    html.Label(" Noise intensity: "),
                    style=INLINE_VERTICAL_ALIGN_STYLE
                ),
                html.Div(
                    dcc.Input(
                        id='NLON_intensity',
                        type='number',
                        min=1,
                        max=10,
                        step=1,
                        value=1
                    ),
                    style=INLINE_VERTICAL_ALIGN_NO_MARGIN_STYLE
                ),
                html.Div(
                    html.Label(" Num Samples: "),
                    style=INLINE_VERTICAL_ALIGN_STYLE
                ),
                html.Div(
                    dcc.Input(
                        id='NLON_samples',
                        type='number',
                        min=1,
                        max=500,
                        step=1,
                        value=100
                    ),
                    style=INLINE_VERTICAL_ALIGN_NO_MARGIN_STYLE
                ),
            ], style=FULL_WIDTH_INLINE_STYLE),
        ]),
        html.Div([
            html.Label("LON colouring:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='LON-node-colour-mode',
                options=[
                    {'label': 'Basic', 'value': 'basic'},
                    {'label': 'By fitness (range)', 'value': 'fitness'},
                    {'label': 'By feasibility (0/1)', 'value': 'feasible'},
                    {'label': 'By neighbour feasibility proportion', 'value': 'neigh'},
                ],
                value='fitness',
                placeholder='Node colour mode',
                style={'width': '50%', 'marginTop': '6px'}
            ),
            dcc.Checklist(
                id='LON-edge-colour-feas',
                options=[{'label': 'Colour LON edges by target feasibility', 'value': 'edge_feas'}],
                value=[],
                labelStyle={'display': 'inline-block', 'marginTop': '6px'}
            ),
            html.Div([
                html.Label("LMDS multiplier: ", style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '6px'}),
                dcc.Input(
                    id='lmds-multiplier',
                    type='number',
                    min=0.1,
                    step=0.1,
                    value=1.0,
                    style={'width': '70px', 'display': 'inline-block'}
                ),
            ], style={'marginTop': '6px'}),
        ]),
        html.Hr(),
    ]


def create_plot_options_section():
    """
    Create the general plotting options section.

    Returns:
        list: List of plot options components.
    """
    return [
        dcc.Checklist(
            id='options',
            options=[
                {'label': 'Show Labels', 'value': 'show_labels'},
                {'label': 'Hide STN Nodes', 'value': 'hide_STN_nodes'},
                {'label': 'Hide LON Nodes', 'value': 'hide_LON_nodes'},
                {'label': '3D Plot', 'value': 'plot_3D'},
                {'label': 'Use strength for LON node size', 'value': 'LON_node_strength'},
            ],
            value=['plot_3D', 'LON_node_strength']
        ),
        dcc.Dropdown(
            id='plotType',
            options=[
                {'label': 'RegLon', 'value': 'RegLon'},
                {'label': 'NLon_box', 'value': 'NLon_box'},
                {'label': 'STN', 'value': 'STN'},
            ],
            value='RegLon',
            placeholder='Select plot type',
            style=DROPDOWN_STYLE_WITH_MARGIN
        ),
        dcc.Dropdown(
            id='layout',
            options=[
                {'label': 'Fruchterman Reignold force directed', 'value': 'spring'},
                {'label': 'Kamada Kawai force directed', 'value': 'kamada_kawai'},
                {'label': 'MDS dissimilarity', 'value': 'mds'},
                {'label': 'Random Landmark MDS', 'value': 'r_lmds'},
                {'label': 'FPS Landmark MDS', 'value': 'fps_lmds'},
                {'label': 't-SNE dissimilarity', 'value': 'tsne'},
                {'label': 'raw solution values', 'value': 'raw'},
                {'label': 'Hamming (delta, ref)', 'value': 'hamming_delta_ref'}
            ],
            value='mds',
            placeholder='Select a layout',
            style=DROPDOWN_STYLE_WITH_MARGIN
        ),
        dcc.Dropdown(
            id='hover-info',
            options=[
                {'label': 'Show Fitness', 'value': 'fitness'},
                {'label': 'Show Iterations', 'value': 'iterations'},
                {'label': 'Show solutions', 'value': 'solutions'}
            ],
            value='fitness',
            placeholder='Select hover information',
            style=DROPDOWN_STYLE_WITH_MARGIN
        ),
        # Plot angle options
        html.Label(" Azimuth degrees: "),
        dcc.Input(
            id='azimuth_deg',
            type='number',
            value=35,
            min=0,
            max=360,
            step=1,
            style=SMALL_INPUT_STYLE
        ),
        html.Label(" Elevation degrees: "),
        dcc.Input(
            id='elevation_deg',
            type='number',
            value=60,
            min=0,
            max=90,
            step=1,
            style=SMALL_INPUT_STYLE
        ),
        # Node size inputs
        html.Label("STN Node Min:"),
        dcc.Input(
            id='STN-node-min',
            type='number',
            min=0,
            max=100,
            step=0.01,
            value=5,
            style=SMALL_INPUT_STYLE
        ),
        html.Label("STN Node Max:"),
        dcc.Input(
            id='STN-node-max',
            type='number',
            min=0,
            max=100,
            step=0.01,
            value=20,
            style=SMALL_INPUT_STYLE
        ),
        html.Label("LON Node Min:"),
        dcc.Input(
            id='LON-node-min',
            type='number',
            min=0,
            max=100,
            step=0.000001,
            value=10,
            style=SMALL_INPUT_STYLE
        ),
        html.Label("LON Node Max:"),
        dcc.Input(
            id='LON-node-max',
            type='number',
            min=0,
            max=100,
            step=0.000001,
            value=10.1,
            style=SMALL_INPUT_STYLE
        ),
        html.Br(),
        # Edge size inputs
        html.Label("LON Edge thickness:"),
        dcc.Slider(
            id='LON-edge-size-slider',
            min=1,
            max=100,
            step=1,
            value=5,
            marks={i: str(i) for i in range(1, 100, 10)},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        html.Div([
            html.Label("STN Edge Thickness"),
        ], style=INLINE_BLOCK_WITH_MARGIN),
        html.Div([
            dcc.Input(
                id='STN-edge-size-slider',
                type='number',
                min=0,
                max=100,
                step=1,
                value=5
            ),
        ], style=INLINE_BLOCK_STYLE),
        html.Hr(),
    ]


def create_opacity_options_section():
    """
    Create the opacity options section.

    Returns:
        list: List of opacity options components.
    """
    return [
        html.Label("Opacity options:", style={'fontWeight': 'bold'}),
        html.Div([
            html.Label(" Noise bar opacity: "),
            dcc.Input(
                id='opacity_noise_bar',
                type='number',
                min=0.0,
                max=1.0,
                step=0.1,
                value=1,
                style={'marginRight': '10px'}
            ),
            html.Label(" LON node opacity: "),
            dcc.Input(
                id='LON_node_opacity',
                type='number',
                min=0.0,
                max=1.0,
                step=0.1,
                value=1,
                style={'marginRight': '10px'}
            ),
            html.Label(" LON edge opacity: "),
            dcc.Input(
                id='LON_edge_opacity',
                type='number',
                min=0.0,
                max=1.0,
                step=0.1,
                value=1,
                style={'marginRight': '10px'}
            ),
            html.Label(" STN node opacity: "),
            dcc.Input(
                id='STN_node_opacity',
                type='number',
                min=0.0,
                max=1.0,
                step=0.1,
                value=1,
                style={'marginRight': '10px'}
            ),
            html.Label(" STN edge opacity: "),
            dcc.Input(
                id='STN_edge_opacity',
                type='number',
                min=0.0,
                max=1.0,
                step=0.1,
                value=1,
                style={'marginRight': '10px'}
            ),
        ], style=FLEX_ROW_STYLE),
        html.Hr(),
    ]


def create_axis_options_section():
    """
    Create the axis range adjustment options section.

    Returns:
        list: List of axis options components.
    """
    return [
        html.Label("Axis options:", style={'fontWeight': 'bold'}),
        html.Div([
            html.Label(" x min: "),
            dcc.Input(
                id='custom_x_min',
                type='number',
                value=None,
                style={'marginRight': '10px'}
            ),
            html.Label(" x max: "),
            dcc.Input(
                id='custom_x_max',
                type='number',
                value=None,
            ),
            html.Label(" y min: "),
            dcc.Input(
                id='custom_y_min',
                type='number',
                value=None,
                style={'marginRight': '10px'}
            ),
            html.Label(" y max: "),
            dcc.Input(
                id='custom_y_max',
                type='number',
                value=None,
            ),
            html.Label(" z min: "),
            dcc.Input(
                id='custom_z_min',
                type='number',
                value=None,
                style={'marginRight': '10px'}
            ),
            html.Label(" z max: "),
            dcc.Input(
                id='custom_z_max',
                type='number',
                value=None,
            ),
        ], style=FLEX_ROW_STYLE),
        dcc.Checklist(
            id='log-z-axis',
            options=[{'label': ' Use log for z axis', 'value': 'log_z'}],
            value=[],
            inline=True,
        ),
        html.Hr(),
    ]


def create_annotation_options_section():
    """
    Create the annotation options section with toggles for plot annotations.

    Returns:
        list: List of annotation options components.
    """
    return [
        html.Label("Annotation options:", style={'fontWeight': 'bold'}),
        dcc.Checklist(
            id='annotation-options',
            options=[
                {'label': ' Start nodes', 'value': 'annotate-start-nodes'},
                {'label': ' Optimum', 'value': 'annotate-optimum'},
                {'label': ' End nodes', 'value': 'annotate-end-nodes'},
                {'label': ' Misjudgements', 'value': 'annotate-mistakes'},
                {'label': ' Info panel', 'value': 'annotate-info-panel'},
                {'label': ' Print mode', 'value': 'print-mode'},
                {'label': ' Condense names in print mode', 'value': 'condense-print-names'},
                {'label': ' Show guides', 'value': 'show-guides'},
            ],
            value=['annotate-start-nodes', 'annotate-optimum', 'annotate-end-nodes', 'annotate-mistakes', 'annotate-info-panel'],
            inline=True,
        ),
        html.Div([
            html.Label("Info x pos (%):"),
            dcc.Input(
                id='info-panel-x',
                type='number',
                value=90,
                min=0,
                max=100,
                style={'width': '60px', 'marginLeft': '5px', 'marginRight': '20px'},
            ),
            html.Label("Info y pos (%):"),
            dcc.Input(
                id='info-panel-y',
                type='number',
                value=75,
                min=0,
                max=100,
                style={'width': '60px', 'marginLeft': '5px', 'marginRight': '20px'},
            ),
            html.Label("Axes text scale:"),
            dcc.Input(
                id='axes-text-scale',
                type='number',
                value=1.0,
                min=0.1,
                step=0.1,
                style={'width': '60px', 'marginLeft': '5px', 'marginRight': '20px'},
            ),
            html.Label("Annotation text scale:"),
            dcc.Input(
                id='annotation-text-scale',
                type='number',
                value=1.0,
                min=0.1,
                step=0.1,
                style={'width': '60px', 'marginLeft': '5px'},
            ),
        ], style={'marginTop': '6px'}),
        html.Hr(),
    ]


def create_main_plot_section():
    """
    Create the main LON/STN plot and info section.

    Returns:
        list: List of main plot components.
    """
    return [
        dcc.Graph(id='trajectory-plot'),
        html.H3("STN Stats"),
        html.Div(id='stn-stats-table'),
        html.H3("LON Stats"),
        html.Div(id='lon-stats-table'),
        html.Div(id="print_STN_series_labels", style=SELECTION_OUTPUT_STYLE),
        html.H3("Plot Information"),
        dcc.Checklist(
            id='show-text-info',
            options=[{'label': 'Show text info', 'value': 'show'}],
            value=['show'],
        ),
        html.Div(id='run-print-info', style=MONOSPACE_STYLE),
    ]
