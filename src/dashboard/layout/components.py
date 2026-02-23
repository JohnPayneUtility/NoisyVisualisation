"""
Reusable UI component functions for Dashboard layout.

Each function returns a Dash component or group of components
that can be assembled in the main layout.
"""
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


def create_problem_selection_tabs():
    """
    Create the tabbed section for problem selection.

    Returns:
        html.Div: Problem selection tabs container.
    """
    return html.Div([
        dcc.Tabs(id='problemTabSelection', value='p1', children=[
            dcc.Tab(
                label='Select problem',
                value='p1',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
            dcc.Tab(
                label='Select additional problem (optional)',
                value='p2',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE
            ),
        ]),
        html.Div(id='problemTabsContent'),
    ], style=SECTION_STYLE)


def create_2d_plot_tabs():
    """
    Create the tabbed section for 2D performance plots.

    Returns:
        html.Div: 2D plot tabs container.
    """
    return html.Div([
        html.H3("2D Performance Plotting"),
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
                    {'label': 'Prior noise STN plot', 'value': 'prior'},
                    {'label': 'Prior noise STN V2 plot', 'value': 'prior_v2'},
                    {'label': 'Prior noise STN V3 plot', 'value': 'prior_v3'},
                    {'label': 'Prior noise STN V4 plot', 'value': 'prior_v4'},
                    {'label': 'Prior noise STN V5 plot', 'value': 'prior_v5'},
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
                {'label': 'Show fitness box plots', 'value': 'show_stn_boxplots'},
            ],
            value=[],
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        dcc.Dropdown(
            id='stn-node-size-metric',
            options=[
                {'label': 'Generations as representative sol', 'value': 'generations'},
                {'label': 'Evaluations as representative sol', 'value': 'evaluations'},
            ],
            value='generations',
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
                    ],
                    value=[],
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ], style=FLEX_WITH_GAP_STYLE),
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
                            {'label': 'KP Prior', 'value': 'kpp'},
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
                {'label': 'raw solution values', 'value': 'raw'}
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
        html.Div(id="print_STN_series_labels", style=SELECTION_OUTPUT_STYLE),
        html.Div(id='run-print-info', style=MONOSPACE_STYLE),
    ]
