"""The shared-graph visualisation: the STN/LON trajectory plot and its print-mode, axis and
series-label controls.

`update_plot` is the single orchestrator over one `nx.MultiDiGraph` (plan I-10c): STN and/or LON
population, styling, LON statistics and guide nodes, then one positioning pass, then presentation.
"""

import dash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # for continuous color scales
import networkx as nx
from dash import html, dash_table, Input, Output
from dataclasses import replace as dataclass_replace

from ..instance import app
from ..helpers import (
    convert_to_single_edges_format,
    filter_local_optima,
    filter_negative_LO,
    get_mean_run,
    get_median_run,
    select_top_runs_by_fitness,
    _get_noise_param_label,
)
from ...viz import (
    parse_callback_inputs,
    generate_run_summary_string,
    add_stn_trajectories,
    add_mo_fronts,
    add_prior_noise_stn_v4,
    add_prior_noise_stn_v5,
    add_prior_noise_stn_algo_pov,
    add_lon_nodes,
    add_lon_edges,
    debug_mo_counts,
    style_nodes,
    calculate_positions,
    LON_SCATTER_AXIS_LABELS,
    LON_SCATTER_DEFAULT_X_AXIS,
    LON_SCATTER_DEFAULT_Y_AXIS,
    LON_SCATTER_DEFAULT_PLOT_STYLE,
    plot_lon_stats,
    plot_lon_stats_multi,
    build_all_traces,
    create_guide_traces,
    create_axis_settings,
    create_figure,
)
from ...analysis.graph_stats import (
    calculate_lon_statistics,
    compute_node_feasibility_error,
    compute_pairwise_correlations,
    compute_correlation_pair,
)
from ...analysis.misjudgements import (
    increasing_noise_step_indices,
    comparison_misjudgement_step_indices,
    constraint_misjudgement_step_indices,
)
from ..components import (
    build_correlation_table,
    build_selected_correlation_display,
)
from ...common import is_continuous_solution


def _add_guide_nodes(G: nx.MultiDiGraph) -> None:
    """Add binary guide nodes to G before position calculation.

    Skipped silently for continuous problems or when G has no solutions.
    Series 1: all-zeros and all-ones.
    Series 2: one bit set per node, left to right (bit 0 first).
    Series 3: increasing Hamming weight from the right (bit n-1 first),
              so the first node of series 3 always differs from series 2.
    """
    n = None
    for _, attr in G.nodes(data=True):
        sol = attr.get('solution')
        if sol:
            if is_continuous_solution(sol):
                return
            n = len(sol)
            break
    if not n:
        return

    G.add_node('Guide_S1_zeros', type='guide', guide_series=1, fitness=0, solution=[0] * n)
    G.add_node('Guide_S1_ones',  type='guide', guide_series=1, fitness=0, solution=[1] * n)

    for k in range(n):
        sol = [1 if i == k else 0 for i in range(n)]
        G.add_node(f'Guide_S2_bit_{k}', type='guide', guide_series=2, fitness=0, solution=sol)

    for k in range(1, n):
        sol = [0] * (n - k) + [1] * k
        G.add_node(f'Guide_S3_hw_{k}', type='guide', guide_series=3, fitness=0, solution=sol)


@app.callback(
    Output('run-print-info', 'style'),
    Input('show-text-info', 'value')
)
def toggle_run_print_info(value):
    from ..layout.styles import MONOSPACE_STYLE
    if value and 'show' in value:
        return MONOSPACE_STYLE
    return {**MONOSPACE_STYLE, 'display': 'none'}


@app.callback(
    Output('print_STN_series_labels', "children"),
    Input('STN_series_labels', 'data')
)
def update_plotted_series_labels(series_list):
    if not series_list:
        return "No rows selected in Table 2."
    # series_labels = [series_list[i] for i in series_list]
    return f"Plotted series: {series_list}"

# ==========
# main plot callbacks
# ==========

# callback for custom axis values
@app.callback(
    Output('axis-values', 'data'),
    [Input('custom_x_min', 'value'),
     Input('custom_x_max', 'value'),
     Input('custom_y_min', 'value'),
     Input('custom_y_max', 'value'),
     Input('custom_z_min', 'value'),
     Input('custom_z_max', 'value'),
     Input('log-z-axis', 'value')]
)
def clean_axis_values(custom_x_min, custom_x_max, custom_y_min, custom_y_max, custom_z_min, custom_z_max, log_z):
    def clean(val):
        # If the input is empty, an empty string, or None, return None.
        # Otherwise, assume it's a valid number (or you could cast to float/int).
        if val in [None, ""]:
            return None
        return val  # or float(val) if needed
    
    return {
        "custom_x_min": clean(custom_x_min),
        "custom_x_max": clean(custom_x_max),
        "custom_y_min": clean(custom_y_min),
        "custom_y_max": clean(custom_y_max),
        "custom_z_min": clean(custom_z_min),
        "custom_z_max": clean(custom_z_max),
        "log_z": 'log_z' in (log_z or []),
    }

def _build_stn_stats_table(stn_algo_data, stn_labels, problem_goal='maximise', noise_param_label='noise'):
    """Build the STN stats DataTable from per-algorithm trajectory data."""
    if not stn_algo_data:
        return html.Div()

    def fmt(values):
        if not values:
            return 'N/A'
        if len(values) == 1:
            return str(values[0]) if values[0] is not None else 'N/A'
        return '[' + ','.join(str(v) if v is not None else 'N/A' for v in values) + ']'

    minimising = str(problem_goal or 'maximise')[:3].lower() == 'min'

    def count_misjudgements(entry):
        fits = entry[1] if entry and len(entry) > 1 and entry[1] is not None else []
        if minimising:
            return sum(1 for i in range(len(fits) - 1) if fits[i + 1] > fits[i])
        return sum(1 for i in range(len(fits) - 1) if fits[i + 1] < fits[i])

    rows = []
    for idx, selected_trajectories, all_run_trajectories in stn_algo_data:
        label = stn_labels[idx] if stn_labels and idx < len(stn_labels) else [f'Algo {idx}']
        algo_name = label[0] if len(label) > 0 else f'Algo {idx}'
        noise = label[1] if len(label) > 1 else '?'
        algo_name = f'{algo_name} {noise_param_label}={noise}'

        nodes_per_run = [
            len(entry[0]) for entry in selected_trajectories
            if entry and len(entry) > 0 and entry[0] is not None
        ]

        all_node_counts = [
            len(entry[0]) for entry in all_run_trajectories
            if entry and len(entry) > 0 and entry[0] is not None
        ]
        mean_nodes = round(sum(all_node_counts) / len(all_node_counts)) if all_node_counts else 'N/A'

        avg_evals_per_run = []
        for entry in selected_trajectories:
            sol_evals = entry[12] if entry and len(entry) > 12 else None
            if sol_evals:
                avg_evals_per_run.append(round(sum(sol_evals) / len(sol_evals)))
            else:
                avg_evals_per_run.append(None)

        all_evals_flat = [
            e for entry in all_run_trajectories
            if entry and len(entry) > 12 and entry[12]
            for e in entry[12]
        ]
        mean_avg_evals = round(sum(all_evals_flat) / len(all_evals_flat)) if all_evals_flat else 'N/A'

        misjudgements_per_run = [count_misjudgements(e) for e in selected_trajectories]
        all_misjudgement_counts = [count_misjudgements(e) for e in all_run_trajectories]
        mean_misjudgements = round(sum(all_misjudgement_counts) / len(all_misjudgement_counts)) \
            if all_misjudgement_counts else 'N/A'

        rows.append({
            'algo': algo_name,
            'nodes_rendered': fmt(nodes_per_run),
            'mean_nodes_all': str(mean_nodes),
            'avg_evals_rendered': fmt(avg_evals_per_run),
            'mean_avg_evals_all': str(mean_avg_evals),
            'misjudgements_rendered': fmt(misjudgements_per_run),
            'mean_misjudgements_all': str(mean_misjudgements),
        })

    if not rows:
        return html.Div()

    return dash_table.DataTable(
        columns=[
            {'name': 'Algorithm', 'id': 'algo'},
            {'name': 'Nodes (rendered runs)', 'id': 'nodes_rendered'},
            {'name': 'Mean nodes (all runs)', 'id': 'mean_nodes_all'},
            {'name': 'Avg evals/node (rendered runs)', 'id': 'avg_evals_rendered'},
            {'name': 'Avg evals/node (all runs)', 'id': 'mean_avg_evals_all'},
            {'name': 'Misjudgements (rendered runs)', 'id': 'misjudgements_rendered'},
            {'name': 'Mean misjudgements (all runs)', 'id': 'mean_misjudgements_all'},
        ],
        data=rows,
        style_table={'width': '1300px'},
        style_cell={'textAlign': 'center', 'padding': '8px'},
        style_header={'fontWeight': 'bold'},
    )


# print mode: uncheck info panel, set scale defaults
@app.callback(
    [Output('annotation-options', 'value'),
     Output('axes-text-scale', 'value'),
     Output('annotation-text-scale', 'value')],
    Input('annotation-options', 'value'),
    prevent_initial_call=True,
)
def handle_print_mode(annotation_options):
    options = annotation_options or []
    if 'print-mode' in options:
        options = [o for o in options if o != 'annotate-info-panel']
        return options, 1.2, 2
    return options, dash.no_update, dash.no_update


# callback for main plot
@app.callback(
    [Output('trajectory-plot', 'figure'),
     Output('run-print-info', 'children'),
     Output('stn-stats-table', 'children'),
     Output('lon-stats-table', 'children'),
     Output('lon-feas-error-scatter', 'figure'),
     Output('lon-selected-correlation', 'children'),
     Output('lon-feas-error-correlations', 'children')],
    [Input("optimum", "data"),
     Input("PID", "data"),
     Input("opt_goal", "data"),
     Input('options', 'value'),
     Input('run-options', 'value'),
     Input('STN_lower_fit_limit', 'value'),
     Input('LON-fit-percent', 'value'),
     Input('LON-options', 'value'),
     Input('LON-node-colour-mode', 'value'), # CoLON colour
     Input('LON-surface-colour', 'value'),
     Input('LON-edge-colour-feas', 'value'), # CoLON colour
     Input('lmds-multiplier', 'value'),
     Input('NLON_fit_func', 'value'),
     Input('NLON_intensity', 'value'),
     Input('NLON_samples', 'value'),
     Input('NLON_penalty', 'value'),
     Input('layout', 'value'),
     Input('plotType', 'value'),
     Input('hover-info', 'value'),
     Input('azimuth_deg', 'value'),
     Input('elevation_deg', 'value'),
     Input('STN_data_processed', 'data'),
     Input('STN_series_labels', 'data'),
     Input('run-index', 'value'),
     Input('run-selector', 'value'),
     Input('LON_data', 'data'),
     Input('axis-values', 'data'),
     Input('opacity_noise_bar', 'value'),
     Input('LON_node_opacity', 'value'),
     Input('LON_edge_opacity', 'value'),
     Input('STN_node_opacity', 'value'),
     Input('STN_edge_opacity', 'value'),
     Input('STN-node-min', 'value'),
     Input('STN-node-max', 'value'),
     Input('LON-node-min', 'value'),
     Input('LON-node-max', 'value'),
     Input('LON-edge-size-slider', 'value'),
     Input('STN-edge-size-slider', 'value'),
     Input('noisy_fitnesses_data', 'data'),
     Input('stn-plot-type', 'value'),
     Input('STN_MO_data', 'data'),
     Input('STN_MO_series_labels', 'data'),
     Input('stn-node-size-metric', 'value'),
     Input('annotation-options', 'value'),
     Input('fit_func_store', 'data'),
     Input('info-panel-x', 'value'),
     Input('info-panel-y', 'value'),
     Input('axes-text-scale', 'value'),
     Input('annotation-text-scale', 'value'),
     Input('plot-theme', 'value'),
     Input('plot_2d_data', 'data'),
     Input('lon-scatter-x-axis', 'value'),
     Input('lon-scatter-y-axis', 'value'),
     Input('lon-scatter-plot-style', 'value'),
     Input('lon-scatter-multi-noise', 'value')]
)
def update_plot(optimum, PID, opt_goal, options, run_options, STN_lower_fit_limit,
                LO_fit_percent, LON_options, LON_node_colour_mode, LON_surface_colour, LON_edge_colour_feas,
                lmds_multiplier, NLON_fit_func, NLON_intensity, NLON_samples, NLON_penalty, layout_value, plot_type,
                hover_info_value, azimuth_deg, elevation_deg, all_trajectories_list, STN_labels,
                run_start_index, n_runs_display, local_optima, axis_values,
                opacity_noise_bar, LON_node_opacity, LON_edge_opacity, STN_node_opacity, STN_edge_opacity,
                STN_node_min, STN_node_max, LON_node_min, LON_node_max,
                LON_edge_size_slider, STN_edge_size_slider, noisy_fitnesses_list,
                stn_plot_type, STN_MO_data, STN_MO_series_labels, stn_node_size_metric,
                annotation_options, fit_func, info_panel_x, info_panel_y,
                axes_text_scale, annotation_text_scale, plot_theme, plot_2d_data,
                lon_scatter_x, lon_scatter_y, lon_scatter_plot_style, lon_scatter_multi_noise):
    """
    Main visualization callback - orchestrates the visualization pipeline.

    This refactored callback delegates to specialized modules in src/visualization/
    for graph building, node styling, positioning, and trace creation.
    """
    print('\033[1m\033[31mCreating new Plot...\033[0m', flush=True)

    # ==========
    # STEP 1: Parse all callback inputs into configuration object
    # ==========
    config = parse_callback_inputs(
        optimum=optimum,
        pid=PID,
        opt_goal=opt_goal,
        options=options or [],
        run_options=run_options or [],
        stn_lower_fit_limit=STN_lower_fit_limit,
        lo_fit_percent=LO_fit_percent,
        lon_options=LON_options or [],
        lon_node_colour_mode=LON_node_colour_mode,
        lon_surface_colour=LON_surface_colour,
        lon_edge_colour_feas=LON_edge_colour_feas or [],
        lmds_multiplier=lmds_multiplier,
        nlon_fit_func=NLON_fit_func,
        nlon_intensity=NLON_intensity,
        nlon_samples=NLON_samples,
        nlon_penalty=NLON_penalty,
        layout_value=layout_value,
        plot_type=plot_type,
        hover_info_value=hover_info_value,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        run_start_index=run_start_index,
        n_runs_display=n_runs_display,
        axis_values=axis_values or {},
        opacity_noise_bar=opacity_noise_bar,
        lon_node_opacity=LON_node_opacity,
        lon_edge_opacity=LON_edge_opacity,
        stn_node_opacity=STN_node_opacity,
        stn_edge_opacity=STN_edge_opacity,
        stn_node_min=STN_node_min,
        stn_node_max=STN_node_max,
        lon_node_min=LON_node_min,
        lon_node_max=LON_node_max,
        lon_edge_size_slider=LON_edge_size_slider,
        stn_edge_size_slider=STN_edge_size_slider,
        stn_plot_type=stn_plot_type,
        node_size_metric=stn_node_size_metric or 'generations',
        colorscale=plot_theme or 'Viridis',
    )

    # Apply viridis color palette for series if enabled
    if config.stn.use_viridis:
        n_series = len(STN_labels or STN_MO_series_labels or [])
        if n_series > 0:
            positions = [i / max(n_series - 1, 1) for i in range(n_series)]
            config.algo_colors = [px.colors.sample_colorscale(config.colorscale, p)[0] for p in positions]

    # Lock colours by algorithm name (matching the 2D performance plot color mapping)
    if 'lock-algo-colours' in (run_options or []) and plot_2d_data:
        all_algo_df = pd.DataFrame(plot_2d_data)
        if 'algo_name' in all_algo_df.columns:
            all_algos = sorted(all_algo_df['algo_name'].dropna().unique().tolist())
            n_all = len(all_algos)
            if n_all > 0:
                positions = [i / max(n_all - 1, 1) for i in range(n_all)] if n_all > 1 else [0.5]
                all_colors = px.colors.sample_colorscale(config.colorscale, positions)
                algo_color_map = dict(zip(all_algos, all_colors))
                labels = STN_labels or STN_MO_series_labels or []
                config.algo_colors = [
                    algo_color_map.get(lbl[0] if isinstance(lbl, (list, tuple)) else lbl,
                                       config.algo_colors[i % len(config.algo_colors)])
                    for i, lbl in enumerate(labels)
                ]

    # Resolve noise parameter label once (used in info panel and stats table)
    annotation_opts_early = annotation_options or []
    noise_param = (
        _get_noise_param_label(fit_func)
        if 'problem-specific-noise-label' in annotation_opts_early
        else 'noise'
    )

    # ==========
    # STEP 2: Initialize graph and node mappings
    # ==========
    G = nx.MultiDiGraph()
    stn_node_mapping = {}
    lon_node_mapping = {}
    debug_summaries = []
    stn_algo_data = []
    node_noise = {}
    fitness_dict = {}

    # ==========
    # STEP 3: Build graph - add nodes and edges
    # ==========

    # If 'evaluations' node size metric is selected, swap sol_iterations (index 3)
    # with sol_iterations_evals (index 12) in each run entry before graph building.
    if config.node_size_metric == 'evaluations' and all_trajectories_list:
        for series_entries in all_trajectories_list:
            for entry in series_entries:
                if len(entry) > 12 and entry[12]:
                    entry[3] = entry[12]

    if config.stn_plot_type == 'multiobjective':
        # Multi-objective mode
        print('ADDING NODES IN MULTIOBJECTIVE MODE')
        for idx, mo_runs in enumerate(STN_MO_data or []):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]
            selected_runs = []
            if config.n_runs_display > 0:
                selected_runs.extend(mo_runs[config.run_start_index:config.run_start_index + config.n_runs_display])
            add_mo_fronts(G, selected_runs, edge_color, idx, config.noisy_node_color)

        debug_summary_component = html.Div("None implemented for MO")

    elif config.stn_plot_type == 'prior_v4' and all_trajectories_list:
        # Prior noise STN V4 mode
        print('ADDING NODES IN PRIOR NOISE STN V4 MODE')
        optimisation_goal = opt_goal[:3].lower() if opt_goal else 'max'

        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]

            selected_trajectories = []
            if config.n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[config.run_start_index:config.run_start_index + config.n_runs_display])
            if config.show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if config.show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if config.show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if config.show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            add_prior_noise_stn_v4(
                G, selected_trajectories, edge_color, idx, config.noisy_node_color,
                dedup=config.stn.dedup_prior_noise,
                show_alt_rep=config.stn.show_alt_rep,
                show_alt_rep_no_fit=config.stn.show_alt_rep_no_fit,
                stn_node_min=config.node_size.stn_min,
                use_est_discarded_as_base=config.use_est_discarded_as_base,
            )

            stn_algo_data.append((idx, selected_trajectories, all_run_trajectories))
            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)

    elif config.stn_plot_type == 'prior_v5' and all_trajectories_list:
        # Prior noise STN V5 mode
        print('ADDING NODES IN PRIOR NOISE STN V5 MODE')
        optimisation_goal = opt_goal[:3].lower() if opt_goal else 'max'

        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]

            selected_trajectories = []
            if config.n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[config.run_start_index:config.run_start_index + config.n_runs_display])
            if config.show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if config.show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if config.show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if config.show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            add_prior_noise_stn_v5(
                G, selected_trajectories, edge_color, idx, config.noisy_node_color,
                dedup=config.stn.dedup_prior_noise,
                show_alt_rep=config.stn.show_alt_rep,
                show_alt_rep_no_fit=config.stn.show_alt_rep_no_fit,
                stn_node_min=config.node_size.stn_min,
                use_est_discarded_as_base=config.use_est_discarded_as_base,
            )

            stn_algo_data.append((idx, selected_trajectories, all_run_trajectories))
            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)

    elif config.stn_plot_type == 'prior_algo_pov' and all_trajectories_list:
        # Prior noise STN algo POV mode
        print('ADDING NODES IN PRIOR NOISE STN ALGO POV MODE')
        optimisation_goal = opt_goal[:3].lower() if opt_goal else 'max'

        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]

            selected_trajectories = []
            if config.n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[config.run_start_index:config.run_start_index + config.n_runs_display])
            if config.show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if config.show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if config.show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if config.show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            add_prior_noise_stn_algo_pov(
                G, selected_trajectories, edge_color, idx, config.noisy_node_color,
                dedup=config.stn.dedup_prior_noise,
                show_alt_rep=config.stn.show_alt_rep,
                show_alt_rep_no_fit=config.stn.show_alt_rep_no_fit,
                stn_node_min=config.node_size.stn_min,
                use_est_discarded_as_base=config.use_est_discarded_as_base,
            )

            stn_algo_data.append((idx, selected_trajectories, all_run_trajectories))
            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)

    elif all_trajectories_list:
        # Single-objective STN mode
        optimisation_goal = opt_goal[:3].lower() if opt_goal else 'max'

        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = config.algo_colors[idx % len(config.algo_colors)]

            selected_trajectories = []
            if config.n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[config.run_start_index:config.run_start_index + config.n_runs_display])
            if config.show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if config.show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if config.show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if config.show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            stn_node_mapping = add_stn_trajectories(
                G, selected_trajectories, edge_color, idx, stn_node_mapping, config
            )

            stn_algo_data.append((idx, selected_trajectories, all_run_trajectories))
            summary_str = generate_run_summary_string(selected_trajectories)
            debug_summaries.append((summary_str, edge_color))

        # Create debug summary component
        summary_components = []
        for summary_str, color in debug_summaries:
            summary_components.append(
                html.Div(summary_str, style={'color': color, 'whiteSpace': 'pre-wrap', 'marginBottom': '10px'})
            )
        debug_summary_component = html.Div(summary_components)
    else:
        debug_summary_component = html.Div("No trajectory data available.")

    stn_stats_table = _build_stn_stats_table(stn_algo_data, STN_labels, problem_goal=opt_goal, noise_param_label=noise_param)

    print('STN TRAJECTORIES ADDED')
    debug_mo_counts(G, by="run_idx", label="[MO]", list_fronts=True, max_list=50)

    # ==========
    # STEP 4: Add LON nodes and edges (if provided)
    # ==========
    opt_feas_map = {}
    neigh_feas_map = {}
    visit_prop_map = {}

    if local_optima:
        # Extract CoLON colour maps
        if isinstance(local_optima, dict):
            opt_feas_map = local_optima.get("opt_feas_map", {}) or {}
            neigh_feas_map = local_optima.get("neigh_feas_map", {}) or {}
            visit_prop_map = local_optima.get("visit_prop_map", {}) or {}

        # Convert and filter local optima data
        local_optima_processed = convert_to_single_edges_format(local_optima)
        local_optima_processed = filter_local_optima(local_optima_processed, config.lon.fit_percent)
        if config.lon.filter_negative:
            local_optima_processed = filter_negative_LO(local_optima_processed)

        # Add LON nodes
        lon_node_mapping, node_noise = add_lon_nodes(
            G, local_optima_processed, lon_node_mapping, config, PID
        )
        fitness_dict = {node: data['fitness'] for node, data in G.nodes(data=True)}

        # Add LON edges
        add_lon_edges(G, local_optima_processed, lon_node_mapping, config, opt_feas_map)
        print('LOCAL OPTIMA ADDED')

    # ==========
    # STEP 5: Apply node styling (sizes and colors)
    # ==========
    style_nodes(G, config, opt_feas_map, neigh_feas_map, visit_prop_map)

    # ==========
    # STEP 6: Calculate statistics
    # ==========
    calculate_lon_statistics(G, verbose=True)

    # ==========
    # STEP 6.5: Inject binary guide nodes before layout (so they're placed naturally)
    # ==========
    if 'show-guides' in (annotation_options or []):
        _add_guide_nodes(G)

    # ==========
    # STEP 7: Calculate node positions
    # ==========
    pos = calculate_positions(G, config.layout_type, config.stn_plot_type, config.plot_3d, config.lon.lmds_multiplier)

    # ==========
    # STEP 8: Build visualization traces
    # ==========
    if config.plot_type in ('RegLon', 'NLon_box', 'NLon_IQR'):
        print('CREATING PLOT...')
        traces = build_all_traces(G, pos, config, node_noise, fitness_dict, neigh_feas_map)

        if 'show-guides' in (annotation_options or []):
            traces.extend(create_guide_traces(G, pos))

        # ==========
        # STEP 9: Configure axes and create figure
        # ==========
        xaxis_settings, yaxis_settings, zaxis_settings = create_axis_settings(
            G, pos, config, node_noise, axes_text_scale=axes_text_scale or 1.0)

        # Build scene annotations from enabled annotation options
        ann_font_size = round(11 * (annotation_text_scale or 1.0))
        log_z = config.axis.log_z
        def ann_z(z):
            import math
            if log_z and z is not None and z > 0:
                return math.log10(z)
            return z
        scene_annotations = []
        if 'annotate-start-nodes' in (annotation_options or []):
            if 'single-start-node' in (annotation_options or []):
                for node, attr in G.nodes(data=True):
                    if attr.get('start_node') and attr.get('series_idx', 0) == 0 and node in pos:
                        x, y = pos[node][:2]
                        z = attr.get('fitness', 0)
                        scene_annotations.append(dict(
                            x=x, y=y, z=ann_z(z),
                            text='Start node',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='black',
                            ax=80, ay=0,
                            font=dict(size=ann_font_size, color='black'),
                        ))
                        break
            else:
                seen_positions = set()
                for node, attr in G.nodes(data=True):
                    if attr.get('start_node') and node in pos:
                        x, y = pos[node][:2]
                        z = attr.get('fitness', 0)
                        pos_key = (round(x, 6), round(y, 6), round(z, 6))
                        if pos_key not in seen_positions:
                            seen_positions.add(pos_key)
                            scene_annotations.append(dict(
                                x=x, y=y, z=ann_z(z),
                                text='Start node',
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=1.5,
                                arrowcolor='black',
                                ax=80, ay=0,
                                font=dict(size=ann_font_size, color='black'),
                            ))

        if 'annotate-end-nodes' in (annotation_options or []):
            seen_positions = set()
            for node, attr in G.nodes(data=True):
                if attr.get('end_node') and node in pos:
                    if config.optimum is not None and attr.get('fitness') == config.optimum:
                        continue
                    x, y = pos[node][:2]
                    z = attr.get('fitness', 0)
                    pos_key = (round(x, 6), round(y, 6), round(z, 6))
                    if pos_key not in seen_positions:
                        seen_positions.add(pos_key)
                        scene_annotations.append(dict(
                            x=x, y=y, z=ann_z(z),
                            text='End node',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='black',
                            ax=80, ay=0,
                            font=dict(size=ann_font_size, color='black'),
                        ))

        if 'annotate-mistakes' in (annotation_options or []):
            seen_positions = set()
            maximizing = (config.opt_goal or 'max')[:3].lower() == 'max'
            algo_pov = config.stn_plot_type == 'prior_algo_pov'
            for u, v, edge_attr in G.edges(data=True):
                if not edge_attr.get('edge_type', '').startswith('STN'):
                    continue
                if algo_pov:
                    # Base nodes are named _Noisy; compare satellite (_True) fitness
                    if v not in pos or G.nodes[v].get('type') == 'STN_ALT':
                        continue
                    u_true = u.replace('_Noisy', '_True')
                    v_true = v.replace('_Noisy', '_True')
                    fit_u = G.nodes[u_true].get('fitness') if u_true in G.nodes else None
                    fit_v = G.nodes[v_true].get('fitness') if v_true in G.nodes else None
                else:
                    if v not in pos or 'Noisy' in v or G.nodes[v].get('type') == 'STN_ALT':
                        continue
                    fit_u = G.nodes[u].get('fitness')
                    fit_v = G.nodes[v].get('fitness')
                if fit_u is None or fit_v is None:
                    continue
                is_decline = fit_v < fit_u if maximizing else fit_v > fit_u
                if is_decline:
                    x, y = pos[v][:2]
                    z = G.nodes[v].get('fitness') if algo_pov else fit_v
                    pos_key = (round(x, 6), round(y, 6), round(z, 6))
                    if pos_key not in seen_positions:
                        seen_positions.add(pos_key)
                        scene_annotations.append(dict(
                            x=x, y=y, z=ann_z(z),
                            text='',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='red',
                            ax=20, ay=0,
                            font=dict(size=ann_font_size, color='red'),
                        ))

        if 'annotate-advanced-mistakes' in (annotation_options or []):
            minimising = (config.opt_goal or 'max')[:3].lower() == 'min'
            MISTAKE_TYPES = [
                ('increasing_noise', 'blue', 1),
                ('comparison', 'red', 2),
                ('constraint', 'magenta', 3),
            ]
            hits = {key: set() for key, _, _ in MISTAKE_TYPES}  # dedup by node_label per type

            # Node naming differs by STN plot type. 'prior_v4'/'prior_v5' dedicate one
            # deterministically-named node per (algo, run, step) with the _True suffix
            # holding true fitness; 'prior_algo_pov' uses the same scheme but the base
            # node (_Noisy suffix) holds the noisy fitness (matches the convention the
            # existing 'annotate-mistakes' arrows already use for this view). Other
            # modes (default 'posterior', 'posterior_algo_pov') dedupe nodes by solution
            # via stn_node_mapping (built in add_stn_trajectories).
            stn_plot_type = config.stn_plot_type

            for algo_idx, selected_trajectories_adv, _ in stn_algo_data:
                for run_idx, entry in enumerate(selected_trajectories_adv):
                    if not entry or len(entry) < 3 or entry[0] is None:
                        continue
                    unique_solutions, true_fits, noisy_fits = entry[0], entry[1], entry[2]

                    def node_for(step_idx):
                        if stn_plot_type in ('prior_v4', 'prior_v5'):
                            label = f"STN_S{algo_idx}_R{run_idx}_Sol{step_idx}_True"
                        elif stn_plot_type == 'prior_algo_pov':
                            label = f"STN_S{algo_idx}_R{run_idx}_Sol{step_idx}_Noisy"
                        else:
                            label = stn_node_mapping.get((tuple(unique_solutions[step_idx]), "STN"))
                        return label if label and label in pos else None

                    for step_idx in increasing_noise_step_indices(true_fits, noisy_fits):
                        node = node_for(step_idx)
                        if node:
                            hits['increasing_noise'].add(node)
                    for step_idx in comparison_misjudgement_step_indices(true_fits, noisy_fits, minimising):
                        node = node_for(step_idx)
                        if node:
                            hits['comparison'].add(node)
                    for step_idx in constraint_misjudgement_step_indices(true_fits):
                        node = node_for(step_idx)
                        if node:
                            hits['constraint'].add(node)

            # Scale the stacking offset to the plot's actual fitness range so dots
            # are visible and proportionate regardless of the problem's fitness scale.
            all_z = [ann_z(attr.get('fitness')) for _, attr in G.nodes(data=True) if attr.get('fitness') is not None]
            z_span = (max(all_z) - min(all_z)) if len(all_z) >= 2 else 1.0
            offset_unit = (z_span or 1.0) * 0.03

            for key, color, slot in MISTAKE_TYPES:
                xs, ys, zs = [], [], []
                for node in hits[key]:
                    x, y = pos[node][:2]
                    z = ann_z(G.nodes[node].get('fitness', 0))
                    xs.append(x); ys.append(y); zs.append(z + slot * offset_unit)
                if xs:
                    traces.append(go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        mode='markers',
                        marker=dict(size=4, color=color, symbol='circle'),
                        name=f'Advanced misjudgement: {key.replace("_", " ")}',
                        showlegend=True,
                    ))

        if 'annotate-optimum' in (annotation_options or []) and config.optimum is not None:
            for node, attr in G.nodes(data=True):
                if attr.get('fitness') == config.optimum and 'Noisy' not in node and node in pos:
                    x, y = pos[node][:2]
                    z = attr.get('fitness', 0)
                    scene_annotations.append(dict(
                        x=x, y=y, z=ann_z(z),
                        text='Global optimum',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor='black',
                        ax=40, ay=-40,
                        font=dict(size=ann_font_size, color='black'),
                    ))
                    break  # Only one global optimum

        fig = create_figure(traces, config, xaxis_settings, yaxis_settings, zaxis_settings,
                            scene_annotations=scene_annotations or None)

        # Add 2D info panel annotation in top-right corner
        annotation_opts = annotation_options or []
        print_mode = 'print-mode' in annotation_opts
        if 'annotate-info-panel' in annotation_opts or print_mode:
            lines = []
            if not print_mode:
                if PID:
                    lines.append(f'<b>PID:</b> {PID}')
                if fit_func:
                    lines.append(f'<b>Fitness:</b> {fit_func}')
                    lines.append('&#9679; True solution/fitness')
                    lines.append('&#9632; Noisy solution/fitness')
                    lines.append('<i>Node size = evals at node</i>')

            if STN_labels:
                if lines:
                    lines.append('')  # blank line separator
                condense = print_mode and 'condense-print-names' in annotation_opts
                if condense:
                    algo_names = [label[0] if len(label) > 0 else '?' for label in STN_labels]
                    noise_vals = [str(label[1]) if len(label) > 1 else '?' for label in STN_labels]
                    all_same_algo = len(set(algo_names)) == 1
                    all_same_noise = len(set(noise_vals)) == 1
                else:
                    all_same_algo = all_same_noise = False
                for idx, label in enumerate(STN_labels):
                    algo_name = label[0] if len(label) > 0 else '?'
                    noise = label[1] if len(label) > 1 else '?'
                    color = config.algo_colors[idx % len(config.algo_colors)]
                    if condense and all_same_algo:
                        label_text = f'{noise_param}={noise}'
                    elif condense and all_same_noise:
                        label_text = f'<b>{algo_name}</b>'
                    else:
                        label_text = f'<b>{algo_name}</b> {noise_param}={noise}'
                    lines.append(f'<span style="color:{color}">{label_text}</span>')
            if lines:
                annotation_kwargs = dict(
                    xref='paper', yref='paper',
                    x=(info_panel_x if info_panel_x is not None else 90) / 100,
                    y=(info_panel_y if info_panel_y is not None else 75) / 100,
                    xanchor='right', yanchor='top',
                    text='<br>'.join(lines),
                    showarrow=False,
                    align='right',
                    font=dict(size=ann_font_size),
                )
                if not print_mode:
                    annotation_kwargs.update(
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='grey',
                        borderwidth=1,
                    )
                fig.add_annotation(**annotation_kwargs)
    else:
        # Fallback for other plot types
        fig = go.Figure()

    # Build LON stats table
    lon_nodes = [n for n in G.nodes() if "Local Optimum" in n]
    total_optima = len(lon_nodes)
    if total_optima > 0:
        if opt_feas_map:
            feasible = sum(
                1 for n in lon_nodes
                if opt_feas_map.get(",".join(str(int(x)) for x in G.nodes[n].get('solution', []))) == 1
            )
            infeasible = sum(
                1 for n in lon_nodes
                if opt_feas_map.get(",".join(str(int(x)) for x in G.nodes[n].get('solution', []))) == 0
            )
            feas_display, infeas_display = str(feasible), str(infeasible)
        else:
            feas_display, infeas_display = 'N/A', 'N/A'
        maximizing = (config.opt_goal or 'max')[:3].lower() == 'max'
        best_fitness = max(G.nodes[n].get('fitness', float('-inf')) for n in lon_nodes) if maximizing \
            else min(G.nodes[n].get('fitness', float('inf')) for n in lon_nodes)
        global_opt_weight = sum(
            G.nodes[n].get('weight', 0)
            for n in lon_nodes
            if G.nodes[n].get('fitness') == best_fitness
        )
        lon_stats_table = dash_table.DataTable(
            columns=[
                {'name': 'Total Optima', 'id': 'total'},
                {'name': 'Feasible Optima', 'id': 'feasible'},
                {'name': 'Infeasible Optima', 'id': 'infeasible'},
                {'name': 'Global Optima Weight', 'id': 'go_weight'},
            ],
            data=[{
                'total': total_optima,
                'feasible': feas_display,
                'infeasible': infeas_display,
                'go_weight': global_opt_weight,
            }],
            style_table={'width': '700px'},
            style_cell={'textAlign': 'center', 'padding': '8px'},
            style_header={'fontWeight': 'bold'},
        )
    else:
        lon_stats_table = html.Div()

    # Build LON node scatter/violin (selectable axes and plot style) and correlations
    scatter_x_key = lon_scatter_x or LON_SCATTER_DEFAULT_X_AXIS
    scatter_y_key = lon_scatter_y or LON_SCATTER_DEFAULT_Y_AXIS
    scatter_plot_style = lon_scatter_plot_style or LON_SCATTER_DEFAULT_PLOT_STYLE
    scatter_multi_noise = 'multi-noise' in (lon_scatter_multi_noise or [])
    if config.plot_type in ('NLon_box', 'NLon_IQR') and node_noise and fitness_dict:
        if scatter_multi_noise:
            # Recompute noisy samples at every intensity from 1 up to the
            # configured 'Noise intensity', plotting each as its own series.
            max_intensity = max(int(config.noisy_lon.intensity or 1), 1)
            node_stats_by_intensity = {}
            for intensity in range(1, max_intensity + 1):
                if intensity == int(config.noisy_lon.intensity):
                    intensity_node_noise = node_noise
                else:
                    intensity_config = dataclass_replace(
                        config, noisy_lon=dataclass_replace(config.noisy_lon, intensity=float(intensity))
                    )
                    _, intensity_node_noise = add_lon_nodes(
                        G, local_optima_processed, lon_node_mapping, intensity_config, PID
                    )
                node_stats_by_intensity[intensity] = compute_node_feasibility_error(
                    G, pos, intensity_node_noise, fitness_dict, neigh_feas_map
                )
            feas_error_fig = plot_lon_stats_multi(node_stats_by_intensity, scatter_x_key, scatter_y_key, scatter_plot_style)
            node_stats = [s for stats in node_stats_by_intensity.values() for s in stats]
        else:
            node_stats = compute_node_feasibility_error(G, pos, node_noise, fitness_dict, neigh_feas_map)
            feas_error_fig = plot_lon_stats(node_stats, scatter_x_key, scatter_y_key, scatter_plot_style)
        selected_correlation = compute_correlation_pair(node_stats, scatter_x_key, scatter_y_key)
        selected_corr_component = build_selected_correlation_display(
            selected_correlation,
            LON_SCATTER_AXIS_LABELS.get(scatter_x_key, scatter_x_key),
            LON_SCATTER_AXIS_LABELS.get(scatter_y_key, scatter_y_key),
        )
        feas_error_corr_component = build_correlation_table(compute_pairwise_correlations(node_stats))
    else:
        feas_error_fig = plot_lon_stats([], scatter_x_key, scatter_y_key, scatter_plot_style)
        selected_corr_component = html.Div()
        feas_error_corr_component = html.Div("Select NLon_box or NLon_IQR plot type to view this chart.")

    return (fig, debug_summary_component, stn_stats_table, lon_stats_table,
            feas_error_fig, selected_corr_component, feas_error_corr_component)
