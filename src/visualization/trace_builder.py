"""
Trace building functions for the visualization module.

This module contains functions for creating Plotly trace objects for
edges, nodes, labels, boxplots, and assembling the final figure.
"""

from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .config import PlotConfig
from ..common import hamming_distance
from ..dashboard.DashboardHelpers import quadratic_bezier, should_label_edge


def _precalculate_stn_edge_colors(G: nx.MultiDiGraph) -> Dict[Tuple, Dict[str, int]]:
    """
    Pre-calculate color indices for STN edge pairs to enable consistent curved edge rendering.

    Args:
        G: NetworkX graph containing edges

    Returns:
        Dictionary mapping (u, v) pairs to color index mappings
    """
    stn_edges_by_pair_color = {}
    for u_pre, v_pre, key_pre, data_pre in G.edges(data=True, keys=True):
        if data_pre.get("edge_type") in ("STN", "STN_SO"):
            pair_pre = (u_pre, v_pre)
            color_pre = data_pre.get('color', 'default_color')
            if pair_pre not in stn_edges_by_pair_color:
                stn_edges_by_pair_color[pair_pre] = {}
            if color_pre not in stn_edges_by_pair_color[pair_pre]:
                stn_edges_by_pair_color[pair_pre][color_pre] = 0

    # Create a mapping from color to index for each pair
    color_indices_for_pair = {}
    for pair, colors_dict in stn_edges_by_pair_color.items():
        sorted_colors = sorted(colors_dict.keys())
        color_indices_for_pair[pair] = {color: idx for idx, color in enumerate(sorted_colors)}

    return color_indices_for_pair


def create_edge_traces(
    G: nx.MultiDiGraph,
    pos: Dict[str, Tuple[float, float]],
    config: PlotConfig
) -> Tuple[List[go.Scatter3d], List[float], List[float], List[float], List[str]]:
    """
    Create Plotly traces for all edges in the graph.

    Args:
        G: NetworkX graph containing edges
        pos: Dictionary mapping node names to (x, y) positions
        config: PlotConfig object with settings

    Returns:
        Tuple of (edge_traces, edge_label_x, edge_label_y, edge_label_z, edge_labels)
    """
    traces = []
    edge_label_x = []
    edge_label_y = []
    edge_label_z = []
    edge_labels = []

    # Pre-calculate color indices for STN edges
    color_indices_for_pair = _precalculate_stn_edge_colors(G)

    option_curve_edges = config.curve_edges
    STN_edge_size_slider = config.stn.edge_size
    LON_edge_size_slider = config.lon.edge_size
    LON_edge_opacity = config.opacity.lon_edge
    STN_edge_opacity = config.opacity.stn_edge
    STN_hamming = config.stn.show_hamming
    LON_hamming = config.lon.show_hamming
    hide_LON_nodes = config.hide_lon_nodes
    hide_STN_nodes = config.hide_stn_nodes

    print('Plotting edges...')

    for u, v, key, data in G.edges(data=True, keys=True):
        # Skip edges if nodes don't have positions
        if u not in pos or v not in pos:
            print(f"Skipping edge ({u}, {v}) as node positions are missing.")
            continue

        # Skip edges involving hidden node types
        node_u_type = G.nodes[u].get('type', '')
        node_v_type = G.nodes[v].get('type', '')
        if hide_LON_nodes and ("LON" in str(node_u_type) or "LON" in str(node_v_type)):
            continue
        if hide_STN_nodes and any(t in str(node_u_type) or t in str(node_v_type) for t in ["STN", "NoisySTN"]):
            continue

        edge_type = data.get("edge_type", "")
        edge_color = data.get('color', 'grey')
        edge_opacity = 1.0
        mid_x, mid_y, mid_z = 0, 0, 0

        # Determine opacity based on edge type
        if edge_type == 'LON':
            edge_opacity = LON_edge_opacity
        elif edge_type in ('STN', 'STN_SO'):
            edge_opacity = STN_edge_opacity
        elif edge_type in ('Noise', 'Noise_SO'):
            edge_opacity = STN_edge_opacity

        # Get start and end points (2D position + Z fitness)
        x0, y0 = pos[u][:2]
        x1, y1 = pos[v][:2]
        z0 = G.nodes[u].get('fitness', 0)
        z1 = G.nodes[v].get('fitness', 0)
        z0 = float(z0) if z0 is not None else 0
        z1 = float(z1) if z1 is not None else 0

        # Process curved STN edges
        if option_curve_edges and edge_type in ("STN", "STN_SO") and (u, v) in color_indices_for_pair:
            pair = (u, v)
            current_edge_color = data.get('color', 'default_color')

            color_indices_map = color_indices_for_pair[pair]
            total_distinct_colors = len(color_indices_map)

            if current_edge_color in color_indices_map:
                color_idx = color_indices_map[current_edge_color]

                base_curvature = 0.2
                max_offset_factor = 1.5

                if total_distinct_colors > 1:
                    curvature = base_curvature * max_offset_factor * (
                        (color_idx - (total_distinct_colors - 1) / 2.0) / ((total_distinct_colors - 1) / 2.0)
                    )
                    if abs(curvature) < 0.01:
                        curvature = 0.05 * np.sign(color_idx - (total_distinct_colors - 1) / 2.0 + 1e-6)
                elif total_distinct_colors == 1:
                    curvature = base_curvature
                else:
                    curvature = 0

                # Prevent extremely large curvatures if start/end points are very close
                dist_xy = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if dist_xy < 0.1:
                    curvature *= (dist_xy / 0.1)

                start_2d = (x0, y0)
                end_2d = (x1, y1)
                curve_xy = quadratic_bezier(start_2d, end_2d, curvature=curvature, n_points=20)

                z_values = np.linspace(z0, z1, len(curve_xy))

                edge_trace = go.Scatter3d(
                    x=list(curve_xy[:, 0]),
                    y=list(curve_xy[:, 1]),
                    z=list(z_values),
                    mode='lines',
                    line=dict(width=STN_edge_size_slider, color=current_edge_color),
                    opacity=edge_opacity,
                    hoverinfo='none',
                    showlegend=False
                )
                traces.append(edge_trace)

                mid_index = len(curve_xy) // 2
                mid_x = curve_xy[mid_index, 0]
                mid_y = curve_xy[mid_index, 1]
                mid_z = z_values[mid_index]
            else:
                # Fallback to straight line
                edge_trace = go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines',
                    line=dict(width=STN_edge_size_slider, color=current_edge_color, dash='dot'),
                    opacity=edge_opacity * 0.8, hoverinfo='none', showlegend=False
                )
                traces.append(edge_trace)
                mid_x, mid_y, mid_z = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        else:
            # Draw straight lines for LON, Noise edges, or STN if curving is off
            line_width = 1
            dash_style = 'solid'
            if edge_type == 'LON':
                norm_w = data.get('norm_weight', 0)
                line_width = 1 + norm_w * (LON_edge_size_slider - 1)
                edge_color = data.get('color', 'black')
            elif edge_type in ('Noise', 'Noise_SO'):
                line_width = 3
                dash_style = 'solid'
                edge_color = data.get('color', 'grey')
            elif edge_type in ('STN', 'STN_SO'):
                line_width = STN_edge_size_slider
                edge_color = data.get('color', 'green')

            edge_trace = go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(width=max(0.5, line_width), color=edge_color, dash=dash_style),
                opacity=edge_opacity,
                hoverinfo='none',
                showlegend=False
            )
            traces.append(edge_trace)
            mid_x, mid_y, mid_z = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2

        # Add edge labels (Hamming distance) if requested
        if should_label_edge(u, v, STN_hamming, LON_hamming):
            try:
                sol_u = G.nodes[u].get('solution')
                sol_v = G.nodes[v].get('solution')
                if sol_u and sol_v:
                    hd = hamming_distance(sol_u, sol_v)
                    edge_label_x.append(mid_x)
                    edge_label_y.append(mid_y)
                    edge_label_z.append(mid_z + 0.1)
                    edge_labels.append(f"H={hd}")
            except Exception as e:
                print(f"Error calculating Hamming distance for label ({u}, {v}): {e}")

    return traces, edge_label_x, edge_label_y, edge_label_z, edge_labels


def create_edge_label_trace(
    edge_label_x: List[float],
    edge_label_y: List[float],
    edge_label_z: List[float],
    edge_labels: List[str]
) -> Optional[go.Scatter3d]:
    """
    Create a trace for edge labels.

    Args:
        edge_label_x: X coordinates for labels
        edge_label_y: Y coordinates for labels
        edge_label_z: Z coordinates for labels
        edge_labels: Label text

    Returns:
        Scatter3d trace for labels, or None if no labels
    """
    if not edge_labels:
        return None

    return go.Scatter3d(
        x=edge_label_x, y=edge_label_y, z=edge_label_z,
        mode='text', text=edge_labels,
        textposition="middle center",
        textfont=dict(size=10, color='black'),
        hoverinfo='none',
        showlegend=False
    )


def create_node_traces(
    G: nx.MultiDiGraph,
    pos: Dict[str, Tuple[float, float]],
    config: PlotConfig
) -> Tuple[go.Scatter3d, go.Scatter3d]:
    """
    Create Plotly traces for nodes.

    Args:
        G: NetworkX graph containing nodes
        pos: Dictionary mapping node names to (x, y) positions
        config: PlotConfig object with settings

    Returns:
        Tuple of (LON_node_trace, STN_node_trace)
    """
    print('Plotting nodes...')

    node_x, node_y, node_z = [], [], []
    node_sizes, node_colors = [], []

    LON_node_x, LON_node_y, LON_node_z = [], [], []
    LON_node_sizes, LON_node_colors = [], []

    for node, attr in G.nodes(data=True):
        if node not in pos:
            continue

        x, y = pos[node][:2]
        z = attr.get('fitness', 0)

        if "Local Optimum" in node:
            LON_node_x.append(x)
            LON_node_y.append(y)
            LON_node_z.append(z)
            LON_node_sizes.append(attr.get('size', 1))
            LON_node_colors.append(attr.get('color', 'grey'))
        else:
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_sizes.append(attr.get('size', 1))
            node_colors.append(attr.get('color', 'blue'))

    LON_node_trace = go.Scatter3d(
        x=LON_node_x,
        y=LON_node_y,
        z=LON_node_z,
        mode='markers',
        marker=dict(
            size=LON_node_sizes,
            color=LON_node_colors,
            opacity=config.opacity.lon_node
        ),
        showlegend=False
    )

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors
        ),
        opacity=config.opacity.stn_node,
        showlegend=False
    )

    return LON_node_trace, node_trace


def create_estimated_fitness_traces(
    G: nx.MultiDiGraph,
    pos: Dict[str, Tuple[float, float]],
    config: PlotConfig
) -> List[go.Scatter3d]:
    """
    Create cross marker traces for estimated fitness values.

    For each STN node that has an estimated fitness, places a cross marker
    at the same x,y position but at the estimated fitness z-value.

    Args:
        G: NetworkX graph containing nodes with estimated fitness attributes
        pos: Dictionary mapping node names to (x, y) positions
        config: PlotConfig object with settings

    Returns:
        List of Scatter3d traces for cross markers
    """
    traces = []

    for attr_key, count_key, color, label in [
        ('estimated_fitness_adopted', 'count_estimated_adopted', 'purple', 'Est. fitness (adopted)'),
        ('estimated_fitness_discarded', 'count_estimated_discarded', 'orange', 'Est. fitness (discarded)'),
    ]:
        # Check if this variant is enabled
        if attr_key == 'estimated_fitness_adopted' and not config.show_estimated_adopted:
            continue
        if attr_key == 'estimated_fitness_discarded' and not config.show_estimated_discarded:
            continue

        cross_x, cross_y, cross_z = [], [], []
        hover_texts = []

        for node, attr in G.nodes(data=True):
            if node not in pos:
                continue
            if attr.get('type') != 'STN':
                continue

            est_fit = attr.get(attr_key)
            if est_fit is None:
                continue

            x, y = pos[node][:2]
            cross_x.append(x)
            cross_y.append(y)
            cross_z.append(float(est_fit))
            true_fit = attr.get('fitness', 0)
            n_evals = attr.get(count_key, '?')
            hover_texts.append(f"True: {true_fit}<br>Est: {est_fit}<br>n: {n_evals}")

        if cross_x:
            trace = go.Scatter3d(
                x=cross_x,
                y=cross_y,
                z=cross_z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    symbol='cross',
                    opacity=1,
                ),
                text=hover_texts,
                hoverinfo='text',
                name=label,
                showlegend=True,
            )
            traces.append(trace)

    return traces


def create_boxplot_traces(
    pos: Dict[str, Tuple[float, float]],
    node_noise: Dict[str, List[float]],
    fitness_dict: Dict[str, float],
    config: PlotConfig
) -> List[go.Scatter3d]:
    """
    Create mini boxplot traces for noisy fitness visualization.

    Args:
        pos: Dictionary mapping node names to (x, y) positions
        node_noise: Dictionary mapping node names to lists of noisy fitness values
        fitness_dict: Dictionary mapping node names to base fitness values
        config: PlotConfig object with settings

    Returns:
        List of Scatter3d traces for boxplot elements
    """
    traces = []
    dx = 0.05  # horizontal offset for the mini boxplot
    opacity_noise_bar = config.opacity.noise_bar

    print('Plotting noise bar plots...')

    for node in pos:
        if node in fitness_dict and node in node_noise:
            x, y = pos[node][:2]
            base_z = fitness_dict[node]
            noise = np.array(node_noise[node])

            # Compute quartiles and extremes
            min_val = np.min(noise)
            q1 = np.percentile(noise, 25)
            med = np.median(noise)
            q3 = np.percentile(noise, 75)
            max_val = np.max(noise)

            # Map to z values
            if max_val == min_val:
                z_min = z_q1 = z_med = z_q3 = z_max = base_z
            else:
                z_min = min_val
                z_q1 = q1
                z_med = med
                z_q3 = q3
                z_max = max_val

            x_box = x

            # Create traces for each component of the boxplot
            trace_whisker_top = go.Scatter3d(
                x=[x_box, x_box],
                y=[y, y],
                z=[z_q3, z_max],
                mode='lines',
                line=dict(color='grey', width=2),
                opacity=opacity_noise_bar,
                showlegend=False
            )
            trace_whisker_bottom = go.Scatter3d(
                x=[x_box, x_box],
                y=[y, y],
                z=[z_q1, z_min],
                mode='lines',
                line=dict(color='grey', width=2),
                opacity=opacity_noise_bar,
                showlegend=False
            )
            trace_cap_top = go.Scatter3d(
                x=[x_box - dx / 2, x_box + dx / 2],
                y=[y, y],
                z=[z_max, z_max],
                mode='lines',
                line=dict(color='grey', width=2),
                opacity=opacity_noise_bar,
                showlegend=False
            )
            trace_cap_bottom = go.Scatter3d(
                x=[x_box - dx / 2, x_box + dx / 2],
                y=[y, y],
                z=[z_min, z_min],
                mode='lines',
                line=dict(color='grey', width=2),
                opacity=opacity_noise_bar,
                showlegend=False
            )
            trace_box = go.Scatter3d(
                x=[x_box, x_box],
                y=[y, y],
                z=[z_q1, z_q3],
                mode='lines',
                line=dict(color='black', width=4),
                opacity=opacity_noise_bar,
                showlegend=False
            )
            trace_medianx = go.Scatter3d(
                x=[x_box - dx, x_box + dx],
                y=[y, y],
                z=[z_med, z_med],
                mode='lines',
                line=dict(color='red', width=3),
                opacity=opacity_noise_bar,
                showlegend=False
            )
            trace_mediany = go.Scatter3d(
                x=[x, x],
                y=[y - dx, y + dx],
                z=[z_med, z_med],
                mode='lines',
                line=dict(color='red', width=3),
                opacity=opacity_noise_bar,
                showlegend=False
            )

            traces.extend([
                trace_whisker_top, trace_whisker_bottom,
                trace_cap_top, trace_cap_bottom,
                trace_box,
                trace_medianx, trace_mediany
            ])

    return traces


def create_axis_settings(
    G: nx.MultiDiGraph,
    pos: Dict[str, Tuple[float, float]],
    config: PlotConfig,
    node_noise: Optional[Dict[str, List[float]]] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Create axis settings dictionaries for the 3D plot.

    Args:
        G: NetworkX graph
        pos: Node positions
        config: PlotConfig object
        node_noise: Optional noisy fitness data

    Returns:
        Tuple of (xaxis_settings, yaxis_settings, zaxis_settings)
    """
    # Calculate substitute values for when custom axis range is missing
    if len(G.nodes) > 0:
        x_values = [pos[node][0] for node in G.nodes() if node in pos]
        y_values = [pos[node][1] for node in G.nodes() if node in pos]
        fit_values = [data['fitness'] for _, data in G.nodes(data=True) if 'fitness' in data]

        if x_values:
            x_min_sub, x_max_sub = min(x_values) - 1, max(x_values) + 1
        else:
            x_min_sub, x_max_sub = -1, 1

        if y_values:
            y_min_sub, y_max_sub = min(y_values) - 1, max(y_values) + 1
        else:
            y_min_sub, y_max_sub = -1, 1

        if fit_values:
            z_min_sub, z_max_sub = min(fit_values) - 1, max(fit_values) + 1
        else:
            z_min_sub, z_max_sub = -1, 1

        if node_noise:
            z_max_sub = max(max(noisy_list) for noisy_list in node_noise.values()) + 1
            z_min_sub = min(min(noisy_list) for noisy_list in node_noise.values()) - 1
    else:
        x_min_sub = x_max_sub = y_min_sub = y_max_sub = z_min_sub = z_max_sub = 1

    show_xy_labels = config.layout_type == 'raw'
    xaxis_settings = dict(
        title='x1' if show_xy_labels else '',
        titlefont=dict(size=24, color='black'),
        tickfont=dict(size=16, color='black'),
        showticklabels=show_xy_labels
    )
    yaxis_settings = dict(
        title='x2' if show_xy_labels else '',
        titlefont=dict(size=24, color='black'),
        tickfont=dict(size=16, color='black'),
        showticklabels=show_xy_labels
    )

    z_axis_title = 'hypervolume' if config.stn_plot_type == 'multiobjective' else 'fitness'
    zaxis_settings = dict(
        title=z_axis_title,
        titlefont=dict(size=24, color='black'),
        tickfont=dict(size=16, color='black'),
    )

    # Apply custom axis options
    axis_config = config.axis
    if axis_config.x_min is not None or axis_config.x_max is not None:
        custom_x_min = axis_config.x_min if axis_config.x_min is not None else x_min_sub
        custom_x_max = axis_config.x_max if axis_config.x_max is not None else x_max_sub
        xaxis_settings["range"] = [custom_x_min, custom_x_max]

    if axis_config.y_min is not None or axis_config.y_max is not None:
        custom_y_min = axis_config.y_min if axis_config.y_min is not None else y_min_sub
        custom_y_max = axis_config.y_max if axis_config.y_max is not None else y_max_sub
        yaxis_settings["range"] = [custom_y_min, custom_y_max]

    if axis_config.z_min is not None or axis_config.z_max is not None:
        custom_z_min = axis_config.z_min if axis_config.z_min is not None else z_min_sub
        custom_z_max = axis_config.z_max if axis_config.z_max is not None else z_max_sub
        zaxis_settings["range"] = [custom_z_min, custom_z_max]

    return xaxis_settings, yaxis_settings, zaxis_settings


def create_figure(
    traces: List[go.Scatter3d],
    config: PlotConfig,
    xaxis_settings: Dict,
    yaxis_settings: Dict,
    zaxis_settings: Dict,
    output_path: str = 'plots/3dplot.html'
) -> go.Figure:
    """
    Assemble all traces into a final Plotly figure.

    Args:
        traces: List of all trace objects
        config: PlotConfig object
        xaxis_settings: X-axis configuration
        yaxis_settings: Y-axis configuration
        zaxis_settings: Z-axis configuration
        output_path: Path to write HTML file

    Returns:
        Assembled Plotly Figure
    """
    print('Assembling figure...')

    camera_eye = config.camera.get_camera_eye()

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=False,
        width=1200,
        height=1200,
        scene=dict(
            camera=dict(eye=camera_eye),
            xaxis=xaxis_settings,
            yaxis=yaxis_settings,
            zaxis=zaxis_settings
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    print('Displaying plot')
    fig.write_html(output_path)

    return fig


def build_all_traces(
    G: nx.MultiDiGraph,
    pos: Dict[str, Tuple[float, float]],
    config: PlotConfig,
    node_noise: Optional[Dict[str, List[float]]] = None,
    fitness_dict: Optional[Dict[str, float]] = None
) -> List[go.Scatter3d]:
    """
    Build all traces for the visualization.

    This is the main entry point that combines edge, node, and boxplot traces.

    Args:
        G: NetworkX graph
        pos: Node positions
        config: PlotConfig object
        node_noise: Optional noisy fitness data
        fitness_dict: Optional base fitness data

    Returns:
        List of all Scatter3d traces
    """
    traces = []

    # Create edge traces
    edge_traces, edge_label_x, edge_label_y, edge_label_z, edge_labels = create_edge_traces(G, pos, config)
    traces.extend(edge_traces)

    # Add edge labels
    edge_label_trace = create_edge_label_trace(edge_label_x, edge_label_y, edge_label_z, edge_labels)
    if edge_label_trace:
        traces.append(edge_label_trace)

    # Create node traces
    LON_node_trace, node_trace = create_node_traces(G, pos, config)
    traces.append(LON_node_trace)
    traces.append(node_trace)

    # Add boxplot traces if needed
    if config.plot_type == 'NLon_box' and node_noise and fitness_dict:
        boxplot_traces = create_boxplot_traces(pos, node_noise, fitness_dict, config)
        traces.extend(boxplot_traces)

    # Add STN fitness boxplots if enabled
    if config.show_stn_boxplots:
        stn_node_noise = {}
        stn_fitness_dict = {}
        for node, attr in G.nodes(data=True):
            if attr.get('type') != 'STN' or node not in pos:
                continue
            variant_fits = attr.get('noisy_variant_fitnesses', [])
            if len(variant_fits) > 1:
                stn_node_noise[node] = variant_fits
                stn_fitness_dict[node] = attr.get('fitness', 0)
        if stn_node_noise:
            stn_boxplot_traces = create_boxplot_traces(pos, stn_node_noise, stn_fitness_dict, config)
            traces.extend(stn_boxplot_traces)

    # Add estimated fitness cross markers if enabled
    if config.show_estimated_adopted or config.show_estimated_discarded:
        estimated_traces = create_estimated_fitness_traces(G, pos, config)
        traces.extend(estimated_traces)

    return traces
