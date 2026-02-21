"""
Configuration dataclasses for the visualization module.

This module defines typed configuration objects that encapsulate all the
settings used throughout the visualization pipeline, replacing the need
to pass 40+ individual parameters through function calls.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class NodeSizeConfig:
    """Configuration for node sizes in the visualization."""
    stn_min: float = 5.0
    stn_max: float = 15.0
    lon_min: float = 5.0
    lon_max: float = 15.0


@dataclass
class OpacityConfig:
    """Configuration for opacity values of different elements."""
    lon_node: float = 1.0
    lon_edge: float = 1.0
    stn_node: float = 1.0
    stn_edge: float = 1.0
    noise_bar: float = 0.5


@dataclass
class AxisConfig:
    """Configuration for custom axis ranges."""
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None


@dataclass
class CameraConfig:
    """Configuration for camera position in 3D view."""
    azimuth_deg: float = 45.0
    elevation_deg: float = 30.0

    def get_camera_eye(self, r: float = 2.5) -> Dict[str, float]:
        """Calculate camera eye position from azimuth and elevation."""
        import numpy as np
        azimuth = np.deg2rad(self.azimuth_deg)
        elevation = np.deg2rad(self.elevation_deg)
        return dict(
            x=r * np.cos(elevation) * np.cos(azimuth),
            y=r * np.cos(elevation) * np.sin(azimuth),
            z=r * np.sin(elevation)
        )


@dataclass
class STNConfig:
    """Configuration specific to STN (Search Trajectory Network) visualization."""
    final_x_gens: Optional[int] = 150  # None = keep all
    lower_fit_limit: Optional[float] = None
    upper_fit_limit: Optional[float] = None
    stride: int = 1  # 1 = keep all, 2 = every 2nd, etc.
    edge_size: float = 2.0
    show_hamming: bool = False
    dedup_prior_noise: bool = False


@dataclass
class LONConfig:
    """Configuration specific to LON (Local Optima Network) visualization."""
    filter_negative: bool = False
    show_hamming: bool = False
    fit_percent: float = 100.0
    node_colour_mode: str = 'fitness'  # 'fitness' | 'feasible' | 'neigh'
    edge_colour_feas: bool = False
    edge_size: float = 2.0


@dataclass
class NoisyLONConfig:
    """Configuration for noisy LON visualization."""
    fit_func: str = 'kpv1s'
    intensity: float = 1.0
    samples: int = 10


@dataclass
class PlotConfig:
    """
    Main configuration object that holds all visualization settings.

    This replaces the 40+ individual parameters passed to update_plot callback.
    """
    # Display options
    show_labels: bool = False
    hide_stn_nodes: bool = False
    hide_lon_nodes: bool = False
    plot_3d: bool = True
    use_solution_iterations: bool = False
    lon_node_strength: bool = False
    local_optima_color: bool = False
    curve_edges: bool = True

    # Run selection options
    show_best: bool = False
    show_mean: bool = False
    show_median: bool = False
    show_worst: bool = False

    # Estimated fitness display
    show_estimated_adopted: bool = False
    show_estimated_discarded: bool = False
    show_stn_boxplots: bool = False

    # Mode settings
    stn_plot_type: str = 'posterior'  # 'posterior', 'prior', 'multiobjective'
    layout_type: str = 'mds'
    plot_type: str = 'RegLon'
    hover_info: str = 'fitness'

    # Optimization settings
    optimum: Optional[float] = None
    opt_goal: str = 'max'
    problem_id: str = ''

    # Run display settings
    run_start_index: int = 0
    n_runs_display: int = 1

    # Sub-configurations
    node_size: NodeSizeConfig = field(default_factory=NodeSizeConfig)
    opacity: OpacityConfig = field(default_factory=OpacityConfig)
    axis: AxisConfig = field(default_factory=AxisConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    stn: STNConfig = field(default_factory=STNConfig)
    lon: LONConfig = field(default_factory=LONConfig)
    noisy_lon: NoisyLONConfig = field(default_factory=NoisyLONConfig)

    # Colors
    algo_colors: List[str] = field(default_factory=lambda: ['blue', 'orange', 'purple', 'cyan', 'magenta', 'brown'])
    noisy_node_color: str = 'grey'
    node_color_shared: str = 'green'


def parse_callback_inputs(
    optimum: Optional[float],
    pid: str,
    opt_goal: str,
    options: List[str],
    run_options: List[str],
    stn_lower_fit_limit: Optional[float],
    lo_fit_percent: float,
    lon_options: List[str],
    lon_node_colour_mode: str,
    lon_edge_colour_feas: List[str],
    nlon_fit_func: str,
    nlon_intensity: float,
    nlon_samples: int,
    layout_value: str,
    plot_type: str,
    hover_info_value: str,
    azimuth_deg: float,
    elevation_deg: float,
    run_start_index: int,
    n_runs_display: int,
    axis_values: Dict[str, Any],
    opacity_noise_bar: float,
    lon_node_opacity: float,
    lon_edge_opacity: float,
    stn_node_opacity: float,
    stn_edge_opacity: float,
    stn_node_min: float,
    stn_node_max: float,
    lon_node_min: float,
    lon_node_max: float,
    lon_edge_size_slider: float,
    stn_edge_size_slider: float,
    stn_plot_type: str,
) -> PlotConfig:
    """
    Parse raw callback inputs into a structured PlotConfig object.

    This function converts the many individual callback parameters into a
    well-organized configuration object with sensible defaults.

    Args:
        All the callback Input parameters from the Dash callback.

    Returns:
        PlotConfig: A fully populated configuration object.
    """
    # Parse checkbox options
    options = options or []
    run_options = run_options or []
    lon_options = lon_options or []
    lon_edge_colour_feas = lon_edge_colour_feas or []

    return PlotConfig(
        # Display options
        show_labels='show_labels' in options,
        hide_stn_nodes='hide_STN_nodes' in options,
        hide_lon_nodes='hide_LON_nodes' in options,
        plot_3d='plot_3D' in options,
        use_solution_iterations='use_solution_iterations' in options,
        lon_node_strength='LON_node_strength' in options,
        local_optima_color='local_optima_color' in options,
        curve_edges=True,

        # Run selection options
        show_best='show_best' in run_options,
        show_mean='show_mean' in run_options,
        show_median='show_median' in run_options,
        show_worst='show_worst' in run_options,

        # Estimated fitness display
        show_estimated_adopted='show_estimated_adopted' in run_options,
        show_estimated_discarded='show_estimated_discarded' in run_options,
        show_stn_boxplots='show_stn_boxplots' in run_options,

        # Mode settings
        stn_plot_type=stn_plot_type or 'posterior',
        layout_type=layout_value,
        plot_type=plot_type,
        hover_info=hover_info_value,

        # Optimization settings
        optimum=optimum,
        opt_goal=opt_goal,
        problem_id=pid,

        # Run display settings
        run_start_index=run_start_index,
        n_runs_display=n_runs_display,

        # Sub-configurations
        node_size=NodeSizeConfig(
            stn_min=stn_node_min,
            stn_max=stn_node_max,
            lon_min=lon_node_min,
            lon_max=lon_node_max,
        ),
        opacity=OpacityConfig(
            lon_node=lon_node_opacity,
            lon_edge=lon_edge_opacity,
            stn_node=stn_node_opacity,
            stn_edge=stn_edge_opacity,
            noise_bar=opacity_noise_bar,
        ),
        axis=AxisConfig(
            x_min=axis_values.get('custom_x_min'),
            x_max=axis_values.get('custom_x_max'),
            y_min=axis_values.get('custom_y_min'),
            y_max=axis_values.get('custom_y_max'),
            z_min=axis_values.get('custom_z_min'),
            z_max=axis_values.get('custom_z_max'),
        ),
        camera=CameraConfig(
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
        ),
        stn=STNConfig(
            lower_fit_limit=stn_lower_fit_limit,
            edge_size=stn_edge_size_slider,
            show_hamming='STN-hamming' in run_options,
            dedup_prior_noise='dedup-prior-noise' in run_options,
        ),
        lon=LONConfig(
            filter_negative='LON-filter-neg' in lon_options,
            show_hamming='LON-hamming' in lon_options,
            fit_percent=lo_fit_percent,
            node_colour_mode=lon_node_colour_mode,
            edge_colour_feas='edge_feas' in lon_edge_colour_feas,
            edge_size=lon_edge_size_slider,
        ),
        noisy_lon=NoisyLONConfig(
            fit_func=nlon_fit_func,
            intensity=nlon_intensity,
            samples=nlon_samples,
        ),
    )
