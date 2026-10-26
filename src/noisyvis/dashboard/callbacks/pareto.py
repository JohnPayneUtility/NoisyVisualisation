"""The Pareto-front tab: one figure, dispatched through the Pareto plot registry, and the controls of the
Pareto Rank Layers plot type."""

import plotly.graph_objects as go
from dash import Input, Output, State, ctx, no_update

from ..instance import app
from ..data import df
from ..mo_frames import (
    PARETO_NOISY_PAIRS,
    RANK_LAYERS_PLOT,
    pareto_front_entries,
    pareto_pair_entries,
    pareto_runs,
    rank_layer_controls,
    rank_layer_selection,
    rank_layer_trigger,
)
from ...problems.instances import get_knapsack_problem_stats
from ...viz.plots import get_pareto_plot
from ...viz.plots.base import FRONT_KEYS, UNSUPPORTED_METRIC_TITLE, create_empty_figure, selected_front

# Plots that draw fronts in objective space: they follow the front type and the attainment option
FRONT_PLOTS = ('Basic', 'Subplots', 'SubplotsMulti', 'SubplotsHighlight', 'paretoAnimation')
# Plots that accept gridlines= (the "Show invalid solution gridlines" option)
GRIDLINE_PLOTS = ('Basic',)


def knapsack_gridlines(pid):
    """
    The (x, y) invalid-solution gridlines for the selected problem: for a knapsack PID, the sum of its item
    values (objective 1 can never exceed it) and zero weight (objective 2 can never go below it); None when
    the PID's problem is not 'knapsack'.
    """
    if not pid or not isinstance(pid, str):  # the PID store starts as []
        return None
    if 'PID' not in df.columns or 'problem_name' not in df.columns:
        return None
    if 'knapsack' not in set(df.loc[df['PID'] == pid, 'problem_name']):
        return None
    stats = get_knapsack_problem_stats(pid)
    return None if stats is None else (stats['sum_values'], 0.0)


@app.callback(
    Output("plotParetoFront", "figure"),
    [Input('STN_data', 'data'),
     Input('paretoFrontPlotType', 'value'),
     Input('IndVsDist_IndType', 'value'),
     Input('IndVsDist_DistType', 'value'),
     Input('paretoPlotNumRuns', 'value'),
     Input('paretoPlotWindowSize', 'value'),
     Input('paretoAttainmentSurface', 'value'),
     Input('paretoInvalidGridlines', 'value'),
     Input('PID', 'data'),
     Input('paretoRankRun', 'value'),
     Input('paretoRankBasis', 'value'),
     Input('paretoRankGeneration', 'value'),
     Input('paretoRankMax', 'value'),
     Input('paretoRankStaircase', 'value')],
)
def updateParetoPlot(stn_rows, paretoFrontPlotType, IndVsDist_IndType, IndVsDist_DistType, nruns, windowSize,
                     attainment, invalid_gridlines=None, pid=None, rank_run=None, rank_basis=None,
                     rank_generation=None, rank_max=None, rank_staircase=None):
    """
    Update the Pareto front plot based on the selected plot type.
    Uses the plotting registry for dynamic dispatch.

    The fronts are built server-side from the selected rows (STN_data): the "Noisy (A vs B)" plot types
    draw their own two fronts (mo_frames.PARETO_NOISY_PAIRS); every other type draws or measures the front
    type selected in IndVsDist_IndType. The GRIDLINE_PLOTS also draw the selected problem's invalid-solution
    gridlines (knapsack_gridlines) when that option is ticked. Pareto Rank Layers draws the selected run's
    population at the exact generation, under the rank basis, up to the highest rank shown
    (mo_frames.rank_layer_selection); it reads only its own controls.
    """
    plot_func = get_pareto_plot(paretoFrontPlotType)
    if plot_func is None:
        return go.Figure()

    if paretoFrontPlotType == 'RankLayers':
        layers, run_label = rank_layer_selection(df, stn_rows, rank_run, rank_basis, rank_generation, rank_max)
        return plot_func(layers, run_label, staircase=bool(rank_staircase),
                         uirevision=f"{rank_run}-{layers.get('basis')}")

    attainment = bool(attainment)
    if paretoFrontPlotType in PARETO_NOISY_PAIRS:
        front_a, front_b = PARETO_NOISY_PAIRS[paretoFrontPlotType]
        frontdata, series_labels = pareto_runs(df, stn_rows, lambda view: pareto_pair_entries(view, front_a, front_b))
        return plot_func(frontdata, series_labels, labels=(FRONT_KEYS[front_a][3], FRONT_KEYS[front_b][3]))

    front = selected_front(IndVsDist_IndType)
    if front is None:
        return create_empty_figure(UNSUPPORTED_METRIC_TITLE)
    frontdata, series_labels = pareto_runs(df, stn_rows, lambda view: pareto_front_entries(view, front))

    # Only the plots that accept it receive gridlines=
    extra = {}
    if paretoFrontPlotType in GRIDLINE_PLOTS and invalid_gridlines:
        extra['gridlines'] = knapsack_gridlines(pid)

    # Handle plot-specific arguments
    if paretoFrontPlotType == 'SubplotsMulti':
        fig = plot_func(frontdata, series_labels, nruns=nruns, IndVsDist_IndType=front, attainment=attainment,
                        **extra)
    elif paretoFrontPlotType in FRONT_PLOTS:
        fig = plot_func(frontdata, series_labels, IndVsDist_IndType=front, attainment=attainment, **extra)
    elif paretoFrontPlotType in ('IndVsDist', 'IGDVsDist'):
        fig = plot_func(frontdata, series_labels, distance_method=IndVsDist_DistType, nruns=nruns,
                        IndVsDist_IndType=front)
    elif paretoFrontPlotType == 'MoveCorr':
        fig = plot_func(frontdata, series_labels, IndVsDist_IndType=front, window=windowSize)
    else:
        # PPM, Hist, Scatter: the front type only
        fig = plot_func(frontdata, series_labels, IndVsDist_IndType=front)
    return fig


@app.callback(
    [Output('paretoFrontOptions', 'style'),
     Output('paretoRankLayerOptions', 'style')],
    Input('paretoFrontPlotType', 'value'),
)
def toggle_pareto_plot_options(paretoFrontPlotType):
    """Pareto Rank Layers shows its own controls instead of the front plots' ones; every other type is
    unchanged."""
    if paretoFrontPlotType == RANK_LAYERS_PLOT:
        return {'display': 'none'}, {}
    return {}, {'display': 'none'}


# The rank-layer control props update_rank_layer_controls sets, and their rank_layer_controls keys
RANK_CONTROL_OUTPUTS = (
    ('paretoRankRun', 'options', 'run_options'), ('paretoRankRun', 'value', 'run'),
    ('paretoRankStep', 'max', 'step_max'), ('paretoRankStep', 'marks', 'step_marks'),
    ('paretoRankStep', 'value', 'step'), ('paretoRankStep', 'disabled', 'nav_disabled'),
    ('paretoRankGeneration', 'max', 'gen_max'), ('paretoRankGeneration', 'value', 'gen'),
    ('paretoRankGeneration', 'disabled', 'nav_disabled'), ('paretoRankStepLabel', 'children', 'status'),
    ('paretoRankMax', 'max', 'rank_max'), ('paretoRankMax', 'marks', 'rank_marks'),
    ('paretoRankMax', 'value', 'rank_value'), ('paretoRankMax', 'disabled', 'rank_disabled'),
)


@app.callback(
    [Output('paretoRankRun', 'options'),
     Output('paretoRankRun', 'value'),
     Output('paretoRankStep', 'max'),
     Output('paretoRankStep', 'marks'),
     Output('paretoRankStep', 'value'),
     Output('paretoRankStep', 'disabled'),
     Output('paretoRankGeneration', 'max'),
     Output('paretoRankGeneration', 'value'),
     Output('paretoRankGeneration', 'disabled'),
     Output('paretoRankStepLabel', 'children'),
     Output('paretoRankMax', 'max'),
     Output('paretoRankMax', 'marks'),
     Output('paretoRankMax', 'value'),
     Output('paretoRankMax', 'disabled')],
    [Input('STN_data', 'data'),
     Input('paretoRankRun', 'value'),
     Input('paretoRankBasis', 'value'),
     Input('paretoRankStep', 'value'),
     Input('paretoRankGeneration', 'value'),
     Input('paretoRankFirst', 'n_clicks'),
     Input('paretoRankLast', 'n_clicks')],
    [State('paretoRankMax', 'value'),
     State('paretoRankMax', 'max')],
)
def update_rank_layer_controls(stn_rows, rank_run, rank_basis, rank_step, rank_generation, _first, _last,
                               rank_max_value, rank_max):
    """
    Keep the rank-layer controls in step with the selected rows and with each other (one circular callback:
    the run, step and generation are both its inputs and its outputs). The changed inputs pick the trigger
    (mo_frames.rank_layer_trigger, by priority), and mo_frames.rank_layer_controls the new values; a prop
    whose value does not change is not sent.
    """
    state = rank_layer_controls(df, stn_rows, rank_run, rank_basis, rank_step, rank_generation,
                                rank_layer_trigger(ctx.triggered_prop_ids), rank_max_value, rank_max)
    current = {('paretoRankRun', 'value'): rank_run, ('paretoRankStep', 'value'): rank_step,
               ('paretoRankGeneration', 'value'): rank_generation}
    return tuple(no_update if (component, prop) in current and current[(component, prop)] == state[key]
                 else state[key] for component, prop, key in RANK_CONTROL_OUTPUTS)
