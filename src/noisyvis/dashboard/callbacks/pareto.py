"""The Pareto-front tab: one figure, dispatched through the Pareto plot registry."""

import plotly.graph_objects as go
from dash import Input, Output

from ..instance import app
from ..data import df
from ..mo_frames import PARETO_NOISY_PAIRS, pareto_front_entries, pareto_pair_entries, pareto_runs
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
     Input('PID', 'data')],
)
def updateParetoPlot(stn_rows, paretoFrontPlotType, IndVsDist_IndType, IndVsDist_DistType, nruns, windowSize,
                     attainment, invalid_gridlines=None, pid=None):
    """
    Update the Pareto front plot based on the selected plot type.
    Uses the plotting registry for dynamic dispatch.

    The fronts are built server-side from the selected rows (STN_data): the "Noisy (A vs B)" plot types
    draw their own two fronts (mo_frames.PARETO_NOISY_PAIRS); every other type draws or measures the front
    type selected in IndVsDist_IndType. The GRIDLINE_PLOTS also draw the selected problem's invalid-solution
    gridlines (knapsack_gridlines) when that option is ticked.
    """
    plot_func = get_pareto_plot(paretoFrontPlotType)
    if plot_func is None:
        return go.Figure()

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
