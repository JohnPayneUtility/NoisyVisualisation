"""The Pareto-front tab: one figure, dispatched through the Pareto plot registry."""

import plotly.graph_objects as go
from dash import Input, Output

from ..instance import app
from ...viz.plots import get_pareto_plot


@app.callback(
    Output("plotParetoFront", "figure"),
    [Input('MO_data_PPP', 'data'),
     Input('STN_MO_series_labels', 'data'),
     Input('paretoFrontPlotType', 'value'),
     Input('IndVsDist_IndType', 'value'),
     Input('IndVsDist_DistType', 'value'),
     Input('paretoPlotNumRuns', 'value'),
     Input('paretoPlotWindowSize', 'value')]
)
def updateParetoPlot(frontdata, series_labels, paretoFrontPlotType, IndVsDist_IndType, IndVsDist_DistType, nruns, windowSize):
    """
    Update the Pareto front plot based on the selected plot type.
    Uses the plotting registry for dynamic dispatch.
    """
    plot_func = get_pareto_plot(paretoFrontPlotType)
    if plot_func is None:
        return go.Figure()

    # Handle plot-specific arguments
    if paretoFrontPlotType == 'SubplotsMulti':
        return plot_func(frontdata, series_labels, nruns=nruns)
    elif paretoFrontPlotType == 'IndVsDist':
        return plot_func(frontdata, series_labels, distance_method=IndVsDist_DistType, nruns=nruns)
    elif paretoFrontPlotType == 'IGDVsDist':
        return plot_func(frontdata, series_labels, distance_method=IndVsDist_DistType, nruns=nruns)
    elif paretoFrontPlotType == 'MoveCorr':
        return plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType, window=windowSize)
    elif paretoFrontPlotType == 'Hist':
        return plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType)
    elif paretoFrontPlotType == 'Scatter':
        return plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType)
    else:
        # Default case: just pass frontdata and series_labels
        return plot_func(frontdata, series_labels)
