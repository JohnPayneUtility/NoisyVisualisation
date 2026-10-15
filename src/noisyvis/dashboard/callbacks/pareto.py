"""The Pareto-front tab: one figure, dispatched through the Pareto plot registry."""

import plotly.graph_objects as go
from dash import Input, Output, State

from ..instance import app
from ..data import df
from ..mo_frames import front_runs
from ...viz.plots import get_pareto_plot
from ...viz.plots.base import UNSUPPORTED_METRIC_TITLE, create_empty_figure, hv_front

# Plots that draw fronts in objective space: they follow the metric selection and the attainment option
FRONT_PLOTS = ('Basic', 'Subplots', 'SubplotsMulti', 'SubplotsHighlight', 'paretoAnimation')
# Plots of a hypervolume against front movement: they follow the metric selection
HV_PLOTS = ('IndVsDist', 'MoveCorr', 'Hist', 'Scatter')


def _front_of(plot_type, ind_type):
    """The front sequence a plot reads: the metric's front for the front and HV plots, the clean front for
    Progress per Movement (a clean-HV plot), the noisy front for Noisy and IGD Vs Distance."""
    if plot_type in FRONT_PLOTS or plot_type in HV_PLOTS:
        return hv_front(ind_type)
    if plot_type == 'PPM':
        return 'clean'
    return 'noisy'


@app.callback(
    Output("plotParetoFront", "figure"),
    [Input('MO_data_PPP', 'data'),
     Input('STN_MO_series_labels', 'data'),
     Input('paretoFrontPlotType', 'value'),
     Input('IndVsDist_IndType', 'value'),
     Input('IndVsDist_DistType', 'value'),
     Input('paretoPlotNumRuns', 'value'),
     Input('paretoPlotWindowSize', 'value'),
     Input('paretoHistoricalArchive', 'value'),
     Input('paretoAttainmentSurface', 'value')],
    State('STN_data', 'data'),
)
def updateParetoPlot(frontdata, series_labels, paretoFrontPlotType, IndVsDist_IndType, IndVsDist_DistType, nruns,
                     windowSize, historical, attainment, stn_rows):
    """
    Update the Pareto front plot based on the selected plot type.
    Uses the plotting registry for dynamic dispatch.

    The front sequence comes from the current population fronts or, with the historical option, the
    passive archives; the current noisy front is the precomputed MO_data_PPP, every other sequence is
    built server-side from the selected rows.
    """
    plot_func = get_pareto_plot(paretoFrontPlotType)
    if plot_func is None:
        return go.Figure()

    historical, attainment = bool(historical), bool(attainment)
    front = _front_of(paretoFrontPlotType, IndVsDist_IndType)
    if front is None:
        return create_empty_figure(UNSUPPORTED_METRIC_TITLE)
    if historical or front != 'noisy':
        frontdata = front_runs(df, stn_rows, 'historical' if historical else 'current', front)

    # Handle plot-specific arguments
    if paretoFrontPlotType == 'SubplotsMulti':
        fig = plot_func(frontdata, series_labels, nruns=nruns, IndVsDist_IndType=IndVsDist_IndType,
                        attainment=attainment)
    elif paretoFrontPlotType in FRONT_PLOTS:
        fig = plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType, attainment=attainment)
    elif paretoFrontPlotType == 'IndVsDist':
        fig = plot_func(frontdata, series_labels, distance_method=IndVsDist_DistType, nruns=nruns,
                        IndVsDist_IndType=IndVsDist_IndType)
    elif paretoFrontPlotType == 'IGDVsDist':
        fig = plot_func(frontdata, series_labels, distance_method=IndVsDist_DistType, nruns=nruns)
    elif paretoFrontPlotType == 'MoveCorr':
        fig = plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType, window=windowSize)
    elif paretoFrontPlotType == 'Hist':
        fig = plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType)
    elif paretoFrontPlotType == 'Scatter':
        fig = plot_func(frontdata, series_labels, IndVsDist_IndType=IndVsDist_IndType)
    else:
        # Default case: just pass frontdata and series_labels
        fig = plot_func(frontdata, series_labels)

    if historical and fig.layout.title.text:
        fig.update_layout(title_text=f"{fig.layout.title.text} (historical passive archive)")
    return fig
