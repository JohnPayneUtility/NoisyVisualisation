"""Schematic tab: the illustrative STN schematic figure and its legend."""

from dash import Input, Output

from ..instance import app
from ..layout import _build_schematic_figure, _build_schematic_legend


@app.callback(
    Output('schematic-graph', 'figure'),
    Output('schematic-legend', 'children'),
    Input('schematic-misjudgements', 'value'),
    Input('schematic-simple-annotations', 'value'),
    Input('schematic-boxplots', 'value'),
)
def update_schematic(misjudgement_values, simple_values, boxplot_values):
    simple = bool(simple_values and 'simple' in simple_values)
    misjudgements = bool(misjudgement_values and 'misjudgements' in misjudgement_values)
    box_plots = bool(boxplot_values and 'boxplots' in boxplot_values)
    return (
        _build_schematic_figure(simple_mode=simple, show_misjudgements=misjudgements, show_box_plots=box_plots),
        _build_schematic_legend(simple_mode=simple, show_misjudgements=misjudgements, show_box_plots=box_plots),
    )
