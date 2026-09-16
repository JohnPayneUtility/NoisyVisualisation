# app.py
import dash
from dash import html, dcc

app = dash.Dash(__name__, use_pages=True)
app.layout = html.Div([
    dcc.Store(id="selected-run-ids", storage_type="session"),
    html.H1("NoisyVis Dashboard"),
    html.Nav([dcc.Link(p["name"], href=p["path"]) for p in dash.page_registry.values()],
             style={"display":"flex","gap":"1rem","marginBottom":"1rem"}),
    dash.page_container,
])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=True, use_reloader=False)
