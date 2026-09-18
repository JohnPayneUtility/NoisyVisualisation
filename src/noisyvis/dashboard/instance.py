"""The dashboard's single Dash application instance.

Its own module so that the callback modules and the entry module share one `app` without importing
the entry module (which runs as `__main__` under `python -m`).
"""

import dash

app = dash.Dash(__name__, suppress_callback_exceptions=True)
# app = dash.Dash(__name__) # Don't suppress exceptions
