from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import callback_context
from dash_app.callbacks import comparison_callbacks


from dash_app.app import app
from dash_app.layouts import home, comparison

# Initial theme
initial_theme = dbc.themes.BOOTSTRAP

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Real vs Fake Comparison", href="/comparison")),
        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("Select Theme", header=True),
                dbc.DropdownMenuItem("Bootstrap", id="theme-bootstrap"),
                dbc.DropdownMenuItem("Cyborg", id="theme-cyborg"),
                dbc.DropdownMenuItem("Lux", id="theme-lux"),
                dbc.DropdownMenuItem("Slate", id="theme-slate"),
                dbc.DropdownMenuItem("Yeti", id="theme-yeti"),
            ],
            nav=True,
            in_navbar=True,
            label="Theme",
        ),
    ],
    brand="Audio Deepfake Detection",
    brand_href="/",
    color="primary",
    dark=True,
)

# Define the layout with a navbar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='theme-store', data=initial_theme),
    navbar,
    html.Div(id='page-content'),
    html.Link(id='app-stylesheet', rel='stylesheet', href=initial_theme)
])

# Callback to update page content based on URL
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname.startswith('/comparison/'):
        selected_id = pathname.split('/')[-1]
        return comparison.get_comparison_layout(selected_id)
    elif pathname == '/comparison':
        return comparison.layout
    else:
        return home.layout

# Callback to update theme based on dropdown selection
@app.callback(
    Output('theme-store', 'data'),
    [Input('theme-bootstrap', 'n_clicks'),
     Input('theme-cyborg', 'n_clicks'),
     Input('theme-lux', 'n_clicks'),
     Input('theme-slate', 'n_clicks'),
     Input('theme-yeti', 'n_clicks')]
)
def update_theme(*args):
    ctx = callback_context
    if not ctx.triggered:
        return initial_theme
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'theme-cyborg':
            return dbc.themes.CYBORG
        elif button_id == 'theme-lux':
            return dbc.themes.LUX
        elif button_id == 'theme-slate':
            return dbc.themes.SLATE
        elif button_id == 'theme-yeti':
            return dbc.themes.YETI
        else:
            return dbc.themes.BOOTSTRAP

# Callback to update external_stylesheets based on the selected theme
@app.callback(
    Output('app-stylesheet', 'href'),
    [Input('theme-store', 'data')]
)
def update_stylesheet(theme):
    return theme

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
