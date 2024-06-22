from dash import dcc, html
import dash_bootstrap_components as dbc

# Utils
from dash_utils.comparison_utils import display_searchable_dropdown

# Fetch all options from the helper function
all_options = display_searchable_dropdown()

def get_comparison_layout(selected_id=None):
    return dbc.Container(
        [
            dcc.Dropdown(
                id='folder-dropdown',
                options=all_options,
                placeholder="Select an audio ID",
                searchable=True,
                style={'width': '50%', 'margin-bottom': '20px'},
                value=selected_id
            ),
            html.Div(id='comparison-content')
        ],
        className="py-5",
    )

layout = get_comparison_layout()
