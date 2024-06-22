from dash.dependencies import Input, Output, State
from dash_app.app import app

# Plots
from dash_app.plots.comparison_plots import generate_plots

# Utils
from dash_utils.comparison_utils import display_searchable_dropdown, print_selected_folder_name


# Callback to update the comparison content based on selected folder
@app.callback(
    Output('comparison-content', 'children'),
    [Input('folder-dropdown', 'value')]
)
def update_comparison_content(selected_folder):
    if selected_folder:
        print_selected_folder_name(selected_folder)
        return generate_plots(selected_folder)
    return "No folder selected"

# Callback to update dropdown options based on search input
@app.callback(
    Output('folder-dropdown', 'options'),
    [Input('folder-dropdown', 'search_value')],
    [State('folder-dropdown', 'value')]
)
def update_dropdown_options(search_value, current_value):
    all_options = display_searchable_dropdown()
    if search_value:
        # Filter options based on the search value
        filtered_options = [opt for opt in all_options if search_value.lower() in opt['label'].lower()]
        return filtered_options
    return all_options
