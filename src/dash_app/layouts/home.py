from dash import html
import dash_bootstrap_components as dbc

layout = html.Div(
    dbc.Container(
        [
            html.H1("Welcome to the Audio Deepfake Detection Dashboard", className="mt-5"),
            html.P(
                "Use this dashboard to explore and analyze audio files for deepfake detection.",
                className="lead",
            ),
            dbc.Button("Learn More", color="primary", href="/comparison"),
        ],
        className="py-5",
    )
)
