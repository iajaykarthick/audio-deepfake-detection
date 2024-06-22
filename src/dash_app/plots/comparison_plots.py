import os
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc

from dash_utils.comparison_utils import get_audio_files
from features.feature_extractor import FeatureExtractor


def generate_plots(selected_id, fake_indices=[1, 2]):
    if not selected_id:
        return html.Div("No audio ID selected.")

    # Get the real and fake audio files
    real_audio, fake_audios = get_audio_files(selected_id, fake_indices)

    # Create timestamps
    t_real = np.linspace(0, real_audio.duration, len(real_audio.data))
    t_fakes = [np.linspace(0, fake_audio.duration, len(fake_audio.data)) for fake_audio in fake_audios]

    # Extract features
    real_extractor = FeatureExtractor(real_audio.data, real_audio.sr)
    real_features = real_extractor.extract_all_features()
    fake_extractors = [FeatureExtractor(fake_audio.data, fake_audio.sr) for fake_audio in fake_audios]
    fake_features = [fake_extractor.extract_all_features() for fake_extractor in fake_extractors]

    # Determine number of columns based on available fake audios
    num_fakes = len(fake_audios)
    cols = 1 + num_fakes

    # Create subplots
    subplot_titles = ["Real Audio ID: {}".format(selected_id)]
    subplot_titles += ["Fake Audio ID: F{}_{} ".format(str(idx).zfill(2), selected_id) for idx in fake_indices]
    subplot_titles += ["MFCC"] * cols + ["Spectrogram"] * cols

    specs = [[{"type": "scatter"} for _ in range(cols)]]
    specs += [[{"type": "heatmap"} for _ in range(cols)] for _ in range(2)]

    fig = make_subplots(
        rows=3, cols=cols, shared_xaxes=True, vertical_spacing=0.1, horizontal_spacing=0.05,
        subplot_titles=subplot_titles, specs=specs
    )

    # Add real waveform
    fig.add_trace(go.Scatter(x=t_real, y=real_audio.data, mode='lines', line=dict(color='blue')), row=1, col=1)

    # Add fake waveforms
    for i, t_fake in enumerate(t_fakes):
        fig.add_trace(go.Scatter(x=t_fake, y=fake_audios[i].data, mode='lines', line=dict(color='blue')), row=1, col=i + 2)

    # Add MFCC for real audio
    mfcc_time_real = np.linspace(0, real_audio.duration, real_features['mfcc'].shape[1])
    fig.add_trace(go.Heatmap(z=real_features['mfcc'], x=mfcc_time_real, colorscale='Viridis', coloraxis='coloraxis1'), row=2, col=1)

    # Add MFCC for fake audios
    for i, fake_feature in enumerate(fake_features):
        mfcc_time_fake = np.linspace(0, fake_audios[i].duration, fake_feature['mfcc'].shape[1])
        fig.add_trace(go.Heatmap(z=fake_feature['mfcc'], x=mfcc_time_fake, colorscale='Viridis', coloraxis='coloraxis1'), row=2, col=i + 2)

    # Add Spectrogram for real audio
    spec_time_real = np.linspace(0, real_audio.duration, real_features['spectrogram'].shape[1])
    fig.add_trace(go.Heatmap(z=real_features['spectrogram'], x=spec_time_real, colorscale='Viridis', coloraxis='coloraxis2'), row=3, col=1)

    # Add Spectrogram for fake audios
    for i, fake_feature in enumerate(fake_features):
        spec_time_fake = np.linspace(0, fake_audios[i].duration, fake_feature['spectrogram'].shape[1])
        fig.add_trace(go.Heatmap(z=fake_feature['spectrogram'], x=spec_time_fake, colorscale='Viridis', coloraxis='coloraxis2'), row=3, col=i + 2)

    # Calculate dynamic position and height for coloraxis bars
    num_rows = 3  # Number of rows in the subplot
    coloraxis_height = 1.0 / num_rows  # Height of each coloraxis bar

    # Update layout for coloraxis bars
    fig.update_layout(
        coloraxis1=dict(colorscale='Viridis', showscale=True,
                        colorbar=dict(x=1.05, y=1 - (2 * coloraxis_height) + (coloraxis_height / 2), yanchor='middle', len=coloraxis_height)),
        coloraxis2=dict(colorscale='Viridis', showscale=True,
                        colorbar=dict(x=1.05, y=1 - (3 * coloraxis_height) + (coloraxis_height / 2), yanchor='middle', len=coloraxis_height)),
        height=900,
        margin=dict(t=100, b=50),
        showlegend=False  # Hide the legend
    )

    # Enable ticks (numbers) for all subplots
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    # Create audio player components for subplots
    audio_players = [
        html.Div(html.Audio(id='real-audio-player', src=f"/audio/{os.path.basename(real_audio.file_path)}", controls=True, style={'width': '100%'}), style={'width': '100%', 'display': 'inline-block'})
    ]
    audio_players += [
        html.Div(html.Audio(id=f'fake-audio-player-{i}', src=f"/audio/{os.path.basename(fake_audio.file_path)}", controls=True, style={'width': '100%'}), style={'width': '100%', 'display': 'inline-block'})
        for i, fake_audio in enumerate(fake_audios)
    ]

    # Create the layout with audio players and the graph
    return html.Div(
        [
            html.Div(
                audio_players,
                style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}
            ),
            dcc.Graph(id='audio-plots', figure=fig)
        ]
    )