import os
from flask import send_file

from dash import Dash
import dash_bootstrap_components as dbc


from utils.config import load_config

# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, suppress_callback_exceptions=True)

# Set the server attribute for deploying the app
server = app.server

# Application title
app.title = "Audio Deepfake Detection"

# Serve audio files dynamically
@server.route('/audio/<path:filename>')
def serve_audio(filename):
    config = load_config()
    train_raw_audio_path = config['data_paths']['train_raw_audio_path']
    audio_id = filename[4:].split('.')[0] if filename.startswith('F0') else filename.split('.')[0]
    return send_file(os.path.join(train_raw_audio_path, audio_id, filename))

