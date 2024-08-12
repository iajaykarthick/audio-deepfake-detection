import streamlit as st
import pandas as pd
import numpy as np
import base64

from data_loader import fetch_audio_id_to_file_map, get_dataset, create_display_data
from audio_utils import convert_audio_to_wav
from plot_utils import create_psd_plot, create_audio_html

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title("Audio Information Display")

# Fetch the audio_id_to_file_map
username = "ajaykarthick"
dataset_name = "codecfake-audio"
audio_id_to_file_map = fetch_audio_id_to_file_map(username, dataset_name)

# Streamlit layout
col1, col2 = st.columns(2)

with col1:
    audio_ids = list(audio_id_to_file_map.keys())
    audio_id = st.selectbox("Select Audio ID", audio_ids)

with col2:
    playback_rate = st.text_input("Enter Playback Rate", "1.0")

# Ensure the playback rate is a valid float
try:
    playback_rate = float(playback_rate)
except ValueError:
    st.error("Please enter a valid number for the playback rate.")
    playback_rate = 1.0

# Display data in a table layout
if audio_id:
    data = create_display_data(audio_id, audio_id_to_file_map)
    if data:
        for index, row in pd.DataFrame(data).iterrows():
            audio_array = np.array(row['audio'], dtype=np.float32)
            sampling_rate = row['sampling_rate']

            # Convert the audio array to a byte stream using scipy
            virtualfile = convert_audio_to_wav(audio_array, sampling_rate)
            audio_bytes = virtualfile.read()
            b64_audio = base64.b64encode(audio_bytes).decode()

            # Create time series plot
            time_axis = np.arange(audio_array.shape[0]) / sampling_rate

            # Create the main plot and the feature plot
            col_main, col_feature = st.columns(2)

            with col_main:
                # Embed HTML with JavaScript for audio playback
                st.components.v1.html(
                    create_audio_html(index, time_axis, audio_array, row['real_or_fake'], b64_audio, playback_rate, sampling_rate), 
                    height=320
                )

            with col_feature:
                psd_plot = create_psd_plot(audio_array, sampling_rate)
                st.plotly_chart(psd_plot, use_container_width=True)

    else:
        st.write("No data available for the selected audio ID.")
else:
    st.write("Please select an audio ID.")
