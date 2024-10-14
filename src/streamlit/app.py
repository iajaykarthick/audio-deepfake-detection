import streamlit as st
import numpy as np
import pandas as pd
import librosa
from io import BytesIO
import matplotlib.pyplot as plt
from features.extraction.low_level_features_extractor import LowLevelFeatureExtractor
from features.extraction.high_level_features_extractor import HighLevelFeatureExtractor
from models.predict import predict

# Set page layout
st.set_page_config(page_title="Audio Deepfake Detection", layout="wide")

# Add a custom style for background and font
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            font-family: 'Courier New', Courier, monospace;
            color: #493628;
        }
        .section-header {
            font-size: 24px;
            font-weight: bold;
            margin-top: 10px;  /* Reduced margin to minimize vertical gap */
        }
        .confidence-score {
            font-size: 20px;
            font-weight: bold;
            color: #ff6f61;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="title">Audio Deepfake Detection</h1>', unsafe_allow_html=True)
st.write("This application helps you detect whether an audio file is a deepfake or genuine.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

# Extract features from audio
def extract_features(audio_data, sample_rate):
    df = pd.DataFrame({
        'audio_id': [0],
        'audio_arr': [audio_data],
        'srate': [sample_rate],
        'real_or_fake': [0]
    })
    audio_processor = LowLevelFeatureExtractor(target_sr=16000, include_only=['spectral', 'prosodic', 'voice_quality'])
    feature_computer = HighLevelFeatureExtractor()
    low_level_gen = audio_processor.low_level_feature_generator(df)
    high_level_features  = list(feature_computer.high_level_feature_generator(low_level_gen))
    features_df = pd.DataFrame(high_level_features)
    return features_df

# Plot waveform
def plot_waveform(audio_data, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 2))  # Wide and short waveform plot
    ax.plot(np.linspace(0, len(audio_data) / sample_rate, len(audio_data)), audio_data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# Process the uploaded file
if uploaded_file is not None:
    # Use columns to display the audio player, waveform, prediction, and confidence side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Audio")
        st.audio(uploaded_file)

        # Show waveform
        st.subheader("Audio Waveform")
        audio_bytes = uploaded_file.read()
        audio_data, sample_rate = librosa.load(BytesIO(audio_bytes), sr=None)
        plot_waveform(audio_data, sample_rate)

    with col2:
        # Extract features
        features_df = extract_features(audio_data, sample_rate)

        predictions, prediction_probabilities = predict(features_df)

        # Display prediction and confidence score
        st.subheader("Prediction Results")

        prediction = predictions[0]
        confidence_score = prediction_probabilities[0][1] * 100

        if prediction == 1:
            st.error("This audio is classified as a Deepfake!")
        else:
            st.success("This audio is classified as Genuine!")

        # Show confidence score using a progress bar
        st.markdown('<h3 class="confidence-score">Confidence Score</h3>', unsafe_allow_html=True)
        st.progress(confidence_score / 100)

        st.write(f"The model is {confidence_score:.2f}% confident in its prediction.")

# Footer or additional information
st.markdown('<h2 class="section-header">How It Works</h2>', unsafe_allow_html=True)
st.write("""
This app uses machine learning models trained on various audio features, such as spectral, prosodic, and voice quality metrics.
It analyzes the audio to classify whether it is a genuine recording or a deepfake, providing a confidence score for its prediction.
""")
