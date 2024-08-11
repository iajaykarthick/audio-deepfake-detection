# Features Visualization Module

This module offers essential visualization tools tailored for audio data analysis, particularly useful in the context of detecting and analyzing deepfake audio. It supports various visualization techniques, including basic plots and advanced dimensional reduction visualizations.

## Purpose

The `features.visualization` module is designed to help data scientists and researchers quickly visualize and understand audio data, enabling the identification of patterns that might suggest manipulation.

## Components

### Directory Structure

- `__init__.py`: Makes the visualization classes and functions available for import.
- `feature_analysis_visualization.py`: Provides advanced visualization functions, such as PCA plots and feature distribution comparisons.
- `visualizer.py`: Includes basic tools for waveform and spectrogram visualization, as well as functions for more detailed analyses like MFCC visualizations.

## Key Functions

- **Basic Audio Visualizations**: The `visualizer.py` script allows for quick inspection of audio waveforms and spectrograms.
- **Feature Distribution and PCA Visualizations**: Found in `feature_analysis_visualization.py`, these functions offer deeper insights into the audio features by showing how they vary across different classifications.
