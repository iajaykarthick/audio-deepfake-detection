# Data Loading Module 

This module, part of a comprehensive framework designed for audio data analysis, specializes in efficiently loading audio data from various datasets. It supports structured data access from datasets like `codecfake` and `wavefake`, ensuring optimized data retrieval for feature extraction and analysis purposes.

## Purpose

The `data.load` module facilitates the streamlined ingestion of audio data, allowing users to efficiently fetch and process audio files necessary for the detection and analysis of deepfake audio. This module is crucial for handling large volumes of data with robust error handling and retry mechanisms.

## Components

### Directory Structure

- `__init__.py`: Initializes the `data.load` package, making its functionalities accessible.
- `data_loader.py`: Central script for fetching and loading audio data based on dataset specifications.
- `codecfake.py`: Contains functions specific to fetching data from the `codecfake` dataset hosted on Hugging Face.
- `wavefake.py`: Similar to `codecfake.py`, but tailored for the `wavefake` dataset.

## Key Features

- **Efficient Data Retrieval**: Implements caching mechanisms to minimize network requests and speed up data loading.
- **Robust Error Handling**: Utilizes retry logic and backoff strategies to handle intermittent network issues or data access problems.
- **Versatile Data Access**: Supports loading specific audio IDs or entire partitions from datasets, offering flexibility depending on the analysis needs.

## Functions

- **Audio ID Listing**: Fetch lists of audio IDs available in the datasets to assist in selective data processing.
- **Audio Data Loading**: Load specific audio data using audio IDs, supporting selective analysis and processing.
- **Parquet Data Handling**: Efficiently load large volumes of data stored in parquet files, suitable for batch processing of audio data.

## Usage Example

```python
from data.load import data_loader

# Fetch list of audio IDs from the codecfake dataset
audio_ids = data_loader.get_codecfake_audio_id_list()

# Load specific audio data
audio_data = data_loader.load_audio_data(audio_ids[:10], dataset='codecfake')

# Process and analyze loaded audio data
process_audio_data(audio_data)
```