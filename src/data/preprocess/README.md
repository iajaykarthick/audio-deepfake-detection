# Data Preprocessing Module

This module is designed to handle preprocessing tasks for audio data, focusing on lossless compression of audio files into FLAC or ALAC formats.

## Purpose

The `data.preprocess` module aids in preparing audio files for analysis or storage by compressing them into high-quality, lossless formats. This is crucial for preserving the original audio quality while reducing file size.

## Components

- `audio_compression.py`: Provides functions to compress audio files to FLAC or ALAC formats efficiently.

## Key Function

- **compress_audio_files**: Compresses a list of audio files into the specified lossless codec and stores them in a designated directory.

## Usage Example

```python
from data.preprocess.audio_compression import compress_audio_files

# Specify the list of audio files and the output directory
audio_files = ['path/to/audio1.wav', 'path/to/audio2.wav']
output_dir = 'path/to/compressed_files'

# Compress the files into FLAC format
compress_audio_files(audio_files, output_dir, codec='flac')
```