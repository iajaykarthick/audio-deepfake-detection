# Data Quality Checks Module

This module provides a set of functions designed to assess the quality of audio data, ensuring it meets certain standards necessary for accurate analysis and processing.

## Purpose

The `data.quality_checks` module is essential for preliminary audio data evaluation, helping to identify common issues that might affect further analysis, such as in deepfake detection or other audio processing tasks.

## Features

- **Nyquist Compliance Check**: Verifies that the sampling rate is appropriate for the highest frequency present in the audio data according to the Nyquist Theorem.
- **Clipping Detection**: Identifies whether the audio signal has been clipped, which can distort the sound and affect analyses.
- **Silence Detection**: Detects prolonged periods of silence that may indicate recording errors or non-informative sections.
- **SNR Estimation**: Estimates the Signal-to-Noise Ratio (SNR) to evaluate the amount of noise present in the recording compared to the signal.

## Usage Example

```python
from data.quality_checks import data_quality_checks

# Example: Load your audio data
audio_data, sr = librosa.load('path_to_audio_file.wav')

# Perform quality checks
nyquist_compliance = data_quality_checks.check_nyquist_compliance(audio_data, sr)
clipping_detected = data_quality_checks.detect_clipping(audio_data)
silence_detected = data_quality_checks.detect_silence(audio_data, sr)
snr_estimated = data_quality_checks.estimate_snr(audio_data, sr)

print(f"Nyquist Compliance: {nyquist_compliance}")
print(f"Clipping Detected: {clipping_detected}")
print(f"Silence Detected: {silence_detected}")
print(f"SNR Estimated: {snr_estimated} dB")
```