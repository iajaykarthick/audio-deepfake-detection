import numpy as np
import librosa

def check_nyquist_compliance(audio_data, sr):
    """
    Check if the audio data complies with the Nyquist Theorem.

    Parameters:
    - audio_data (numpy.array): The audio signal array.
    - sr (int): Sampling rate of the audio signal.

    Returns:
    - bool: True if the sampling rate is at least twice the highest frequency in the audio, False otherwise.
    - float: The maximum frequency detected in the audio signal.
    
    The function computes the Fast Fourier Transform (FFT) to find the frequency spectrum,
    identifies the highest frequency component, and checks if the sampling rate is adequate.
    """

    # Compute the FFT of the audio signal
    fft_spectrum = np.fft.rfft(audio_data)
    fft_frequencies = np.fft.rfftfreq(len(audio_data), 1 / sr)

    # Find the maximum frequency component
    max_frequency = fft_frequencies[np.argmax(np.abs(fft_spectrum))]

    # Check if the sampling rate is at least twice the maximum frequency
    is_compliant = sr >= 2 * max_frequency

    return is_compliant, max_frequency

def detect_clipping(audio_data, threshold=0.98):
    """
    Check for clipping in the audio signal.

    Parameters:
    - audio_data (numpy.array): The audio signal array.
    - threshold (float): Threshold to consider a signal value as clipping.

    Returns:
    - bool: True if clipping is detected, False otherwise.
    """
    clipped = np.any(np.abs(audio_data) >= threshold)
    return clipped

def detect_silence(audio_data, sr, threshold=20, duration=0.5):
    """
    Detect long periods of silence in the audio signal.

    Parameters:
    - audio_data (numpy.array): The audio signal array.
    - sr (int): Sampling rate of the audio.
    - threshold (float): The decibel threshold under which a segment is considered silent.
    - duration (float): Minimum duration in seconds for a silent segment to be considered.

    Returns:
    - bool: True if prolonged silence is detected, False otherwise.
    """
    silent = librosa.effects.split(audio_data, top_db=threshold)
    max_silence = max((s[1] - s[0]) for s in silent) / sr
    return max_silence >= duration

def estimate_snr(audio_data, sr, n_fft=2048):
    """
    Estimate the signal-to-noise ratio (SNR) of the audio.

    Parameters:
    - audio_data (numpy.array): The audio signal array.
    - sr (int): Sampling rate of the audio.
    - n_fft (int): Number of FFT components.

    Returns:
    - float: Estimated SNR in decibels (dB).
    """
    S = librosa.stft(audio_data, n_fft=n_fft)
    amplitude = np.abs(S)
    signal_power = np.mean(amplitude**2)
    noise_power = np.var(amplitude)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr