import librosa
import numpy as np


class SpectralFeatureExtractor:
    """
    A class to extract various spectral features from audio data using the librosa library.

    Attributes:
        y (numpy.array): Audio time series.
        sr (int): Sampling rate of the audio time series.

    Methods:
        extract(features_to_extract=None): Extracts specified spectral features from audio.
        spectral_centroid(): Computes the spectral centroid of the audio.
        spectral_bandwidth(): Computes the spectral bandwidth of the audio.
        spectral_contrast(): Computes the spectral contrast of the audio.
        spectral_flatness(): Computes the spectral flatness of the audio.
        spectral_rolloff(): Computes the spectral rolloff of the audio.
        zero_crossing_rate(): Computes the zero crossing rate of the audio.
        mfccs(): Computes the Mel-frequency cepstral coefficients (MFCCs) of the audio.
        chroma_stft(): Computes the chromagram from a waveform or power spectrogram.
        spectral_flux(): Computes the spectral flux of the audio.
    """
    def __init__(self, y, sr):
        """
        Initializes the SpectralFeatureExtractor with audio data.
        """
        self.y = y
        self.sr = sr

    def extract(self, features_to_extract=None):
        """
        Extracts the specified spectral features.
        
        Args:
            features_to_extract (list of str, optional): A list of feature names to extract.
                Defaults to extracting all available features if None.

        Returns:
            dict: A dictionary containing the extracted features.
        """
        feature_funcs = {
            'spectral_centroid': self.spectral_centroid,
            'spectral_bandwidth': self.spectral_bandwidth,
            'spectral_contrast': self.spectral_contrast,
            'spectral_flatness': self.spectral_flatness,
            'spectral_rolloff': self.spectral_rolloff,
            'zero_crossing_rate': self.zero_crossing_rate,
            'mfccs': self.mfccs,
            'chroma_stft': self.chroma_stft,
            'spectral_flux': self.spectral_flux
        }

        if features_to_extract is None:
            features_to_extract = feature_funcs.keys()

        features = {}
        for feature in features_to_extract:
            if feature in feature_funcs:
                features[feature] = feature_funcs[feature]()
        return features

    def spectral_centroid(self):
        """
        Computes the spectral centroid of the audio.
        """
        return librosa.feature.spectral_centroid(y=self.y, sr=self.sr).flatten()

    def spectral_bandwidth(self):
        """
        Computes the spectral bandwidth of the audio.
        """
        return librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr).flatten()

    def spectral_contrast(self):
        """
        Computes the spectral contrast of the audio.
        """
        return librosa.feature.spectral_contrast(y=self.y, sr=self.sr).flatten()

    def spectral_flatness(self):
        """
        Computes the spectral flatness of the audio.
        """
        return librosa.feature.spectral_flatness(y=self.y).flatten()

    def spectral_rolloff(self):
        """
        Computes the spectral rolloff point of the audio.
        """
        return librosa.feature.spectral_rolloff(y=self.y, sr=self.sr).flatten()

    def zero_crossing_rate(self):
        """
        Computes the zero crossing rate of the audio.
        """
        return librosa.feature.zero_crossing_rate(self.y).flatten()

    def mfccs(self):
        """
        Computes the Mel-frequency cepstral coefficients (MFCCs) of the audio.
        """
        return librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13).flatten()

    def chroma_stft(self):
        """
        Computes the chromagram from a waveform or power spectrogram.
        """
        return librosa.feature.chroma_stft(y=self.y, sr=self.sr).flatten()

    def spectral_flux(self):
        """
        Computes the spectral flux of the audio, indicating the rate of change in the power spectrum.
        """
        S = np.abs(librosa.stft(self.y))
        return np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
