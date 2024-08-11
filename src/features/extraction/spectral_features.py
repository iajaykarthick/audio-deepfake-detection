import librosa
import numpy as np


class SpectralFeatureExtractor:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr

    def extract(self, features_to_extract=None):

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
        return librosa.feature.spectral_centroid(y=self.y, sr=self.sr).flatten()

    def spectral_bandwidth(self):
        return librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr).flatten()

    def spectral_contrast(self):
        return librosa.feature.spectral_contrast(y=self.y, sr=self.sr).flatten()

    def spectral_flatness(self):
        return librosa.feature.spectral_flatness(y=self.y).flatten()

    def spectral_rolloff(self):
        return librosa.feature.spectral_rolloff(y=self.y, sr=self.sr).flatten()

    def zero_crossing_rate(self):
        return librosa.feature.zero_crossing_rate(self.y).flatten()

    def mfccs(self):
        return librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13).flatten()

    def chroma_stft(self):
        return librosa.feature.chroma_stft(y=self.y, sr=self.sr).flatten()

    def spectral_flux(self):
        S = np.abs(librosa.stft(self.y))
        return np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
