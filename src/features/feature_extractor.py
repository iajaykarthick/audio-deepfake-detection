import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self, audio_data, sr):
        self.audio_data = audio_data
        self.sr = sr

    def extract_mfcc(self, n_mfcc=13):
        return librosa.feature.mfcc(y=self.audio_data, sr=self.sr, n_mfcc=n_mfcc)

    def extract_chroma(self):
        return librosa.feature.chroma_stft(y=self.audio_data, sr=self.sr)

    def extract_spectral_contrast(self):
        return librosa.feature.spectral_contrast(y=self.audio_data, sr=self.sr)

    def extract_tonnetz(self):
        return librosa.feature.tonnetz(y=librosa.effects.harmonic(self.audio_data), sr=self.sr)
    
    def extract_spectrogram(self):
        return librosa.amplitude_to_db(librosa.stft(self.audio_data), ref=np.max)

    def extract_all_features(self):
        return {
            'mfcc': self.extract_mfcc(),
            'chroma': self.extract_chroma(),
            'spectral_contrast': self.extract_spectral_contrast(),
            'tonnetz': self.extract_tonnetz(),
            'spectrogram': self.extract_spectrogram()
        }
