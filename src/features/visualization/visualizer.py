import numpy as np
import matplotlib.pyplot as plt
import librosa.display

class Visualizer:
    """
    A class dedicated to visualizing audio data through various types of plots.
    This includes waveforms, spectrograms, and Mel-frequency cepstral coefficients (MFCCs).
    """
    @staticmethod
    def plot_waveform(audio_data, sr, title='Waveform'):
        """
        Displays the waveform of an audio signal.

        Parameters:
        - audio_data (numpy.array): Array containing the audio signal data.
        - sr (int): Sampling rate of the audio signal.
        - title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=sr)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

    @staticmethod
    def plot_waveform_side_by_side(audio_data1, audio_data2, sr1, sr2, supertitle='Waveform Comparison', title1='Real Audio', title2='Fake Audio'):
        """
        Displays two waveforms side by side for comparison.

        Parameters:
        - audio_data1, audio_data2 (numpy.array): Arrays containing the audio signal data.
        - sr1, sr2 (int): Sampling rates of the audio signals.
        - supertitle (str): Super title for the subplot.
        - title1, title2 (str): Titles for the individual subplots.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(supertitle, fontsize=16)
        
        axes[0].set_title(title1)
        librosa.display.waveshow(audio_data1, sr=sr1, ax=axes[0])
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')

        axes[1].set_title(title2)
        librosa.display.waveshow(audio_data2, sr=sr2, ax=axes[1])
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_spectrogram(audio_data, sr, title='Spectrogram'):
        """
        Displays the spectrogram of an audio signal.

        Parameters:
        - audio_data (numpy.array): Array containing the audio signal data.
        - sr (int): Sampling rate of the audio signal.
        - title (str): Title of the plot.
        """
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    @staticmethod
    def plot_spectrogram_side_by_side(audio_data1, audio_data2, sr1, sr2, supertitle='Spectrogram Comparison', title1='Real Audio', title2='Fake Audio'):
        """
        Displays two spectrograms side by side for comparison.

        Parameters:
        - audio_data1, audio_data2 (numpy.array): Arrays containing the audio signal data.
        - sr1, sr2 (int): Sampling rates of the audio signals.
        - supertitle (str): Super title for the subplot.
        - title1, title2 (str): Titles for the individual subplots.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(supertitle, fontsize=16)
        
        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data1)), ref=np.max)
        img1 = librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='log', ax=axes[0])
        axes[0].set_title(title1)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data2)), ref=np.max)
        img2 = librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='log', ax=axes[1])
        axes[1].set_title(title2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_mfcc(mfccs, sr, title='MFCC'):
        """
        Displays the Mel-frequency cepstral coefficients (MFCCs) of an audio signal.

        Parameters:
        - mfccs (numpy.array): Array containing the MFCCs data.
        - sr (int): Sampling rate of the audio signal.
        - title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        plt.show()

    @staticmethod
    def plot_mfcc_side_by_side(mfccs1, mfccs2, sr1, sr2, supertitle='MFCC Comparison', title1='Real Audio', title2='Fake Audio'):
        """
        Displays two sets of MFCCs side by side for comparison.

        Parameters:
        - mfccs1, mfccs2 (numpy.array): Arrays containing the MFCCs data.
        - sr1, sr2 (int): Sampling rates of the audio signals.
        - supertitle (str): Super title for the subplot.
        - title1, title2 (str): Titles for the individual subplots.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(supertitle, fontsize=16)
        
        img1 = librosa.display.specshow(mfccs1, sr=sr1, x_axis='time', ax=axes[0])
        axes[0].set_title(title1)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('MFCC Coefficients')
        fig.colorbar(img1, ax=axes[0])

        img2 = librosa.display.specshow(mfccs2, sr=sr2, x_axis='time', ax=axes[1])
        axes[1].set_title(title2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('MFCC Coefficients')
        fig.colorbar(img2, ax=axes[1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
