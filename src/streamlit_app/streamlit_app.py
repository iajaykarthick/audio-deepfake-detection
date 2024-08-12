import streamlit as st
import os
import numpy as np
import librosa
import pickle
import soundfile as sf
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, ClientSettings
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
import parselmouth
from parselmouth.praat import call
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import math
# import tensorflow as tf
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import jax
from tensorflow.keras.models import load_model
print(tf.__version__)
print(jax.__version__)


st.set_page_config(layout="wide")

# Improved Custom CSS for background color and fonts
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #FFD700;
        font-family: 'Calibri', sans-serif;
    }
    .result-container {
        text-align: center;
        margin-top: 20px;
    }
    .result-text {
        font-size: 26px;
        font-weight: bold;
        color: #58A4B0;
    }
    .confidence-text {
        font-size: 22px;
        color: #DAF7A6;
    }
    h1 {
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 50px;
        color: #FFD700;
    }
    .st-bb {
        background-color: #333333;
        color: #FFFFFF;
    }
    .st-at, .st-at .uploadButton {
        background-color: #444444 !important;
        color: #FFFFFF !important;
    }
    .stTextInput label, .stButton>button {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# # Load scaler and model
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Load the neural network model from JSON file
# with open('neural_network_model.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
# nn_model = load_model('neural_network_model.weights.h5')

# #nn_model = model_from_json(loaded_model_json)
# nn_model.load_weights('neural_network_model_weights.h5')
# nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the model architecture
with open('neural_network_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    nn_model = model_from_json(loaded_model_json)

# Load the weights
nn_model.load_weights('neural_network_model.weights.h5')

# Define feature extraction classes and methods here
class VoiceQualityFeatureExtractor:
    def __init__(self, audio_arr, orig_sr):
        self.audio_arr = audio_arr
        self.orig_sr = orig_sr

    def extract(self):
        features = {}
        features.update(self.extract_jitter())
        features.update(self.extract_shimmer())
        features.update(self.extract_hnr())
        features.update(self.extract_speech_rate())
        return features

    def extract_jitter(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            return {'jitter_local': jitter_local, 'jitter_rap': jitter_rap, 'jitter_ppq5': jitter_ppq5}
        except Exception as e:
            print(f'Error extracting jitter: {e}')
            return {'jitter_local': np.nan, 'jitter_rap': np.nan, 'jitter_ppq5': np.nan}

    def extract_shimmer(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            return {'shimmer_local': shimmer_local, 'shimmer_apq3': shimmer_apq3, 'shimmer_apq5': shimmer_apq5, 'shimmer_dda': shimmer_dda}
        except Exception as e:
            print(f'Error extracting shimmer: {e}')
            return {'shimmer_local': np.nan, 'shimmer_apq3': np.nan, 'shimmer_apq5': np.nan, 'shimmer_dda': np.nan}

    def extract_hnr(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            return {'hnr': hnr}
        except Exception as e:
            print(f'Error extracting HNR: {e}')
            return {'hnr': np.nan}

    def extract_speech_rate(self):
        try:
            sound = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            (voicedcount, npause, originaldur, intensity_duration, speakingrate, articulationrate, asd, totalpauseduration) = self.measure_speech_rate(sound)
            return {
                'voicedcount': voicedcount,
                'npause': npause,
                'originaldur': originaldur,
                'intensity_duration': intensity_duration,
                'speakingrate': speakingrate,
                'articulationrate': articulationrate,
                'asd': asd,
                'totalpauseduration': totalpauseduration
            }
        except Exception as e:
            print(f'Error extracting speech rate: {e}')
            return {
                'voicedcount': np.nan,
                'npause': np.nan,
                'originaldur': np.nan,
                'intensity_duration': np.nan,
                'speakingrate': np.nan,
                'articulationrate': np.nan,
                'asd': np.nan,
                'totalpauseduration': np.nan
            }

    def measure_speech_rate(self, voiceID):
        silencedb = -25
        mindip = 2
        minpause = 0.3
        
        sound = parselmouth.Sound(voiceID)
        originaldur = sound.get_total_duration()
        intensity = sound.to_intensity(50)
        start = call(intensity, "Get time from frame number", 1)
        nframes = call(intensity, "Get number of frames")
        end = call(intensity, "Get time from frame number", nframes)
        min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
        max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

        max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

        threshold = max_99_intensity + silencedb
        threshold2 = max_intensity - max_99_intensity
        threshold3 = silencedb - threshold2
        if threshold < min_intensity:
            threshold = min_intensity

        textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
        silencetier = call(textgrid, "Extract tier", 1)

        silencetable = call(silencetier, "Down to TableOfReal", "sounding")
        npauses = call(silencetable, "Get number of rows")

        speakingtot = 0
        for ipause in range(npauses):
            pause = ipause + 1
            beginsound = call(silencetable, "Get value", pause, 1)
            endsound = call(silencetable, "Get value", pause, 2)
            speakingdur = endsound - beginsound
            speakingtot += speakingdur
        total_pause_duration = originaldur - speakingtot

        intensity_matrix = call(intensity, "Down to Matrix")
        sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
        intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
        intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
        point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
        numpeaks = call(point_process, "Get number of points")
        t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

        timepeaks = []
        peakcount = 0
        intensities = []
        for i in range(numpeaks):
            value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
            if value > threshold:
                peakcount += 1
                intensities.append(value)
                timepeaks.append(t[i])

        validpeakcount = 0
        currenttime = timepeaks[0]
        currentint = intensities[0]
        validtime = []

        for p in range(peakcount - 1):
            following = p + 1
            followingtime = timepeaks[p + 1]
            dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
            diffint = abs(currentint - dip)
            if diffint > mindip:
                validpeakcount += 1
                validtime.append(timepeaks[p])
            currenttime = timepeaks[following]
            currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

        pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
        voicedcount = 0
        voicedpeak = []

        for time in range(validpeakcount):
            querytime = validtime[time]
            whichinterval = call(textgrid, "Get interval at time", 1, querytime)
            whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
            value = pitch.get_value_at_time(querytime) 
            if not math.isnan(value):
                if whichlabel == "sounding":
                    voicedcount += 1
                    voicedpeak.append(validtime[time])

        timecorrection = originaldur / intensity_duration
        call(textgrid, "Insert point tier", 1, "syllables")
        for i in range(len(voicedpeak)):
            position = (voicedpeak[i] * timecorrection)
            call(textgrid, "Insert point", 1, position, "")

        speakingrate = voicedcount / originaldur
        articulationrate = voicedcount / speakingtot
        npause = npauses - 1
        asd = speakingtot / voicedcount

        return voicedcount, npause, originaldur, intensity_duration, speakingrate, articulationrate, asd, total_pause_duration

class StatisticalMeasures:
    @staticmethod
    def compute_statistical_measures(feature_array, prefix):
        measures = ['mean', 'std', 'var', 'min', 'max', 'range', '25th_percentile', '50th_percentile', '75th_percentile', 'skew', 'kurtosis']
        stats = {}
        if len(feature_array) == 0:
            return {f"{prefix}_{measure}": np.nan for measure in measures}
        
        stats[f'{prefix}_mean'] = np.mean(feature_array)
        stats[f'{prefix}_std'] = np.std(feature_array)
        stats[f'{prefix}_var'] = np.var(feature_array)
        stats[f'{prefix}_min'] = np.min(feature_array)
        stats[f'{prefix}_max'] = np.max(feature_array)
        stats[f'{prefix}_range'] = np.ptp(feature_array)
        stats[f'{prefix}_25th_percentile'] = np.percentile(feature_array, 25)
        stats[f'{prefix}_50th_percentile'] = np.percentile(feature_array, 50)
        stats[f'{prefix}_75th_percentile'] = np.percentile(feature_array, 75)
        if len(np.unique(feature_array)) > 1:
            stats[f'{prefix}_skew'] = skew(feature_array)
            stats[f'{prefix}_kurtosis'] = kurtosis(feature_array)
        else:
            stats[f'{prefix}_skew'] = np.nan
            stats[f'{prefix}_kurtosis'] = np.nan
        return stats

class SpectralFeatureExtractor:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr

    def extract(self):
        features = {}
        features.update(self.extract_and_compute('spectral_centroid', self.spectral_centroid()))
        features.update(self.extract_and_compute('spectral_bandwidth', self.spectral_bandwidth()))
        features.update(self.extract_and_compute('spectral_contrast', self.spectral_contrast()))
        features.update(self.extract_and_compute('spectral_flatness', self.spectral_flatness()))
        features.update(self.extract_and_compute('spectral_rolloff', self.spectral_rolloff()))
        features.update(self.extract_and_compute('zero_crossing_rate', self.zero_crossing_rate()))
        features.update(self.extract_and_compute('spectral_flux', self.spectral_flux()))
        mfccs = self.mfccs()
        for i, mfcc in enumerate(mfccs):
            features.update(self.extract_and_compute(f'mfcc_{i+1}', mfcc))
        chroma = self.chroma_stft()
        for i, c in enumerate(chroma):
            features.update(self.extract_and_compute(f'chroma_{i+1}', c))
        return features

    def extract_and_compute(self, feature_name, feature_array):
        return StatisticalMeasures.compute_statistical_measures(feature_array, feature_name)

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
        return librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)

    def chroma_stft(self):
        return librosa.feature.chroma_stft(y=self.y, sr=self.sr)

    def spectral_flux(self):
        S = np.abs(librosa.stft(self.y))
        return np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))

class ProsodicFeatureExtractor:
    def __init__(self, y, sr, audio_arr, orig_sr):
        self.y = y
        self.sr = sr
        self.audio_arr = audio_arr
        self.orig_sr = orig_sr

    def extract(self):
        features = {}
        features.update(self.extract_and_compute('f0', self.extract_f0()))
        features.update(self.extract_and_compute('energy', self.extract_energy()))
        features['speaking_rate'] = self.extract_speaking_rate()
        features['pauses'] = self.extract_pauses()
        features.update(self.extract_formants())
        return features

    def extract_and_compute(self, feature_name, feature_array):
        return StatisticalMeasures.compute_statistical_measures(feature_array, feature_name)

    def extract_f0(self):
        f0, voiced_flag, voiced_probs = librosa.pyin(self.y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = np.nan_to_num(f0)
        return f0

    def extract_energy(self):
        return librosa.feature.rms(y=self.y).flatten()

    def extract_speaking_rate(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            total_duration = snd.get_total_duration()
            intensity = snd.to_intensity()
            intensity_values = intensity.values.T
            threshold = 0.3 * max(intensity_values)
            syllable_count = len([1 for i in intensity_values if i > threshold])
            speaking_rate = syllable_count / total_duration
            return speaking_rate
        except Exception as e:
            print(f'Error extracting speaking rate: {e}')
            return np.nan

    def extract_pauses(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            silences = call(snd, "To TextGrid (silences)", 100, 0, -25, 0.1, 0.1, "silent", "sounding")
            pauses = [(call(silences, "Get start time of interval", 1, i), call(silences, "Get end time of interval", 1, i)) for i in range(1, call(silences, "Get number of intervals", 1) + 1) if call(silences, "Get label of interval", 1, i) == "silent"]
            return len(pauses)
        except Exception as e:
            print(f'Error extracting pauses: {e}')
            return np.nan
        
    def extract_formants(self):
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            formant = call(snd, "To Formant (burg)", 0.025, 5, 5500, 0.025, 50)
            formant_values = {}
            for i in range(1, 4):
                formant_values[f'F{i}_mean'] = call(formant, "Get mean", i, 0, 0, "Hertz")
                formant_values[f'F{i}_stdev'] = call(formant, "Get standard deviation", i, 0, 0, "Hertz")
            return formant_values
        except Exception as e:
            print(f'Error extracting formants: {e}')
            return {}

class LowLevelFeatureExtractor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def resample_audio(self, audio_arr, orig_sr):
        return librosa.resample(audio_arr, orig_sr=orig_sr, target_sr=self.target_sr)

    def extract_features(self, audio_id, audio_arr, orig_sr, real_or_fake):
        y = self.resample_audio(audio_arr, orig_sr)
        
        features = {
            'audio_id': audio_id,
            'real_or_fake': real_or_fake
        }
        
        # Assume SpectralFeatureExtractor, ProsodicFeatureExtractor, VoiceQualityFeatureExtractor are defined and imported
        spectral_extractor = SpectralFeatureExtractor(y, self.target_sr)
        features.update(spectral_extractor.extract())

        prosodic_extractor = ProsodicFeatureExtractor(y, self.target_sr, audio_arr, orig_sr)
        features.update(prosodic_extractor.extract())
        
        voice_quality_extractor = VoiceQualityFeatureExtractor(audio_arr, orig_sr)
        features.update(voice_quality_extractor.extract())

        return features

    def process_audio_files(self, filepaths):
        features = []
        for filepath in filepaths:
            audio_id = os.path.basename(filepath)
            audio_arr, orig_sr = librosa.load(filepath, sr=None)
            real_or_fake = "Unknown"  # Adjust as necessary if you have this info
            features.append(self.extract_features(audio_id, audio_arr, orig_sr, real_or_fake))
        return pd.DataFrame(features)


st.header("Upload or Record Audio")
audio_file = st.file_uploader("Upload an audio file", type=["wav"])

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        self.frames.append(frame.to_ndarray().flatten())
        return frame

    def save_wav(self, filename):
        if self.frames:
            audio_data = np.concatenate(self.frames)
            audio_data = audio_data / np.max(np.abs(audio_data))  
            sf.write(filename, audio_data, 48000)

st.header("Record Live Audio")
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    if st.button("Save Recording"):
        audio_processor = webrtc_ctx.audio_processor
        if audio_processor:
            audio_path = "recorded_audio.wav"
            audio_processor.save_wav(audio_path)
            st.success(f"Recording saved to {audio_path}")
            audio_file = audio_path  # Set this so we can process the saved file

if audio_file is not None:
    if isinstance(audio_file, str):
        audio_file_path = audio_file
    else:
        audio_file_path = os.path.join('uploads', audio_file.name)
        with open(audio_file_path, 'wb') as f:
            f.write(audio_file.read())

    extractor = LowLevelFeatureExtractor()
    features_df = extractor.process_audio_files([audio_file_path])

    # Scale the features
    features_scaled = scaler.transform(features_df.drop(columns=['audio_id', 'real_or_fake', 'pauses']))
    print(features_scaled)
    
    predictions = nn_model.predict(features_scaled).flatten()
    result = "Real" if predictions[0] > 0.5 else "Fake"
    confidence_value = 1 - predictions[0] if predictions[0] < 0.5 else predictions[0]
    confidence = 'High' if confidence_value > 0.8 else 'Low' if confidence_value < 0.3 else 'Moderate'

    def predict_proba(X):
        preds = nn_model.predict(X)
        return np.hstack((1 - preds, preds))
    
    
    random_data = np.random.rand(100, features_df.drop(columns=['audio_id', 'real_or_fake', 'pauses']).shape[1])

    # Scale the random data using the pre-fitted scaler to ensure
    # the synthetic data has the same scaling as your actual dataset.
    training_data = scaler.transform(random_data)

    # Initialize the LIME explainer with the scaled random data.
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=features_df.columns,
        class_names=['Fake', 'Real'],
        mode='classification'
    )

    exp = explainer.explain_instance(features_scaled[0], predict_proba, num_features=10)

    fig_lime = exp.as_pyplot_figure()
    fig_lime.set_size_inches(4, 5)
    ax = fig_lime.gca()

    # Correcting the ytick labels
    ytick_labels = []
    for tick in ax.get_yticklabels():
        match = re.search(r'(\d+)', tick.get_text())
        if match:
            index = int(match.group(1))
            ytick_labels.append(features_df.columns[index])
        else:
            ytick_labels.append(tick.get_text())

    ax.set_yticklabels(ytick_labels)

    st.markdown(
        f"""
        <div class="result-container">
            <div class="result-text">The audio is classified as: {result}</div>
            <div class="confidence-text">with {confidence} confidence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    y, sr = librosa.load(audio_file_path, sr=22050)



    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fig_waveform = go.Figure()
        fig_waveform.add_trace(go.Scatter(y=y, mode='lines', name='Waveform'))
        fig_waveform.update_layout(title='Waveform', xaxis_title='Time', yaxis_title='Amplitude')
        st.plotly_chart(fig_waveform)

    with col2:
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        fig_spectrogram = go.Figure(data=go.Heatmap(z=S_DB, x=np.arange(S_DB.shape[1]), y=librosa.mel_frequencies(n_mels=S_DB.shape[0]), colorscale='Viridis'))
        fig_spectrogram.update_layout(title='Spectrogram', xaxis_title='Time', yaxis_title='Frequency (Hz)')
        st.plotly_chart(fig_spectrogram)

    with col3:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        fig_mfccs = go.Figure(data=go.Heatmap(z=mfccs, x=np.arange(mfccs.shape[1]), y=np.arange(1, mfccs.shape[0] + 1), colorscale='Viridis'))
        fig_mfccs.update_layout(title='MFCCs', xaxis_title='Time', yaxis_title='MFCC Coefficients')
        st.plotly_chart(fig_mfccs)

    with col4:
        st.pyplot(fig_lime)