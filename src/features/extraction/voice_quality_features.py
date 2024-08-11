import parselmouth
from parselmouth.praat import call
import numpy as np
import math


class VoiceQualityFeatureExtractor:
    """
    A class to extract various voice quality features from audio data.

    Attributes:
        audio_arr (numpy.array): The audio array used for processing.
        orig_sr (int): The original sampling rate of the audio.

    Methods:
        extract(features_to_extract=None): Main method to extract specified voice quality features.
        extract_jitter(): Extracts measures of frequency variation (jitter).
        extract_shimmer(): Extracts measures of amplitude variation (shimmer).
        extract_hnr(): Extracts the Harmonics-to-Noise Ratio (HNR).
        extract_speech_rate(): Calculates various speech rate metrics.
        measure_speech_rate(voiceID): Helper method to perform detailed speech rate analysis.
    """
    def __init__(self, audio_arr, orig_sr):
        """
        Initializes the VoiceQualityFeatureExtractor with audio data.
        """
        self.audio_arr = audio_arr
        self.orig_sr = orig_sr

    def extract(self, features_to_extract=None):
        """
        Extracts specified voice quality features from the audio data.
        
        Args:
            features_to_extract (list of str, optional): A list of feature names to extract.
                Defaults to extracting all available features if None.

        Returns:
            dict: A dictionary containing the extracted features.
        """
        feature_funcs = {
            'jitter': self.extract_jitter,
            'shimmer': self.extract_shimmer,
            'hnr': self.extract_hnr,
            'speech_rate': self.extract_speech_rate
        }

        if features_to_extract is None:
            features_to_extract = feature_funcs.keys()

        features = {}
        for feature in features_to_extract:
            if feature in feature_funcs:
                feature_values = feature_funcs[feature]()
                if isinstance(feature_values, dict):
                    features.update(feature_values)
                else:
                    features[feature] = feature_values
        return features

    def extract_jitter(self):
        """
        Extracts jitter measures from the audio data.
        """
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            return {
                'jitter_local': jitter_local,
                'jitter_rap': jitter_rap,
                'jitter_ppq5': jitter_ppq5
            }
        except Exception as e:
            print(f'Error extracting jitter: {e}')
            return {
                'jitter_local': np.nan,
                'jitter_rap': np.nan,
                'jitter_ppq5': np.nan
            }

    def extract_shimmer(self):
        """
        Extracts shimmer measures from the audio data.
        """
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            return {
                'shimmer_local': shimmer_local,
                'shimmer_apq3': shimmer_apq3,
                'shimmer_apq5': shimmer_apq5,
                'shimmer_dda': shimmer_dda
            }
        except Exception as e:
            print(f'Error extracting shimmer: {e}')
            return {
                'shimmer_local': np.nan,
                'shimmer_apq3': np.nan,
                'shimmer_apq5': np.nan,
                'shimmer_dda': np.nan
            }

    def extract_hnr(self):
        """
        Extracts the Harmonics-to-Noise Ratio (HNR) from the audio data.
        """
        try:
            snd = parselmouth.Sound(self.audio_arr, sampling_frequency=self.orig_sr)
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            return {'hnr': hnr}
        except Exception as e:
            print(f'Error extracting HNR: {e}')
            return {'hnr': np.nan}
        
        
    def extract_speech_rate(self):
        """
        Calculates and extracts various metrics related to speech rate.
        """
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
        """
        Performs a detailed analysis to measure various speech rate metrics from the given audio.

        This method calculates metrics like the number of voiced segments, number of pauses,
        the total original duration of the audio, the duration of voiced segments, speaking rate,
        articulation rate, average syllable duration, and the total duration of pauses.
        """
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

        # get .99 quantile to get maximum (without influence of non-speech sound bursts)
        max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

        # estimate Intensity threshold
        threshold = max_99_intensity + silencedb
        threshold2 = max_intensity - max_99_intensity
        threshold3 = silencedb - threshold2
        if threshold < min_intensity:
            threshold = min_intensity

        # get pauses (silences) and speakingtime
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
        
        # Handling division by zero for articulationrate
        if speakingtot != 0:
            articulationrate = voicedcount / speakingtot
        else:
            articulationrate = float('nan')

        # Handling division by zero for asd
        if voicedcount != 0:
            asd = speakingtot / voicedcount
        else:
            asd = float('nan')  
            
        npause = npauses - 1

        return voicedcount, npause, originaldur, intensity_duration, speakingrate, articulationrate, asd, total_pause_duration
