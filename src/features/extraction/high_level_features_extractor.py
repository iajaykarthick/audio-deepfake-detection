import numpy as np
from .stat_measures import StatisticalMeasures
from .features_list import ALL_FEATURES


class HighLevelFeatureExtractor:
    """
    A class to extract high-level statistical measures from low-level audio features, which include
    spectral, prosodic, voice quality characteristics and more.

    Attributes:
        measures (list of str): List of statistical measures to compute for each feature.

    Methods:
        compute_high_level_features(feature_dict): Computes high-level features for a given set of audio features.
        high_level_feature_generator(low_level_gen): Generates high-level features from a generator of low-level features.
    """
    def __init__(self, measures=None):
        """
        Initializes the HighLevelFeatureExtractor with a list of statistical measures.

        Args:
            measures (list of str, optional): A list of statistical measures to apply. Default measures include
                                              mean, standard deviation, variance, min, max, range, and percentiles.
        """
        self.measures = measures if measures is not None else ['mean', 'std', 'var', 'min', 'max', 'range', '25th_percentile', '50th_percentile', '75th_percentile', 'skew', 'kurtosis']

    def compute_high_level_features(self, feature_dict):
        """
        Computes high-level features for a dictionary of extracted low-level features.

        Args:
            feature_dict (dict): Dictionary containing low-level audio feature arrays.

        Returns:
            dict: A dictionary containing high-level statistical features.
        """
        features = {
            'audio_id': feature_dict['audio_id'],
            'real_or_fake': feature_dict['real_or_fake']
        }

        # Compute high-level spectral, prosodic, and voice quality features
        features.update(self._compute_spectral_features(feature_dict, ALL_FEATURES['spectral']))
        features.update(self._compute_prosodic_features(feature_dict, ALL_FEATURES['prosodic']))
        features.update(self._compute_voice_quality_features(feature_dict, ALL_FEATURES['voice_quality']))

        return features
    
    def high_level_feature_generator(self, low_level_gen):
        """
        Generator to process each set of low-level features and compute high-level features.

        Args:
            low_level_gen (generator): Generator yielding dictionaries of low-level features.

        Yields:
            dict: High-level features computed from each low-level feature set.
        """
        for low_level_features in low_level_gen:
            yield self.compute_high_level_features(feature_dict=low_level_features)
    
    
    ##################################################################
    ### Additional private methods to compute each type of feature ###
    ##################################################################
    def _compute_spectral_features(self, feature_dict, spectral_features):
        """
        Computes high-level statistical features for spectral features.
        """
        spectral_features_dict = {}
        for feature_name in spectral_features:
            if feature_name in ['mfccs', 'chroma_stft']:
                continue
            feature_array = feature_dict.get(feature_name)
            if feature_array is not None:
                stats = StatisticalMeasures.compute_statistical_measures(feature_array, self.measures)
                spectral_features_dict.update({f"{feature_name}_{key}": value for key, value in stats.items()})
                
        # Compute MFCC Features
        if 'mfccs' in ALL_FEATURES['spectral'] and 'mfccs' in feature_dict:
            spectral_features_dict.update(self._compute_mfcc_features(feature_dict['mfccs']))
            
        # Compute Chroma Features
        if 'chroma_stft' in ALL_FEATURES['spectral'] and 'chroma_stft' in feature_dict:
            spectral_features_dict.update(self._compute_chroma_features(feature_dict['chroma_stft']))
            
        return spectral_features_dict
    
    def _compute_mfcc_features(self, mfccs_flat):
        """
        MFCC features are computed from a flattened array of MFCC coefficients.
        """
        mfccs = mfccs_flat.reshape(-1, 13)  # Reshape the flattened array to its original form
        mfcc_features_dict = {}
        for i in range(mfccs.shape[1]):
            feature_array = mfccs[:, i]
            stats = StatisticalMeasures.compute_statistical_measures(feature_array, self.measures)
            mfcc_features_dict.update({f"mfcc_{i+1}_{key}": value for key, value in stats.items()})
        return mfcc_features_dict
    
    def _compute_chroma_features(self, chroma_flat):
        """
        Chroma features are computed from a flattened array of chroma coefficients.
        """
        chroma = chroma_flat.reshape(-1, 12)  # Reshape the flattened array to its original form
        chroma_features_dict = {}
        for i in range(chroma.shape[1]):
            feature_array = chroma[:, i]
            stats = StatisticalMeasures.compute_statistical_measures(feature_array, self.measures)
            chroma_features_dict.update({f"chroma_{i+1}_{key}": value for key, value in stats.items()})
        return chroma_features_dict
    
    def _compute_prosodic_features(self, feature_dict, prosodic_features):
        """
        Computes high-level statistical features for prosodic features.
        """
        prosodic_features_dict = {}
        for feature_name in prosodic_features:
            if feature_name in ['speaking_rate', 'pauses', 'formants']:
                continue
            feature_array = feature_dict.get(feature_name)
            if isinstance(feature_array, dict):
                raise TypeError(f"Expected array for {feature_name}, but got {type(feature_array).__name__}")
            if feature_array is not None:
                stats = StatisticalMeasures.compute_statistical_measures(feature_array, self.measures)
                prosodic_features_dict.update({f"{feature_name}_{key}": value for key, value in stats.items()})

        if 'speaking_rate' in prosodic_features and 'speaking_rate' in feature_dict:
            prosodic_features_dict['speaking_rate'] = feature_dict['speaking_rate']

        if 'pauses' in prosodic_features and 'pauses' in feature_dict:
            pauses = feature_dict['pauses']
            if pauses:
                pause_durations = np.array([end - start for start, end in pauses])
                pause_stats = StatisticalMeasures.compute_statistical_measures(pause_durations, self.measures)
                prosodic_features_dict.update({f"pause_{key}": value for key, value in pause_stats.items()})
            else:
                for measure in self.measures:
                    prosodic_features_dict[f'pause_{measure}'] = np.nan
                    
        if 'formants' in prosodic_features and 'formants' in feature_dict:
            formant_values = feature_dict['formants']
            if formant_values:
                for key, value in formant_values.items():
                    prosodic_features_dict[key] = value
                
        return prosodic_features_dict
    
    def _compute_voice_quality_features(self, feature_dict, voice_quality_features):
        """
        Computes high-level statistical features for voice quality features.
        """
        voice_quality_features_dict = {}
        for feature_name in voice_quality_features:
            feature_value = feature_dict.get(feature_name)
            if feature_value is not None:
                voice_quality_features_dict[feature_name] = feature_value
        return voice_quality_features_dict
