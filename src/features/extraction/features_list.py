
# Spectral features
DEFAULT_SPECTRAL_FEATURES = [
    'spectral_centroid', 
    'spectral_bandwidth', 
    'spectral_contrast', 
    'spectral_flatness', 
    'spectral_rolloff', 
    'zero_crossing_rate', 
    'mfccs', 
    'chroma_stft', 
    'spectral_flux'
]
ALL_SPECTRAL_FEATURES = [
    'spectral_centroid', 
    'spectral_bandwidth', 
    'spectral_contrast', 
    'spectral_flatness', 
    'spectral_rolloff', 
    'zero_crossing_rate', 
    'mfccs', 
    'chroma_stft', 
    'spectral_flux'
]

# Prosodic features
DEFAULT_PROSODIC_FEATURES = ['f0', 'energy', 'speaking_rate', 'pauses', 'formants']
ALL_PROSODIC_FEATURES = ['f0', 'energy', 'speaking_rate', 'pauses',  'formants']

# Voice Quality Features
DEFAULT_VOICE_QUALITY_FEATURES = [
    'jitter',
    'shimmer',
    'hnr',
    'speech_rate'
]
ALL_VOICE_QUALITY_FEATURES = [
    'jitter_local',	
    'jitter_rap', 
    'jitter_ppq5', 
    'shimmer_local', 
    'shimmer_apq3', 
    'shimmer_apq5', 
    'shimmer_dda', 
    'hnr',
    'voicedcount',
    'npause',
    'intensity_duration',
    'speakingrate',
    'articulationrate',
    'asd',
    'totalpauseduration'
]


# Default features to extract
DEFAULT_FEATURES = {
    'spectral': DEFAULT_SPECTRAL_FEATURES,
    'prosodic': DEFAULT_PROSODIC_FEATURES,
    'voice_quality': DEFAULT_VOICE_QUALITY_FEATURES
}

# All features to extract
ALL_FEATURES = {
    'spectral': ALL_SPECTRAL_FEATURES,
    'prosodic': ALL_PROSODIC_FEATURES,
    'voice_quality': ALL_VOICE_QUALITY_FEATURES
}

