import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def select_features(features_df: pd.DataFrame):
    selected_features = ['spectral_contrast_var', 'spectral_contrast_range', 'spectral_contrast_mean', 'F3_mean', 'F2_stdev', 'F3_stdev', 'F1_stdev', 'mfcc_13_std', 'F2_mean', 'mfcc_6_75th_percentile', 'mfcc_12_75th_percentile', 'mfcc_9_75th_percentile', 'mfcc_3_75th_percentile', 'mfcc_12_50th_percentile', 'mfcc_9_50th_percentile', 'mfcc_2_50th_percentile', 'mfcc_5_50th_percentile', 'mfcc_7_50th_percentile', 'f0_skew', 'pause_std', 'asd', 'pause_75th_percentile', 'chroma_11_50th_percentile', 'chroma_3_50th_percentile', 'chroma_6_50th_percentile', 'spectral_flux_skew', 'mfcc_12_25th_percentile', 'mfcc_6_25th_percentile', 'mfcc_2_25th_percentile', 'spectral_bandwidth_min', 'zero_crossing_rate_skew', 'chroma_1_range', 'speaking_rate', 'chroma_12_range', 'chroma_2_range', 'chroma_3_range', 'chroma_5_range', 'chroma_10_range', 'spectral_flatness_skew', 'chroma_6_range', 'chroma_8_range', 'chroma_7_range', 'chroma_9_range', 'f0_kurtosis', 'chroma_11_range', 'spectral_bandwidth_kurtosis', 'chroma_6_max', 'chroma_10_max', 'chroma_2_max', 'chroma_12_max', 'chroma_5_max', 'chroma_7_max', 'chroma_4_max', 'chroma_1_max', 'chroma_11_max', 'chroma_4_std', 'chroma_6_std', 'chroma_7_std', 'chroma_3_max', 'chroma_12_std', 'chroma_11_std', 'chroma_2_std', 'chroma_10_std', 'chroma_3_std', 'chroma_9_std', 'chroma_8_std', 'chroma_5_std', 'chroma_1_std', 'zero_crossing_rate_range', 'mfcc_1_skew', 'spectral_rolloff_range', 'f0_25th_percentile', 'pause_skew', 'chroma_9_min', 'mfcc_13_mean', 'mfcc_11_mean', 'zero_crossing_rate_min', 'spectral_bandwidth_max', 'mfcc_10_max', 'f0_75th_percentile', 'mfcc_5_max', 'mfcc_6_mean', 'mfcc_3_max', 'jitter_local', 'spectral_flux_25th_percentile', 'spectral_flatness_min', 'energy_min', 'shimmer_local', 'spectral_flatness_range']
    print(f"Number of features {len(selected_features)}")
    features_df = features_df[selected_features]
    return features_df

def impute_missing_values(features_df: pd.DataFrame):
    """Impute missing values in feature set."""
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features_df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)
    return features_scaled

def preprocess(features_df: pd.DataFrame):
    features_df = select_features(features_df)
    features_df = impute_missing_values(features_df)
    return features_df

def load_model():
    model_path = '/app/models/logistic_regression_model.pkl'
    model = joblib.load(model_path)
    return model

def predict(features_df: pd.DataFrame):
    features_df = preprocess(features_df)
    model = load_model()
    predictions = model.predict(features_df)
    # Prediction Probabilities
    prediction_probabilities = model.predict_proba(features_df)
    return predictions, prediction_probabilities
