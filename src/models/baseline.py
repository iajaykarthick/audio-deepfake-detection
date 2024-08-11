import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, 
                             accuracy_score, precision_score, recall_score)
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn

def load_data(input_file, target_column, selected_features):
    """Load the dataset and extract features and target."""
    df = pd.read_csv(input_file)
    X = df[selected_features]
    y = df[target_column]
    return X, y

def preprocess_data(X):
    """Impute missing values in feature set."""
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(X)

def train_model(X_train, y_train, model):
    """Fit the model to the training data."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Generate various evaluation metrics and plots for the model."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "class_report": classification_report(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "fpr": roc_curve(y_test, y_pred_proba)[0],
        "tpr": roc_curve(y_test, y_pred_proba)[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    return metrics

def save_results(model_name, metrics):
    """Save evaluation metrics and model artifacts."""
    with open(f"{model_name}_classification_report.txt", "w") as f:
        f.write(metrics["class_report"])
    np.savetxt(f"{model_name}_confusion_matrix.csv", metrics["conf_matrix"], delimiter=",", fmt='%d')
    with open(f"{model_name}_roc_auc.txt", "w") as f:
        f.write(f"ROC AUC Score: {metrics['roc_auc']}\n")
    plt.figure()
    plt.plot(metrics["fpr"], metrics["tpr"], label=f'ROC curve (area = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"{model_name}_roc_curve.png")
    mlflow.log_artifacts(f"{model_name}_classification_report.txt")
    mlflow.log_artifacts(f"{model_name}_confusion_matrix.csv")
    mlflow.log_artifacts(f"{model_name}_roc_curve.png")

def configure_mlflow():
    """Configure MLflow tracking URI and set the experiment name."""
    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.set_experiment("binary_baseline")

def run_experiment(model, model_name, X_train, y_train, X_test, y_test):
    """Run MLflow experiment for the given model."""
    with mlflow.start_run(run_name=model_name):
        trained_model = train_model(X_train, y_train, model)
        metrics = evaluate_model(trained_model, X_test, y_test)
        save_results(model_name, metrics)
        mlflow.sklearn.log_model(trained_model, f"{model_name}_model")

def main():
    """Main function to orchestrate the loading, processing, training, and evaluation of models."""
    parser = argparse.ArgumentParser(description="Baseline models evaluation script")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target column')
    args = parser.parse_args()
    
    selected_features = ['spectral_contrast_var', 'spectral_contrast_range', 'spectral_contrast_mean', 'F3_mean', 'F2_stdev', 'F3_stdev', 'F1_stdev', 'mfcc_13_std', 'F2_mean', 'mfcc_6_75th_percentile', 'mfcc_12_75th_percentile', 'mfcc_9_75th_percentile', 'mfcc_3_75th_percentile', 'mfcc_12_50th_percentile', 'mfcc_9_50th_percentile', 'mfcc_2_50th_percentile', 'mfcc_5_50th_percentile', 'mfcc_7_50th_percentile', 'f0_skew', 'pause_std', 'asd', 'pause_75th_percentile', 'chroma_11_50th_percentile', 'chroma_3_50th_percentile', 'chroma_6_50th_percentile', 'spectral_flux_skew', 'mfcc_12_25th_percentile', 'mfcc_6_25th_percentile', 'mfcc_2_25th_percentile', 'spectral_bandwidth_min', 'zero_crossing_rate_skew', 'chroma_1_range', 'speaking_rate', 'chroma_12_range', 'chroma_2_range', 'chroma_3_range', 'chroma_5_range', 'chroma_10_range', 'spectral_flatness_skew', 'chroma_6_range', 'chroma_8_range', 'chroma_7_range', 'chroma_9_range', 'f0_kurtosis', 'chroma_11_range', 'spectral_bandwidth_kurtosis', 'chroma_6_max', 'chroma_10_max', 'chroma_2_max', 'chroma_12_max', 'chroma_5_max', 'chroma_7_max', 'chroma_4_max', 'chroma_1_max', 'chroma_11_max', 'chroma_4_std', 'chroma_6_std', 'chroma_7_std', 'chroma_3_max', 'chroma_12_std', 'chroma_11_std', 'chroma_2_std', 'chroma_10_std', 'chroma_3_std', 'chroma_9_std', 'chroma_8_std', 'chroma_5_std', 'chroma_1_std', 'zero_crossing_rate_range', 'mfcc_1_skew', 'spectral_rolloff_range', 'f0_25th_percentile', 'pause_skew', 'chroma_9_min', 'mfcc_13_mean', 'mfcc_11_mean', 'zero_crossing_rate_min', 'spectral_bandwidth_max', 'mfcc_10_max', 'f0_75th_percentile', 'mfcc_5_max', 'mfcc_6_mean', 'mfcc_3_max', 'jitter_local', 'spectral_flux_25th_percentile', 'spectral_flatness_min', 'energy_min', 'shimmer_local', 'spectral_flatness_range']
    
    # Load the dataset and extract features and target
    X, y = load_data(args.input_file, args.target_column, selected_features)
    y = np.where(y == 'R', 0, 1)
    
    # Preprocess the feature set
    X = preprocess_data(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    configure_mlflow()
    
    # Logistic Regression and Random Forest models
    log_reg = LogisticRegression(solver='liblinear', random_state=42)
    rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    run_experiment(log_reg, "Logistic Regression", X_train, y_train, X_test, y_test)
    run_experiment(rand_forest, "Random Forest", X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
