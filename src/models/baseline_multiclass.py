import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

def load_data(input_file, target_column, selected_features):
    print("Loading data...")
    df = pd.read_csv(input_file)
    X = df[selected_features]
    y = df[target_column]
    return X, y

def preprocess_data(X):
    print("Preprocessing data...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def log_metrics(model, model_name, features, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin, classification_type, classes):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    if classification_type == "Multi-Class":
        roc_auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test_bin, y_proba[:, 1])

    print(f"Model: {model_name}")
    print(f"Classification Type: {classification_type}")
    print(f"Features: {features}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC: {roc_auc}")

    # Plot ROC AUC curve
    plt.figure()
    if classification_type == "Multi-Class":
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (area = {roc_auc_score(y_test_bin[:, i], y_proba[:, i]):.2f})')
    else:
        fpr, tpr, _ = roc_curve(y_test_bin, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classification_type} - ROC AUC Curve')
    plt.legend(loc='lower right')
    plt.grid()

    # Save and log the ROC AUC curve plot
    plot_path = f"roc_auc_curve_{classification_type}_{model_name}.png"
    plt.savefig(plot_path)
    plt.show()

    # Log metrics and artifacts to MLflow
    mlflow.log_param("features", features)
    mlflow.log_param("model", model_name)
    mlflow.log_param("classification_type", classification_type)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_artifact(plot_path)

def main():
    parser = argparse.ArgumentParser(description="Baseline models evaluation script")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target column')
    args = parser.parse_args()
    
    selected_features = ['spectral_contrast_var', 'spectral_contrast_range', 'spectral_contrast_mean', 'F3_mean', 'F2_stdev', 'F3_stdev', 'F1_stdev', 'mfcc_13_std', 'F2_mean', 'mfcc_6_75th_percentile', 'mfcc_12_75th_percentile', 'mfcc_9_75th_percentile', 'mfcc_3_75th_percentile', 'mfcc_12_50th_percentile', 'mfcc_9_50th_percentile', 'mfcc_2_50th_percentile', 'mfcc_5_50th_percentile', 'mfcc_7_50th_percentile', 'f0_skew', 'pause_std', 'asd', 'pause_75th_percentile', 'chroma_11_50th_percentile', 'chroma_3_50th_percentile', 'chroma_6_50th_percentile', 'spectral_flux_skew', 'mfcc_12_25th_percentile', 'mfcc_6_25th_percentile', 'mfcc_2_25th_percentile', 'spectral_bandwidth_min', 'zero_crossing_rate_skew', 'chroma_1_range', 'speaking_rate', 'chroma_12_range', 'chroma_2_range', 'chroma_3_range', 'chroma_5_range', 'chroma_10_range', 'spectral_flatness_skew', 'chroma_6_range', 'chroma_8_range', 'chroma_7_range', 'chroma_9_range', 'f0_kurtosis', 'chroma_11_range', 'spectral_bandwidth_kurtosis', 'chroma_6_max', 'chroma_10_max', 'chroma_2_max', 'chroma_12_max', 'chroma_5_max', 'chroma_7_max', 'chroma_4_max', 'chroma_1_max', 'chroma_11_max', 'chroma_4_std', 'chroma_6_std', 'chroma_7_std', 'chroma_3_max', 'chroma_12_std', 'chroma_11_std', 'chroma_2_std', 'chroma_10_std', 'chroma_3_std', 'chroma_9_std', 'chroma_8_std', 'chroma_5_std', 'chroma_1_std', 'zero_crossing_rate_range', 'mfcc_1_skew', 'spectral_rolloff_range', 'f0_25th_percentile', 'pause_skew', 'chroma_9_min', 'mfcc_13_mean', 'mfcc_11_mean', 'zero_crossing_rate_min', 'spectral_bandwidth_max', 'mfcc_10_max', 'f0_75th_percentile', 'mfcc_5_max', 'mfcc_6_mean', 'mfcc_3_max', 'jitter_local', 'spectral_flux_25th_percentile', 'spectral_flatness_min', 'energy_min', 'shimmer_local', 'spectral_flatness_range']
    
    X, y = load_data(args.input_file, args.target_column, selected_features)
    
    X = preprocess_data(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    classes = np.unique(y)
    num_classes = len(classes)
    
    y_train_bin = label_binarize(y_train, classes=classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of unique classes in y_test: {len(np.unique(y_test))}")

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5001")
    mlflow.set_experiment("multiclass_baseline")

    # Logistic Regression model
    with mlflow.start_run(run_name="Logistic Regression"):
        log_reg = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42)
        log_metrics(log_reg, "Logistic Regression", selected_features, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin, "Multi-Class", classes)
        mlflow.sklearn.log_model(log_reg, "logistic_regression_model")
    
    # Random Forest model
    with mlflow.start_run(run_name="Random Forest"):
        rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        log_metrics(rand_forest, "Random Forest", selected_features, X_train, X_test, y_train, y_test, y_train_bin, y_test_bin, "Multi-Class", classes)
        mlflow.sklearn.log_model(rand_forest, "random_forest_model")
    
    # Save models
    joblib.dump(log_reg, "logistic_regression_model.pkl")
    joblib.dump(rand_forest, "random_forest_model.pkl")

if __name__ == "__main__":
    main()
