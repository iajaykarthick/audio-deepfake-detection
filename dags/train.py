import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import shap
import joblib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

from dag_utils.preprocess_features import handle_missing_values, remove_redundant_features
from dag_utils.plots import plot_roc, plot_feature_importance, plot_calibration_curve, plot_probability_distribution, plot_shap_summary
from dag_utils.model_tracker import evaluate_and_promote_to_staging, promote_best_staging_to_production
from dag_utils.email import generate_and_send_report


# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0
}

# File and model paths configuration
raw_train_csv  = '/app/data/pipeline/training/raw_train.csv'
raw_test_csv   = '/app/data/pipeline/training/raw_test.csv'
train_csv      = '/app/data/pipeline/training/preprocessed_train.csv'
test_csv       = '/app/data/pipeline/training/preprocessed_test.csv'
imputer_path     = '/app/models/imputer.pkl'
scaler_path      = '/app/models/scaler.pkl'
selected_features_path = '/app/models/selected_features.pkl'


model_names = ["Logistic Regression", "Regularized Logistic Regression", "Random Forest", "Calibrated Logistic Regression", "Calibrated Random Forest"]

# Function to find the current training iteration
def find_training_iteration(raw_train_csv_path):
    """
    Determine the current training iteration based on the number of backup files in the directory.
    """
    directory = os.path.dirname(raw_train_csv_path)
    base_filename, base_extension = os.path.splitext(os.path.basename(raw_train_csv_path))
    files_in_directory = os.listdir(directory)
    backup_files = [
        f for f in files_in_directory 
        if f.startswith(base_filename) and (f == base_filename + base_extension or f[len(base_filename):].lstrip('_').rstrip(base_extension).isdigit())
    ]
    training_iteration = len(backup_files) - 1 if base_filename + base_extension in backup_files else len(backup_files)
    print(f"Current training iteration: {training_iteration}")
    return training_iteration

# Find the current training iteration
current_training_iteration = find_training_iteration(raw_train_csv)

# Global configuration for MLflow runs
run_names = {
    f"{model_name}": f"{model_name}_training_{current_training_iteration}" for model_name in model_names
}

# Setup MLflow experiment configuration
mlflow_uri = 'http://mlflow:5001'
experiment_name = 'models_audio_deepfake_detection'
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(experiment_name)



def start_or_resume_run(run_name):
    """
    Start or resume an MLflow run based on the given run name.
    """
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    
    # Search for an existing run with the specified run name
    existing_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )
    
    if existing_runs:
        # Resume the first found run (assume it is the relevant one)
        run_id = existing_runs[0].info.run_id
        print(f"Resuming existing run: {run_id} with name: {run_name}")
    else:
        # Start a new run
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"Starting new run: {run_id} with name: {run_name}")
    
    # Start or resume the run with the specific run_id
    mlflow.start_run(run_id=run_id)
    return run_id



def log_model_info(model, model_name, params, tags, X_test):
    """
    Log model information including parameters, tags, and model registration.
    """
    for key, value in tags.items():
        mlflow.set_tag(key, value)

    # Log parameters
    for key, value in params.items():
        mlflow.log_param(key, value)

    # Infer the model signature
    signature = infer_signature(X_test, model.predict(X_test))

    # Log the model and automatically register it
    try:
        registered_model_name = model_name.replace(" ", "_").lower()
        
        # Log and register the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_name}_model",
            signature=signature,
            registered_model_name=registered_model_name
        )

        print(f"Model {model_name} logged and registered under name: {registered_model_name}")
        
    except Exception as e:
        print(f"Error logging and registering model {model_name}: {str(e)}")

def log_model_performance(model, X_test, y_test, model_name, feature_names):
    """
    Log various performance metrics and plots to MLflow for the given model.
    """
    # Log metrics
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # accuracy
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc_score", roc_auc)
    print(f"{model_name} metrics - Precision: {precision}, Recall: {recall}, F1-score: {f1}, ROC AUC: {roc_auc}")

    # Plot and log ROC Curve
    plot_roc(model, X_test, y_test, model_name)
    roc_curve_path = f"/tmp/{model_name}_roc_curve.png"
    plt.savefig(roc_curve_path)
    plt.close()
    mlflow.log_artifact(roc_curve_path, "plots")
    
    # Plot and log feature importance
    plot_feature_importance(model, feature_names)
    feature_importance_path = f"/tmp/{model_name}_feature_importance.png"
    plt.savefig(feature_importance_path)
    plt.close()
    mlflow.log_artifact(feature_importance_path, "plots")
    
    # Generate and log the classification report as an artifact
    classification_rep = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
    print(f"Classification Report:\n{classification_rep}")
    report_path = f"/tmp/{model_name}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_rep)
    mlflow.log_artifact(report_path, "reports")

    # Generate and log the confusion matrix as an artifact
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"Confusion Matrix - {model_name}")
    conf_matrix_path = f"/tmp/{model_name}_confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # Log the confusion matrix plot
    mlflow.log_artifact(conf_matrix_path, "plots")
    
    # Predicted probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    plot_probability_distribution(y_test, y_probs)
    plt.savefig(f"/tmp/{model_name}_probability_distribution.png")
    plt.close()
    mlflow.log_artifact(f"/tmp/{model_name}_probability_distribution.png", "plots")
    
    # Calibration curve
    plot_calibration_curve({model_name: {'test': y_probs}}, y_test)
    plt.savefig(f"/tmp/{model_name}_calibration_curve.png")
    plt.close()
    mlflow.log_artifact(f"/tmp/{model_name}_calibration_curve.png", "plots")
    
    shap_summary_path = plot_shap_summary(model, X_test, model_name)
    if shap_summary_path:
        mlflow.log_artifact(shap_summary_path, "plots")

def log_model(model, X_test, y_test, model_name, params, tags, feature_names):
    """
    Consolidated function to log both model information and performance.
    """
    run_name = run_names[model_name]
    run_id = start_or_resume_run(run_name)
    # Log model information
    log_model_info(model, model_name, params, tags, X_test)
    
    # Log model performance
    log_model_performance(model, X_test, y_test, model_name, feature_names)
    
    mlflow.end_run()


def save_object(obj, filepath):
    with open(filepath, 'wb') as f:
        joblib.dump(obj, f)

def load_object(filepath):
    with open(filepath, 'rb') as f:
        return joblib.load(f)

def extract_features(**kwargs):
    print("Extracting audio features...")
    # If features_csv is not available, extract features from audio files
    if not os.path.exists(raw_train_csv) or not os.path.exists(raw_test_csv):
        # placeholder to create train csv and test csv
        print('Creating train and test csv')
    else:
        print(f"Audio features already extracted and saved at {raw_train_csv} and {raw_test_csv}")

def preprocess_features(**kwargs):
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print(f"Preprocessed features already exist at {train_csv} and {test_csv}")
        return
    
    print("Preprocessing training features...")
    train_features_df = pd.read_csv(raw_train_csv)
    train_features_df.loc[:, 'target'] = train_features_df['real_or_fake'].apply(lambda x: 0 if x == 'R' else 1)
    print(f"Loaded {train_features_df.shape[0]} training samples with {train_features_df.shape[1] - 3} features each.")
    
    print("Preprocessing testing features...")
    test_features_df = pd.read_csv(raw_test_csv)
    test_features_df.loc[:, 'target'] = test_features_df['real_or_fake'].apply(lambda x: 0 if x == 'R' else 1)
    print(f"Loaded {test_features_df.shape[0]} testing samples with {test_features_df.shape[1] - 3} features each.")
    
    exclude_columns = ['audio_id', 'real_or_fake', 'target']

    # Handle missing values
    train_features_df = handle_missing_values(train_features_df, exclude_columns)
    test_features_df = handle_missing_values(test_features_df, exclude_columns)

    # Persist the imputer for consistent preprocessing
    imputer = SimpleImputer(strategy='mean')
    train_imputed_features = imputer.fit_transform(train_features_df.drop(columns=exclude_columns))
    test_imputed_features = imputer.transform(test_features_df.drop(columns=exclude_columns)) 
    save_object(imputer, imputer_path)
        
    # Standardize features (exclude non-numeric and excluded columns)
    scaler = StandardScaler()
    train_standardized_features = scaler.fit_transform(train_imputed_features)
    test_standardized_features = scaler.transform(test_imputed_features)
    save_object(scaler, scaler_path)
    
    # Convert back to DataFrame and retain exclude columns
    train_standardized_df = pd.DataFrame(train_standardized_features, columns=train_features_df.drop(columns=exclude_columns).columns, index=train_features_df.index)
    train_standardized_df[exclude_columns] = train_features_df[exclude_columns]
    
    test_standardized_df = pd.DataFrame(test_standardized_features, columns=test_features_df.drop(columns=exclude_columns).columns, index=test_features_df.index)
    test_standardized_df[exclude_columns] = test_features_df[exclude_columns]
    
    if not os.path.exists(selected_features_path):
        # Remove redundant features using training data
        cleaned_train_features_df, cleaned_test_features_df, selected_features = remove_redundant_features(train_standardized_df, test_standardized_df, exclude_columns)
    else:
        print("Selected features already exist.")
        selected_features = load_object(selected_features_path)
        cleaned_train_features_df = train_standardized_df[selected_features + exclude_columns]
        cleaned_test_features_df = test_standardized_df[selected_features + exclude_columns]

    # Save selected features for future use
    save_object(selected_features, selected_features_path)
    print(f"Selected features saved to {selected_features_path}")
    
    # Save preprocessed features
    cleaned_train_features_df.to_csv(train_csv, index=False)
    cleaned_test_features_df.to_csv(test_csv, index=False)
    print(f"Preprocessed training features saved to {train_csv}")
    print(f"Preprocessed testing features saved to {test_csv}")


def check_previous_model_versions(**kwargs):
    """
    Check if there are any models in the staging environment for the specified model names.
    """
    client = MlflowClient()
    models_in_staging = []

    for model_name in model_names:
        registered_model_name = model_name.replace(" ", "_").lower()
        try:
            # Search for model versions of the registered model
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            # Check if any version is in staging
            staging_version = next((v for v in versions if v.current_stage == "Staging"), None)
            if staging_version:
                models_in_staging.append(model_name)
        except Exception as e:
            print(f"Error checking model {model_name}: {str(e)}")

    if models_in_staging:
        print(f"Models found in staging: {models_in_staging}")
        return 'test_with_previous_version'
    else:
        print("No models in staging, proceeding to train new models.")
        return 'proceed_to_train_models'

def test_with_previous_version(**kwargs):
    print("Testing with the previous model version...")

    # Load the current preprocessed test data
    test_data = pd.read_csv(test_csv)
    X_test = test_data.drop(columns=['audio_id', 'real_or_fake', 'target'])
    y_test = test_data['target']
    
    # Load the selected features from the previous training
    selected_features = load_object(selected_features_path)

    # Ensure test data has the same features as the training data
    X_test = X_test[selected_features]

    # MLflow client to interact with the model registry
    client = MlflowClient()

    # Iterate over each model name to evaluate the model in staging
    for model_name in model_names:
        registered_model_name = model_name.replace(" ", "_").lower()

        try:
            # Find the latest model version in staging
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            staging_version = next((v for v in versions if v.current_stage == "Staging"), None)

            if staging_version:
                # Load the model from MLflow
                model_uri = f"models:/{registered_model_name}/{staging_version.version}"
                model = mlflow.sklearn.load_model(model_uri)

                print(f"Evaluating {model_name} version {staging_version.version} in staging...")

                # Evaluate the model
                y_pred = model.predict(X_test)
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                accuracy = model.score(X_test, y_test)

                print(f"{model_name} Staging version ROC AUC: {roc_auc}")

                # Log metrics and artifacts to MLflow under the current run name
                run_name = run_names[model_name]
                run_id = start_or_resume_run(run_name)
                mlflow.log_metric("previous_version_accuracy", accuracy)
                mlflow.log_metric("previous_version_precision", precision)
                mlflow.log_metric("previous_version_recall", recall)
                mlflow.log_metric("previous_version_f1_score", f1)
                mlflow.log_metric("previous_version_roc_auc_score", roc_auc)

                # Generate and log the classification report as an artifact
                classification_rep = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
                print(f"Classification Report for {model_name}:\n{classification_rep}")
                report_path = f"/tmp/{model_name}_staging_classification_report.txt"
                with open(report_path, "w") as f:
                    f.write(classification_rep)
                mlflow.log_artifact(report_path, "prev_version_reports")

                # Generate and log the confusion matrix as an artifact
                conf_matrix = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f"Confusion Matrix - {model_name} Staging")
                conf_matrix_path = f"/tmp/{model_name}_staging_confusion_matrix.png"
                plt.savefig(conf_matrix_path)
                plt.close()
                
                # Log the confusion matrix plot
                mlflow.log_artifact(conf_matrix_path, "prev_version_plots")

                # Plot and log ROC Curve
                plot_roc(model, X_test, y_test, f"{model_name} Staging")
                roc_curve_path = f"/tmp/{model_name}_staging_roc_curve.png"
                plt.savefig(roc_curve_path)
                plt.close()
                mlflow.log_artifact(roc_curve_path, "prev_version_plots")

                # Plot and log feature importance if applicable
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    plot_feature_importance(model, selected_features)
                    feature_importance_path = f"/tmp/{model_name}_staging_feature_importance.png"
                    plt.savefig(feature_importance_path)
                    plt.close()
                    mlflow.log_artifact(feature_importance_path, "prev_version_plots")

                # Predicted probabilities
                y_probs = model.predict_proba(X_test)[:, 1]
                plot_probability_distribution(y_test, y_probs)
                plt.savefig(f"/tmp/{model_name}_staging_probability_distribution.png")
                plt.close()
                mlflow.log_artifact(f"/tmp/{model_name}_staging_probability_distribution.png", "prev_version_plots")
                
                # Calibration curve
                plot_calibration_curve({f"{model_name} Staging": {'test': y_probs}}, y_test)
                plt.savefig(f"/tmp/{model_name}_staging_calibration_curve.png")
                plt.close()
                mlflow.log_artifact(f"/tmp/{model_name}_staging_calibration_curve.png", "prev_version_plots")

                mlflow.end_run()
                
            else:
                print(f"No staging version found for {model_name}.")
        except Exception as e:
            print(f"Error testing {model_name} in staging: {str(e)}")

def proceed_to_train_models(**kwargs):
    print("Proceeding to train models...")

def train_models(**kwargs):
    print("Training models...")
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    # Separate features and target
    X_train = train_data.drop(columns=['audio_id', 'real_or_fake', 'target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['audio_id', 'real_or_fake', 'target'])
    y_test = test_data['target']

    # Train a logistic regression model
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    params_logistic = {"max_iter": 1000, "random_state": 42}
    tags_logistic = {"model_type": "Logistic Regression", "run_type": "initial_training", "run_date": datetime.now().strftime("%Y-%m-%d")}
    log_model(logistic_model, X_test, y_test, "Logistic Regression", params_logistic, tags_logistic, X_train.columns)

    # Train a regularized logistic regression model (L2 regularization)
    regularized_model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
    regularized_model.fit(X_train, y_train)
    params_regularized = {"max_iter": 1000, "random_state": 42, "penalty": "l2", "C": 1.0}
    tags_regularized = {"model_type": "Regularized Logistic Regression", "run_type": "initial_training", "run_date": datetime.now().strftime("%Y-%m-%d")}
    log_model(regularized_model, X_test, y_test, "Regularized Logistic Regression", params_regularized, tags_regularized, X_train.columns)

    # Train a random forest model
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)
    params_rf = {"n_estimators": 100, "random_state": 42}
    tags_rf = {"model_type": "Random Forest", "run_type": "initial_training", "run_date": datetime.now().strftime("%Y-%m-%d")}
    log_model(random_forest_model, X_test, y_test, "Random Forest", params_rf, tags_rf, X_train.columns)

    # Calibrate the logistic regression model
    calibrated_logistic_model = CalibratedClassifierCV(logistic_model, method='sigmoid', cv='prefit')
    calibrated_logistic_model.fit(X_train, y_train)
    params_calibrated_logistic = {"method": "sigmoid", "cv": "prefit"}
    tags_calibrated_logistic = {"model_type": "Calibrated Logistic Regression", "run_type": "calibration", "run_date": datetime.now().strftime("%Y-%m-%d")}
    log_model(calibrated_logistic_model, X_test, y_test, "Calibrated Logistic Regression", params_calibrated_logistic, tags_calibrated_logistic, X_train.columns)

    # Calibrate the random forest model
    calibrated_random_forest_model = CalibratedClassifierCV(random_forest_model, method='sigmoid', cv='prefit')
    calibrated_random_forest_model.fit(X_train, y_train)
    params_calibrated_rf = {"method": "sigmoid", "cv": "prefit"}
    tags_calibrated_rf = {"model_type": "Calibrated Random Forest", "run_type": "calibration", "run_date": datetime.now().strftime("%Y-%m-%d")}
    log_model(calibrated_random_forest_model, X_test, y_test, "Calibrated Random Forest", params_calibrated_rf, tags_calibrated_rf, X_train.columns)


def move_to_staging(**kwargs):
    model_names = ["Logistic Regression", "Regularized Logistic Regression", "Random Forest", "Calibrated Logistic Regression", "Calibrated Random Forest"]
    for model_name in model_names:
        evaluate_and_promote_to_staging(model_name)


def move_to_production(**kwargs):
    model_names = ["Logistic Regression", "Regularized Logistic Regression", "Random Forest", "Calibrated Logistic Regression", "Calibrated Random Forest"]
    promote_best_staging_to_production(model_names)

def send_report(**kwargs):
    generate_and_send_report(experiment_name, run_names)


with DAG(
    dag_id='training_pipeline',
    default_args=default_args,
    description='A DAG for engineering audio features and training models',
    schedule_interval=None,
    catchup=False,
) as dag:

    # Define tasks
    extract_features_task = PythonOperator(
        task_id='extract_features',
        python_callable=extract_features,
    )

    preprocess_features_task = PythonOperator(
        task_id='preprocess_features',
        python_callable=preprocess_features,
    )
    
    branch_task = BranchPythonOperator(
        task_id='check_previous_model_versions',
        python_callable=check_previous_model_versions,
    )

    test_with_previous_version_task = PythonOperator(
        task_id='test_with_previous_version',
        python_callable=test_with_previous_version,
    )
    
    dummy_no_previous_version = PythonOperator(
        task_id='proceed_to_train_models',
        python_callable=lambda: 'proceed_to_train_models',
    )
    
    train_models_task = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE
    )
    
    move_to_staging_task = PythonOperator(
        task_id='move_to_staging',
        python_callable=move_to_staging,
    )
    
    move_to_production_task = PythonOperator(
        task_id='move_to_production',
        python_callable=move_to_production,
    )

    generate_report_task = PythonOperator(
        task_id='generate_and_send_report',
        python_callable=send_report,
    )

    ## Define task dependencies
    extract_features_task >> preprocess_features_task >> branch_task
    branch_task >> [test_with_previous_version_task, dummy_no_previous_version]
    test_with_previous_version_task >> train_models_task
    dummy_no_previous_version >> train_models_task
    train_models_task >> move_to_staging_task >> move_to_production_task >> generate_report_task