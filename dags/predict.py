import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
import mlflow
from mlflow.tracking import MlflowClient
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.email import send_email

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

run_name = "model_prediction_run" # Run name for MLflow tracking

# global variable to store the run id
run_id = None

# Paths for data and model components
# test_csv = '/app/data/pipeline/training/preprocessed_test.csv'
test_csv = '/app/data/pipeline/inference/iteration_8.csv'
selected_features_path = '/app/models/selected_features.pkl'
email_recipient = os.getenv('EMAIL_RECIPIENT', 'ajaykarthick483@hotmail.com') # Email recipient for sending results

imputer_path = '/app/models/imputer.pkl'
scaler_path = '/app/models/scaler.pkl'
selected_features_path = '/app/models/selected_features.pkl'


# MLflow setup
mlflow_uri = 'http://mlflow:5001'
mlflow.set_tracking_uri(mlflow_uri)
experiment_name = "model_prediction_pipeline"
mlflow.set_experiment(experiment_name)

def load_object(filepath):
    """
    Load a joblib object from the given file path.
    """
    with open(filepath, 'rb') as f:
        return joblib.load(f)

def load_objects():
    """
    Load the imputer, scaler, and selected features from disk.
    """
    imputer = load_object(imputer_path)
    scaler = load_object(scaler_path)
    selected_features = load_object(selected_features_path)
    
    return imputer, scaler, selected_features

def preprocess_data(data, imputer, scaler, selected_features, exclude_columns=['audio_id', 'real_or_fake', 'target']):
    """
    Preprocess the input data by imputing missing values and scaling the features.
    """
    test_features_df = data.copy()
    test_features_df.loc[:, 'target'] = data['real_or_fake'].apply(lambda x: 0 if x == 'R' else 1)
    # Impute missing values
    test_imputed_features = imputer.transform(test_features_df.drop(columns=exclude_columns)) 
    test_standardized_features = scaler.transform(test_imputed_features)
    test_standardized_df = pd.DataFrame(test_standardized_features, columns=test_features_df.drop(columns=exclude_columns).columns, index=test_features_df.index)
    test_standardized_df[exclude_columns] = test_features_df[exclude_columns]
    cleaned_test_features_df = test_standardized_df[selected_features + exclude_columns]
    
    return cleaned_test_features_df
    

def predict_and_log_results():
    """
    Load the latest production model, perform predictions, and log results to MLflow.
    """
    client = MlflowClient()
    
    # Load the latest model in production
    registered_models = client.search_registered_models()
    production_model_info = None
    
    for registered_model in registered_models:
        model_name = registered_model.name
        production_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        
        if production_versions:
            production_model_info = production_versions[0]
            break

    if not production_model_info:
        raise ValueError("No production model found.")
    
    model_uri = f"models:/{production_model_info.name}/{production_model_info.version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Load preprocessing objects
    imputer, scaler, selected_features = load_objects()

    # Load test data
    test_data = pd.read_csv(test_csv)
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    test_data = preprocess_data(test_data, imputer, scaler, selected_features)
    
    X_test = test_data.drop(columns=['audio_id', 'real_or_fake', 'target'])
    y_test = test_data['target']
    
    # Load selected features
    selected_features = pd.read_pickle(selected_features_path)
    X_test = X_test[selected_features]

    # Make predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    roc_auc = roc_auc_score(y_test, y_probs)
    classification_rep = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Log metrics and artifacts in MLflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", production_model_info.name)
        mlflow.log_param("model_version", production_model_info.version)
        mlflow.log_metric("roc_auc_score", roc_auc)
        
        # get run id and store it in a global variable
        global run_id
        run_id = mlflow.active_run().info.run_id
        
        
        # Save and log classification report
        report_path = "/tmp/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_rep)
        mlflow.log_artifact(report_path)

        # Save and log confusion matrix plot
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title("Confusion Matrix")
        conf_matrix_path = "/tmp/confusion_matrix.png"
        plt.savefig(conf_matrix_path)
        plt.close()
        mlflow.log_artifact(conf_matrix_path)

        # Generate and log ROC curve plot
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        roc_curve_path = "/tmp/roc_curve.png"
        plt.savefig(roc_curve_path)
        plt.close()
        mlflow.log_artifact(roc_curve_path)

        # Save predictions to CSV
        predictions_df = pd.DataFrame({'audio_id': test_data['audio_id'], 'real_or_fake': test_data['real_or_fake'], 'predicted': y_pred, 'probability': y_probs})
        predictions_path = "/tmp/predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)

def send_results_email(**kwargs):
    """
    Retrieve the latest MLflow run, compile the results, and send an email with the predictions and evaluation metrics.
    """
    client = MlflowClient()

    # Get the experiment ID for the experiment name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    experiment_id = experiment.experiment_id
    print(f"Experiment ID: {experiment_id}")
    
    # access global run_name
    global run_name
    
    print(f'tags.mlflow.runName = "{run_name}"')
    # Retrieve the latest run using the global run_name
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    print(f"Runs: {runs}")
    latest_run = runs[0]
    metrics = latest_run.data.metrics
    artifacts_uri = latest_run.info.artifact_uri.replace("mlflow-artifacts:/", "/app/mlartifacts/")

    # Extract iteration number from the test_csv file name
    iteration_number = None
    if "iteration_" in test_csv:
        iteration_number = int(test_csv.split("_")[-1].split(".")[0]) - 1

    # Prepare email content
    subject = f"Model Prediction Results - Iteration {iteration_number}"
    email_body = f"""
    <h3>Model Prediction Results</h3>
    <p><strong>Model Name:</strong> {latest_run.data.params['model_name']}</p>
    <p><strong>Model Version:</strong> {latest_run.data.params['model_version']}</p>
    <p><strong>ROC AUC Score:</strong> {metrics['roc_auc_score']:.4f}</p>
    <p>Attached are the classification report, confusion matrix plot, ROC curve plot, and predictions CSV.</p>
    """

    # Prepare email attachments
    attachments = [
        os.path.join(artifacts_uri, "classification_report.txt"),
        os.path.join(artifacts_uri, "confusion_matrix.png"),
        os.path.join(artifacts_uri, "roc_curve.png"),
        os.path.join(artifacts_uri, "predictions.csv"),
    ]

    # Send email using Airflow's send_email function
    send_email(
        to=email_recipient,
        subject=subject,
        html_content=email_body,
        files=attachments
    )

# Define the DAG
with DAG(
    dag_id='model_prediction_pipeline',
    default_args=default_args,
    description='A DAG to perform model predictions and send results via email',
    schedule_interval=None,
    catchup=False,
) as dag:
    
    prediction_task = PythonOperator(
        task_id='predict_and_log_results',
        python_callable=predict_and_log_results,
    )

    email_task = PythonOperator(
        task_id='send_results_email',
        python_callable=send_results_email,
    )

    # Define task dependencies
    prediction_task >> email_task
