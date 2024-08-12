import os
import joblib
import pandas as pd
from scipy.stats import ks_2samp
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email
from datetime import datetime, timedelta

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

# Paths
train_path = '/app/data/pipeline/trained/train.csv'
test_path = '/app/data/pipeline/inference/iteration_7.csv'
email_recipient = os.getenv('EMAIL_RECIPIENT', 'ajaykarthick483@hotmail.com') # Email recipient for sending results

# Model paths
imputer_path = '/app/models/imputer.pkl'
scaler_path = '/app/models/scaler.pkl'
selected_features_path = '/app/models/selected_features.pkl'

def load_object(filepath):
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

def feature_distribution_drift(**kwargs):
    """
    Detect feature distribution drift using the K-S test and prepare a report.
    """
    exclude_columns = ['audio_id', 'real_or_fake', 'target']

    # Get new data path from DAG run configuration
    dag_run_conf = kwargs.get('dag_run').conf
    new_data_path = dag_run_conf.get('new_data_path', test_path)
    
    print("New data path:", new_data_path)
    
    # Load training and new data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(new_data_path)
    
    # Load preprocessing objects
    imputer, scaler, selected_features = load_objects()

    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    
    test_data = preprocess_data(test_data, imputer, scaler, selected_features)

    # Ensure that columns with 'mean' are included in both datasets
    mean_features = [col for col in train_data.columns if 'mean' in col]

    # Drop exclude columns and only select mean features for drift analysis
    X_train = train_data.drop(columns=exclude_columns).loc[:, mean_features]
    X_new = test_data.drop(columns=exclude_columns).loc[:, mean_features]

    # Calculate KS statistic and identify drifted features
    drifted_features = []
    drift_info = []

    for feature in X_train.columns:
        # Perform KS test between training and new data for the feature
        statistic, p_value = ks_2samp(X_train[feature], X_new[feature])
        
        # Check if p-value indicates drift (if p-value is less than the threshold, drift is detected)
        if p_value < 0.03:  # Use a stricter threshold as needed
            drifted_features.append(feature)
            drift_info.append({
                'feature': feature,
                'ks_statistic': statistic,
                'p_value': p_value
            })
    
    # Store the drift information in XCom
    kwargs['ti'].xcom_push(key='drift_info', value=drift_info)
    print("Drifted features:", drifted_features)
    print("Drift info:", drift_info)

def send_drift_report_email(**kwargs):
    """
    Send an email with the feature drift report.
    """
    # Retrieve drift information from XCom
    drift_info = kwargs['ti'].xcom_pull(key='drift_info', task_ids='detect_feature_drift')

    # Get new data path from DAG run configuration
    dag_run_conf = kwargs.get('dag_run').conf
    new_data_path = dag_run_conf.get('new_data_path', test_path)
    
    # Extract iteration number from the test_csv file name
    iteration_number = None
    if "iteration_" in new_data_path:
        iteration_number = int(new_data_path.split("_")[-1].split(".")[0]) - 1
        
    # Check if drift_info is not None or empty
    if not drift_info:
        report_html = "<p>No significant feature drift detected.</p>"
        subject = f"No Feature Drift Detected - Iteration {iteration_number}"
        send_email(
            to=email_recipient,
            subject=subject,
            html_content=report_html
        )
        return
        
        
    # Prepare a detailed HTML report
    report_html = "<h3>Feature Drift Report</h3>"
    report_html += "<p>The following features showed significant distribution drift:</p>"
    report_html += "<table border='1' style='border-collapse: collapse;'>"
    report_html += "<tr><th>Feature</th><th>KS Statistic</th><th>P-Value</th></tr>"
    for drift in drift_info:
        report_html += f"<tr><td>{drift['feature']}</td><td>{drift['ks_statistic']:.4f}</td><td>{drift['p_value']:.4f}</td></tr>"
    report_html += "</table>"

    # Email details
    subject = f"Feature Drift Report - Iteration {iteration_number}"
    send_email(
        to=email_recipient,
        subject=subject,
        html_content=report_html
    )
    
# Define the DAG
with DAG(
    dag_id='monitoring_pipeline',
    default_args=default_args,
    description='A DAG to detect feature drift and send a report via email',
    schedule_interval=None,
    catchup=False,
) as dag:

    detect_feature_drift = PythonOperator(
        task_id='detect_feature_drift',
        python_callable=feature_distribution_drift,
        provide_context=True,
    )

    send_drift_report = PythonOperator(
        task_id='send_drift_report',
        python_callable=send_drift_report_email,
        provide_context=True,
    )

    # Define task dependencies
    detect_feature_drift >> send_drift_report
