from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient

def delete_all_registered_models():
    """
    Deletes all registered models along with their versions from the MLflow registry.
    """
    client = MlflowClient()
    try:
        # Retrieve a list of all registered models
        registered_models = client.search_registered_models()
        
        # Loop through each registered model
        for model in registered_models:
            model_name = model.name
            try:
                # Loop through and delete each version of the current model
                for version in client.search_model_versions(f"name='{model_name}'"):
                    client.delete_model_version(name=model_name, version=version.version)
                
                # After deleting all versions, delete the registered model itself
                client.delete_registered_model(name=model_name)
                print(f"Deleted registered model: {model_name}")
            
            except Exception as e:
                print(f"Error deleting model {model_name}: {str(e)}")
    except Exception as e:
        print(f"Error retrieving registered models: {str(e)}")

# Define default parameters for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 1),
    'retries': 0
}

# Define the DAG
with DAG(
    dag_id='delete_mlflow_models',
    default_args=default_args,
    description='A DAG to delete all models from MLflow registry',
    schedule_interval=None,
    catchup=False,
) as dag:
    
    # Define the PythonOperator to delete models
    delete_models_task = PythonOperator(
        task_id='delete_all_registered_models',
        python_callable=delete_all_registered_models
    )

    delete_models_task
