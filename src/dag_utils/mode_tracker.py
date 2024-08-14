import mlflow
from mlflow.tracking import MlflowClient


def evaluate_and_promote_to_staging(model_name):
    """
    Evaluate all versions of a registered model and promote the best one to 'Staging' if it improves.

    Args:
        model_name (str): Name of the registered model.
    """
    client = MlflowClient()
    registered_model_name = model_name.replace(" ", "_").lower()

    # Retrieve all model versions
    versions = client.search_model_versions(f"name='{registered_model_name}'")

    if not versions:
        print(f"No models found for {registered_model_name}.")
        return

    best_version = None
    best_metric = float('-inf')  # Assuming higher is better for the metric
    latest_version = None
    latest_metric = None
    first_version = True

    # Evaluate model performance
    for version in versions:
        run_id = version.run_id
        run = client.get_run(run_id)
        
        # Retrieve the metric of interest
        metric = run.data.metrics.get("roc_auc_score")  # Adjust the metric as per your requirement

        if metric:
            # Update best-performing model
            if metric > best_metric:
                best_metric = metric
                best_version = version.version
            
            # Update latest model
            if latest_version is None or version.version > latest_version:
                latest_version = version.version
                latest_metric = metric

            # Check if there's more than one version
            if first_version and version.current_stage != 'None':
                first_version = False

    # Promote to staging if it's the first version or if the latest is the best
    if first_version or (latest_metric and latest_metric > best_metric):
        client.transition_model_version_stage(
            name=registered_model_name,
            version=latest_version,
            stage="Staging"
        )
        print(f"Model {model_name} version {latest_version} promoted to 'Staging'.")
    else:
        print(f"Model {model_name} version {latest_version} not promoted. Current best is version {best_version}.")


def promote_best_staging_to_production(model_names):
    """
    Evaluate models in the 'Staging' stage and promote the best one to 'Production'.

    Args:
        model_names (list): List of registered model names.
    """
    client = MlflowClient()

    best_model_name = None
    best_staging_version = None
    best_staging_metric = float('-inf')

    # Evaluate each model in staging
    for model_name in model_names:
        registered_model_name = model_name.replace(" ", "_").lower()

        # Retrieve models in Staging
        staging_models = client.get_latest_versions(registered_model_name, stages=["Staging"])
        
        # Evaluate performance for each staged model
        for model_version in staging_models:
            run_id = model_version.run_id
            run = client.get_run(run_id)
            
            # Retrieve the metric of interest
            metric = run.data.metrics.get("roc_auc_score") 

            if metric and metric > best_staging_metric:
                best_staging_metric = metric
                best_staging_version = model_version.version
                best_model_name = registered_model_name

    if best_model_name and best_staging_version:
        # Transition the best model version to 'Production'
        client.transition_model_version_stage(
            name=best_model_name,
            version=best_staging_version,
            stage="Production"
        )
        print(f"Model {best_model_name} version {best_staging_version} promoted to 'Production'.")
    else:
        print("No suitable model found in 'Staging' for promotion to 'Production'.")

