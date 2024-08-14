import os
import mlflow
from airflow.utils.email import send_email
from mlflow.tracking import MlflowClient

def generate_and_send_report(experiment_name, run_names):
    client = MlflowClient()
    model_names = [
        "Logistic Regression", 
        "Regularized Logistic Regression", 
        "Random Forest", 
        "Calibrated Logistic Regression", 
        "Calibrated Random Forest"
    ]
    
    subject = "Initial Model Training Report"
    report_lines = []
    report_lines.append("""
    <html>
    <head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        hr { border: 1px solid #ddd; margin-top: 20px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f4f4f4; }
        img { margin-bottom: 20px; border: 1px solid #ddd; padding: 5px; }
    </style>
    </head>
    <body>
    """)
    report_lines.append("<h1>Model Training Report</h1>")
    report_lines.append("<hr/>")

    # Collect data and plots for each model
    for model_name in model_names:
        registered_model_name = model_name.replace(" ", "_").lower()
        report_lines.append(f"<h2>{model_name}</h2>")

        # Get the latest model version based on run names
        run_name = run_names.get(model_name)

        if run_name:
            # Retrieve the run based on the run name
            runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'"
            )
            
            if runs:
                # Assume the first run is the relevant one (latest by this name)
                current_run = runs[0]
                current_run_id = current_run.info.run_id

                previous_metrics = {
                    "Accuracy": current_run.data.metrics.get("previous_version_accuracy"),
                    "Precision": current_run.data.metrics.get("previous_version_precision"),
                    "Recall": current_run.data.metrics.get("previous_version_recall"),
                    "F1-score": current_run.data.metrics.get("previous_version_f1_score"),
                    "ROC AUC": current_run.data.metrics.get("previous_version_roc_auc_score"),
                }

                # Determine the subject based on the presence of previous metrics
                if any(metric is not None for metric in previous_metrics.values()):
                    subject = "Model Retraining Report"
                     
                # Collect current metrics
                current_metrics = {
                    "Accuracy": current_run.data.metrics.get("accuracy"),
                    "Precision": current_run.data.metrics.get("precision"),
                    "Recall": current_run.data.metrics.get("recall"),
                    "F1-score": current_run.data.metrics.get("f1_score"),
                    "ROC AUC": current_run.data.metrics.get("roc_auc_score"),
                }

                # Create a metrics comparison table
                report_lines.append("<table>")
                report_lines.append("<tr><th>Metric</th><th>Current Value</th><th>Previous Value</th><th>Change</th></tr>")

                for metric, current_value in current_metrics.items():
                    previous_value = previous_metrics.get(metric)
                    if current_value is not None and previous_value is not None:
                        change = current_value - previous_value
                        change_str = f"{change:.4f}" if change != 0 else "-"
                        report_lines.append(
                            f"<tr><td>{metric}</td><td>{current_value:.4f}</td><td>{previous_value:.4f}</td><td>{change_str}</td></tr>"
                        )
                    elif current_value is not None:
                        report_lines.append(
                            f"<tr><td>{metric}</td><td>{current_value:.4f}</td><td>N/A</td><td>N/A</td></tr>"
                        )

                report_lines.append("</table>")

                # Add plots
                artifact_uri = current_run.info.artifact_uri
                for plot_name in ["roc_curve", "feature_importance", "confusion_matrix", "probability_distribution", "calibration_curve", "shap_summary"]:
                    plot_path = os.path.join(artifact_uri.replace("file://", ""), "plots", f"{model_name}_{plot_name}.png")
                    if os.path.exists(plot_path):
                        report_lines.append(f"<h3>{plot_name.replace('_', ' ').title()}</h3>")
                        report_lines.append(f"<img src='{plot_path}' alt='{plot_name}' style='width:500px;'/>")

                report_lines.append("<hr/>")
            else:
                report_lines.append("<p>No runs found for this model.</p>")
        else:
            report_lines.append("<p>No version available.</p>")

    report_lines.append("</body></html>")
    report_content = "\n".join(report_lines)
    
    send_email_report(report_content, subject=subject)

def send_email_report(report_content, subject="Model Training Report"):
    
    to = ["ajaykarthick483@hotmail.com"]

    send_email(
        to=to,
        subject=subject,
        html_content=report_content
    )
