import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.metrics import roc_curve, roc_auc_score


# Define function to plot feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(len(indices[:10])), importances[indices[:10]], color="b", align="center")
        plt.yticks(range(len(indices[:10])), [feature_names[i] for i in indices[:10]])
        plt.xlabel("Relative Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
    elif hasattr(model, "coef_"):
        coef = model.coef_[0]
        indices = np.argsort(coef)[::-1]
        plt.figure(figsize=(10, 8))
        plt.title("Feature Coefficients")
        plt.barh(range(len(indices[:10])), coef[indices[:10]], color="b", align="center")
        plt.yticks(range(len(indices[:10])), [feature_names[i] for i in indices[:10]])
        plt.xlabel("Coefficient Value")
        plt.gca().invert_yaxis()
        plt.tight_layout()
    else:
        print("Model does not have feature importances or coefficients attribute.")
        
        
def plot_roc(model, X_test, y_test, model_name):
    """
    Plots the ROC curve for a given model and test dataset.

    Parameters:
    - model: Trained model object.
    - X_test: Test features.
    - y_test: True labels for test data.
    - model_name: Name of the model for plot title.
    """
    # Calculate the ROC curve points
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc='lower right')
    plt.grid()
    
    
def plot_probability_distribution(y_true, y_probs, threshold=0.5):
    # Convert probabilities to predictions
    y_pred = (y_probs >= threshold).astype(int)

    # Separate probabilities for correct and incorrect predictions
    correct_probs = y_probs[y_true == y_pred]
    incorrect_probs = y_probs[y_true != y_pred]

    # Plot distributions
    plt.figure(figsize=(12, 6))
    plt.hist(correct_probs, bins=20, alpha=0.5, label='Correctly Classified', color='green', density=True)
    plt.hist(incorrect_probs, bins=20, alpha=0.5, label='Misclassified', color='red', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution of Predicted Classes')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.show()
    
def plot_calibration_curve(model_to_probs, y_test):
    for model_str, pred_prob_dict in model_to_probs.items():
        pred_probs = pred_prob_dict['test']

        # Create a space for predicted probabilities
        pred_probs_space = np.linspace(pred_probs.min(), pred_probs.max(), 10)

        empirical_probs = []
        pred_probs_midpoints = []

        for i in range(len(pred_probs_space) - 1):
            # Calculate empirical probabilities for each segment of predicted probabilities
            # It calculates the proportion of positive samples in each segment
            empirical_probs.append(np.mean(y_test[(pred_probs > pred_probs_space[i]) & (pred_probs <= pred_probs_space[i+1])]))

            # Calculate midpoints for plotting
            # It calculates the midpoint of each segment of predicted probabilities which gives 
            pred_probs_midpoints.append((pred_probs_space[i] + pred_probs_space[i+1]) / 2)

        # Plot predicted vs. empirical probabilities
        plt.figure(figsize=(10, 4))
        plt.plot(pred_probs_midpoints, empirical_probs, linewidth=2, marker='o')
        plt.title(f"{model_str}", fontsize=20)
        plt.xlabel('predicted prob', fontsize=14)
        plt.ylabel('empirical prob', fontsize=14)

        # Plot ideal calibration line
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

        # Legend and display
        plt.legend(['original', 'ideal'], fontsize=20)
        plt.show()
        
        
def plot_shap_summary(model, X_test, model_name):
    """
    Log SHAP values and plot for a model.

    Args:
        model: Trained model object.
        X_test (DataFrame): Test feature data.
        model_name (str): Name of the model.
    """

    if 'calibrated' not in model_name.lower() and model_name.lower() != 'random forest':
        print(f"Computing SHAP values for {model_name}...")

        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        # Generate SHAP summary plot
        shap.summary_plot(shap_values, X_test)

        # Save the plot
        shap_summary_path = f"/tmp/{model_name}_shap_summary_detailed.png"
        plt.savefig(shap_summary_path, bbox_inches='tight')
        plt.close()

        return shap_summary_path