import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import argparse

def select_important_features(df, features, target_column):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_column], test_size=0.2, random_state=42)
    
    # Initialize and fit the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({'feature': features, 'importance': importances})
    
    # Select features with non-zero importance and sort by importance
    important_features = feature_importances[feature_importances['importance'] > 0].sort_values(by='importance', ascending=False)['feature'].tolist()
    
    return important_features

def remove_highly_correlated_features(df, features, correlation_threshold=0.6):
    # Calculate the correlation matrix
    corr_matrix = df[features].corr().abs()
    
    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
    
    # Initialize a list to keep track of dropped features
    to_drop = set()
    
    # Loop through the features and drop highly correlated ones
    for column in upper.columns:
        if column not in to_drop:
            correlated_features = upper[column][upper[column] > correlation_threshold].index.tolist()
            to_drop.update(correlated_features)
            to_drop.discard(column)  # Ensure the current column is not dropped
    
    # Keep the features not in the drop list
    reduced_features = [feature for feature in features if feature not in to_drop]
    
    return reduced_features

def main():
    parser = argparse.ArgumentParser(description="Feature selection script")
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target column')
    args = parser.parse_args()
    
    # Load the dataset
    df = pd.read_csv(args.input_file)
    
    features = [col for col in df.columns if col not in [args.target_column, 'audio_id']]
    
    # Select important features
    important_features = select_important_features(df, features, args.target_column)
    
    # Remove highly correlated features
    reduced_features = remove_highly_correlated_features(df, important_features)
    
    print(f"Reduced Features: {reduced_features}")

if __name__ == "__main__":
    main()
