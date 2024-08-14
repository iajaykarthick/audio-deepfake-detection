import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def handle_missing_values(features_df: pd.DataFrame, exclude_columns) -> pd.DataFrame:
    if features_df.isnull().values.any():
        # Select only numeric columns for imputation, excluding certain columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        impute_cols = [col for col in numeric_cols if col not in exclude_columns]

        # Impute missing values with mean for selected numeric features
        imputer = SimpleImputer(strategy='mean')
        imputed_features = imputer.fit_transform(features_df[impute_cols])
        
        # Convert back to DataFrame and update only the imputed columns
        imputed_df = pd.DataFrame(imputed_features, columns=impute_cols, index=features_df.index)
        features_df.update(imputed_df)
    
    return features_df


def remove_redundant_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    exclude_columns, 
    quasi_constant_threshold: float = 0.99, 
    correlation_threshold: float = 0.9
) -> (pd.DataFrame, pd.DataFrame, list):
    """
    Remove redundant features from both training and testing datasets based on the training data.
    Returns the cleaned DataFrames and the list of selected feature columns.

    Parameters:
    - train_df: The training dataset DataFrame.
    - test_df: The testing dataset DataFrame.
    - exclude_columns: Columns to exclude from feature reduction.
    - quasi_constant_threshold: Threshold for identifying quasi-constant features.
    - correlation_threshold: Threshold for identifying highly correlated features.

    Returns:
    - cleaned_train_df: The cleaned training DataFrame.
    - cleaned_test_df: The cleaned testing DataFrame.
    - selected_features: List of selected feature column names.
    """

    print(f'Initial train shape: {train_df.shape}')
    print(f'Initial test shape: {test_df.shape}')
    
    # Filter out exclude columns from consideration
    feature_cols = [col for col in train_df.columns if col not in exclude_columns]

    # Step 1: Remove constant features
    constant_features = [col for col in feature_cols if train_df[col].nunique() == 1]
    train_df = train_df.drop(columns=constant_features)
    test_df = test_df.drop(columns=constant_features)
    feature_cols = [col for col in feature_cols if col not in constant_features]
    print(f'Removed {len(constant_features)} constant features.')

    # Step 2: Remove quasi-constant features
    quasi_constant_features = [
        col for col in feature_cols 
        if train_df[col].value_counts(normalize=True).values[0] > quasi_constant_threshold
    ]
    train_df = train_df.drop(columns=quasi_constant_features)
    test_df = test_df.drop(columns=quasi_constant_features)
    feature_cols = [col for col in feature_cols if col not in quasi_constant_features]
    print(f'Removed {len(quasi_constant_features)} quasi-constant features.')

    # Step 3: Remove duplicated features
    transposed_df = train_df[feature_cols].T
    duplicated_features = transposed_df[transposed_df.duplicated()].index.tolist()
    train_df = train_df.drop(columns=duplicated_features)
    test_df = test_df.drop(columns=duplicated_features)
    feature_cols = [col for col in feature_cols if col not in duplicated_features]
    print(f'Removed {len(duplicated_features)} duplicated features.')

    # Step 4: Remove highly correlated features
    corr_matrix = train_df[feature_cols].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_features_to_remove = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)
    ]
    train_df = train_df.drop(columns=correlated_features_to_remove)
    test_df = test_df.drop(columns=correlated_features_to_remove)
    feature_cols = [col for col in feature_cols if col not in correlated_features_to_remove]
    print(f'Removed {len(correlated_features_to_remove)} highly correlated features.')

    selected_features = feature_cols # Update selected features after cleaning

    print(f'Final train shape: {train_df.shape}')
    print(f'Final test shape: {test_df.shape}')
    
    return train_df, test_df, selected_features
