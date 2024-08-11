import os
import pandas as pd
import numpy as np
import sys
from utils.config import load_config


def remove_redundant_features(features_df: pd.DataFrame, quasi_constant_threshold: float = 0.99, correlation_threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove constant, quasi-constant, duplicated, and highly correlated features from the DataFrame.

    Parameters:
        features_df (pd.DataFrame): DataFrame containing the feature data.
        quasi_constant_threshold (float): Threshold for identifying quasi-constant features. Default is 0.99.
        correlation_threshold (float): Threshold for identifying highly correlated features. Default is 0.9.

    Returns:
        pd.DataFrame: DataFrame with redundant features removed.
    """
    
    # Step 1: Remove constant features
    constant_features = [col for col in features_df.columns if features_df[col].nunique() == 1]
    features_df = features_df.drop(columns=constant_features)
    print(f'Removed {len(constant_features)} constant features.')

    # Step 2: Remove quasi-constant features
    quasi_constant_features = [
        col for col in features_df.columns 
        if features_df[col].value_counts(normalize=True).values[0] > quasi_constant_threshold
    ]
    features_df = features_df.drop(columns=quasi_constant_features)
    print(f'Removed {len(quasi_constant_features)} quasi-constant features.')

    # Step 3: Remove duplicated features
    transposed_df = features_df.T
    duplicated_features = transposed_df[transposed_df.duplicated()].index.tolist()
    features_df = features_df.drop(columns=duplicated_features)
    print(f'Removed {len(duplicated_features)} duplicated features.')

    # Step 4: Remove highly correlated features
    corr_matrix = features_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_features_to_remove = [
        column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)
    ]
    features_df = features_df.drop(columns=correlated_features_to_remove)
    print(f'Removed {len(correlated_features_to_remove)} highly correlated features.')

    return features_df

def main(input_file: str, output_file: str, quasi_constant_threshold: float = 0.99, correlation_threshold: float = 0.9):
    """
    Main function to load data, process it, and save the results.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
        quasi_constant_threshold (float): Threshold for quasi-constant features.
        correlation_threshold (float): Threshold for correlated features.
    """
    # Load the dataset
    try:
        features_df = pd.read_csv(input_file)
        features_df = features_df[features_df['audio_id'].apply(lambda x: not x.startswith('LJ'))]
        print(f"Loaded data with shape: {features_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Remove redundant features
    cleaned_features_df = remove_redundant_features(features_df, quasi_constant_threshold, correlation_threshold)

    # Save the cleaned dataset
    try:
        cleaned_features_df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        sys.exit(1)

if __name__ == '__main__':
    config = load_config()
    features_dir = config['data_paths']['features']
    features_csv = os.path.join(features_dir, 'features.csv')
    output_file = os.path.join(features_dir, 'cleaned_features.csv')
    main(features_csv, output_file)
