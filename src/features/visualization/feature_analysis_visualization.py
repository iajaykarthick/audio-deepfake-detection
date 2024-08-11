import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, KernelPCA
from mpl_toolkits.mplot3d import Axes3D


def plot_low_level_feature_dist(df, feature_name):
    """
    Plots boxplots and violin plots of a specified feature grouped by category.
    """
    plot_data = []
    categories = []

    for idx, row in df.iterrows():
        feature_array = row[feature_name]
        real_or_fake = row['real_or_fake']
        plot_data.extend(feature_array)
        categories.extend([real_or_fake] * len(feature_array))

    plot_df = pd.DataFrame({
        'value': plot_data,
        'category': categories
    })

    plt.figure(figsize=(12, 6))
    
    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='category', y='value', data=plot_df)
    plt.title(f'Boxplot of {feature_name} by Category')
    plt.xlabel('Category')
    plt.ylabel('Value')

    # Violin Plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x='category', y='value', data=plot_df)
    plt.title(f'Violin Plot of {feature_name} by Category')
    plt.xlabel('Category')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()


def plot_high_level_feature_dist(df, feature_list, target_column='real_or_fake'):
    """
    Plots boxplots and violin plots for each feature in the feature list grouped by the target column.
    
    This function creates a side-by-side visualization of boxplots and violin plots for each specified feature in the 
    dataframe, allowing for a detailed comparison of distributions across different categories defined by the target column.
    """
    for feature in feature_list:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x=target_column, y=feature, data=df)
        plt.title(f'Boxplot of {feature} by Category')
        plt.xlabel('Category')
        plt.ylabel(feature)

        plt.subplot(1, 2, 2)
        sns.violinplot(x=target_column, y=feature, data=df)
        plt.title(f'Violin Plot of {feature} by Category')
        plt.xlabel('Category')
        plt.ylabel(feature)

        plt.tight_layout()
        plt.show()
        
        
def perform_pca_and_plot(df, selected_features, target_column='real_or_fake'):
    """
    Performs PCA on selected features and plots the first two principal components.
    """
    # Standardize the features
    features = df[selected_features]
    
    # Handle missing values by imputation
    imputer = SimpleImputer(strategy='mean')
    imputed_features = imputer.fit_transform(features)
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df[target_column] = df[target_column].values

    # Plot the PCA results
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue=target_column, data=pca_df, palette='tab20')
    plt.title('PCA of High-Level Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title=target_column)
    plt.show()
    
def perform_pca_and_plot_3d(df, selected_features, target_column='real_or_fake'):
    """
    Performs PCA on selected features and plots the first three principal components in 3D.
    """
    # Handle missing values by imputation
    imputer = SimpleImputer(strategy='mean')
    imputed_features = imputer.fit_transform(df[selected_features])
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    # Apply PCA
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    pca_df[target_column] = df[target_column].values

    # Plot the PCA results in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    targets = pca_df[target_column].unique()
    colors = sns.color_palette('Set2', len(targets))
    
    for target, color in zip(targets, colors):
        indices_to_keep = pca_df[target_column] == target
        ax.scatter(
            pca_df.loc[indices_to_keep, 'PC1'],
            pca_df.loc[indices_to_keep, 'PC2'],
            pca_df.loc[indices_to_keep, 'PC3'],
            label=target,
            c=[color],
            s=50
        )

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA of High-Level Features')
    ax.legend(title=target_column)
    plt.show()

    
def perform_kernel_pca_and_plot(df, selected_features, target_column='real_or_fake', kernel='rbf'):
    """
    Performs Kernel PCA using a specified kernel and plots the first two principal components.
    """
    # Extract the features
    features = df[selected_features]
    
    # Handle missing values by imputation
    imputer = SimpleImputer(strategy='mean')
    imputed_features = imputer.fit_transform(features)
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    # Apply Kernel PCA
    kpca = KernelPCA(n_components=2, kernel=kernel)
    principal_components = kpca.fit_transform(scaled_features)
    
    # Create a DataFrame with the principal components
    kpca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    kpca_df[target_column] = df[target_column].values

    # Plot the Kernel PCA results
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue=target_column, data=kpca_df, palette='Set2')
    plt.title('Kernel PCA of High-Level Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title=target_column)
    plt.show()