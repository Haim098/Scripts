import polars as pl
import numpy as np
from config import *

def load_data():
    """
    Load and preprocess the dataset.

    This function performs the following steps:
    1. Loads the classes and features datasets
    2. Loads the edgelist dataset
    3. Renames the feature columns
    4. Merges the features and classes datasets
    5. Maps the class labels to integer values

    Returns:
        tuple: A tuple containing two Polars DataFrames:
            - df: The main dataset with features and classes
            - df_edgelist: The edgelist dataset
    """
    # Load datasets
    df_classes = pl.read_csv(os.path.join(DATA_DIR, CLASSES_FILE), new_columns=['transid', 'class'])
    df_features = pl.read_csv(os.path.join(DATA_DIR, FEATURES_FILE), has_header=False)
    df_edgelist = pl.read_csv(os.path.join(DATA_DIR, EDGELIST_FILE))

    # Rename feature columns
    feature_names = {f'column_{i}': f'feature_{i - 2}' for i in range(3, df_features.shape[1] + 1)}
    feature_names['column_1'] = 'transid'
    feature_names['column_2'] = 'time_steps'
    df_features = df_features.rename(feature_names)

    # Merge features and classes
    df = df_features.join(df_classes, on='transid', how='left')

    # Map classes to integer values
    class_mapping = {'unknown': 2, '1': 1, '2': 0}
    df = df.with_columns(
        pl.col('class').replace(class_mapping).cast(pl.Int32).alias('class')
    )

    return df, df_edgelist

def split_data(df):
    """
    Split the data into training, validation, and test sets based on time steps.

    Args:
        df (pl.DataFrame): The main dataset containing features and classes

    Returns:
        tuple: A tuple containing three Polars DataFrames:
            - train_df: The training dataset
            - val_df: The validation dataset
            - test_df: The test dataset
    """
    train_df = df.filter(
        (pl.col('time_steps') >= pl.lit(TIME_STEP_RANGE['train'][0])) &
        (pl.col('time_steps') <= pl.lit(TIME_STEP_RANGE['train'][1])) &
        (pl.col('class') != pl.lit(2))
    )

    val_df = df.filter(
        (pl.col('time_steps') >= pl.lit(TIME_STEP_RANGE['valid'][0])) &
        (pl.col('time_steps') <= pl.lit(TIME_STEP_RANGE['valid'][1])) &
        (pl.col('class') != pl.lit(2))
    )

    test_df = df.filter(
        (pl.col('time_steps') >= pl.lit(TIME_STEP_RANGE['test'][0])) &
        (pl.col('time_steps') <= pl.lit(TIME_STEP_RANGE['test'][1])) &
        (pl.col('class') != pl.lit(2))
    )

    return train_df, val_df, test_df

def prepare_features(df):
    """
    Prepare features and labels for model training.

    This function extracts the feature columns and the class labels from the dataset.

    Args:
        df (pl.DataFrame): The dataset containing features and classes

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - X: The feature matrix
            - y: The label vector
    """
    # Select the feature columns
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    # Convert the selected features to numpy array
    X = df.select(feature_cols).to_numpy()
    
    # Convert the 'class' column to numpy array and flatten it
    y = df.select('class').to_numpy().flatten()
    
    return X, y

if __name__ == "__main__":
    # Load the data
    df, df_edgelist = load_data()
    
    # Split the data into train, validation, and test sets
    train_df, val_df, test_df = split_data(df)

    # Prepare features for each set
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    # Print the shapes of the resulting datasets
    print(f"Train set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Edgelist shape: {df_edgelist.shape}")