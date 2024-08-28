import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from config import *

def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def add_aggregated_features(X):
    # Add some basic aggregated features
    X_new = np.hstack([
        X,
        np.mean(X, axis=1).reshape(-1, 1),
        np.std(X, axis=1).reshape(-1, 1),
        np.median(X, axis=1).reshape(-1, 1),
        np.max(X, axis=1).reshape(-1, 1),
        np.min(X, axis=1).reshape(-1, 1)
    ])
    return X_new

def create_graph_features(df, df_edgelist):
    # Create a dictionary to store the number of incoming and outgoing edges for each transaction
    edge_counts = {txid: {'in': 0, 'out': 0} for txid in df['transid']}
    
    # Count incoming and outgoing edges
    for row in df_edgelist.iter_rows(named=True):
        txId1, txId2 = row['txId1'], row['txId2']
        if txId1 in edge_counts:
            edge_counts[txId1]['out'] += 1
        if txId2 in edge_counts:
            edge_counts[txId2]['in'] += 1
    
    # Add new columns to the dataframe
    df = df.with_columns([
        pl.Series('in_degree', [edge_counts[txid]['in'] for txid in df['transid']]),
        pl.Series('out_degree', [edge_counts[txid]['out'] for txid in df['transid']])
    ])
    
    return df

def engineer_features(X_train, X_val, X_test, df_train, df_val, df_test, df_edgelist):
    # Add graph features
    df_train = create_graph_features(df_train, df_edgelist)
    df_val = create_graph_features(df_val, df_edgelist)
    df_test = create_graph_features(df_test, df_edgelist)
    
    # Add the new features to X
    X_train = np.hstack([X_train, df_train.select(['in_degree', 'out_degree']).to_numpy()])
    X_val = np.hstack([X_val, df_val.select(['in_degree', 'out_degree']).to_numpy()])
    X_test = np.hstack([X_test, df_test.select(['in_degree', 'out_degree']).to_numpy()])

    # Add aggregated features
    X_train_new = add_aggregated_features(X_train)
    X_val_new = add_aggregated_features(X_val)
    X_test_new = add_aggregated_features(X_test)

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train_new, X_val_new, X_test_new)

    return X_train_scaled, X_val_scaled, X_test_scaled

if __name__ == "__main__":
    from data_loader import load_data, split_data, prepare_features

    df, df_edgelist = load_data()
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    X_train_eng, X_val_eng, X_test_eng = engineer_features(X_train, X_val, X_test, train_df, val_df, test_df, df_edgelist)

    print(f"Original feature shape: {X_train.shape}")
    print(f"Engineered feature shape: {X_train_eng.shape}")