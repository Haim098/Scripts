import numpy as np
from model_training import load_model
from feature_engineering import engineer_features
from config import ANOMALY_THRESHOLD
import polars as pl


class AnomalyDetector:
    def __init__(self, model_path="anomaly_detection_model.joblib"):
        self.model = load_model(model_path)
        self.n_features = self.model.n_features_in_

    def detect(self, transactions, df_edgelist):
        # Ensure transactions is 2D
        if transactions.ndim == 1:
            transactions = transactions.reshape(1, -1)

        # Create dummy DataFrames for engineer_features
        df_dummy = pl.DataFrame({'transid': range(transactions.shape[0])})
        
        # Engineer features (assuming single sample or batch)
        X_engineered, _, _ = engineer_features(transactions, transactions, transactions, 
                                               df_dummy, df_dummy, df_dummy, df_edgelist)

        # Ensure we use only the features the model was trained on
        X_engineered = X_engineered[:, :self.n_features]

        # Get probabilities
        probabilities = self.model.predict_proba(X_engineered)

        # Anomaly if probability of class 1 (illicit) > threshold
        is_anomaly = probabilities[:, 1] > ANOMALY_THRESHOLD

        return is_anomaly, probabilities[:, 1]

def format_output(transaction, is_anomaly, anomaly_score):
    return {
        "transaction": transaction,
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": float(anomaly_score)
    }


if __name__ == "__main__":
    from data_loader import load_data, prepare_features

    # Load some test data
    df = load_data()
    X, _ = prepare_features(df.sample(5))  # Get 5 random transactions

    # Initialize and use the detector
    detector = AnomalyDetector()
    is_anomaly, anomaly_scores = detector.detect(X)

    # Format and print results
    for i, (transaction, anomaly, score) in enumerate(zip(X, is_anomaly, anomaly_scores)):
        result = format_output(transaction, anomaly, score)
        print(f"Transaction {i + 1}:")
        print(f"  Is Anomaly: {result['is_anomaly']}")
        print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
        print()