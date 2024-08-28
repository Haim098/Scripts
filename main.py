import logging
from data_loader import load_data, split_data, prepare_features
from feature_engineering import engineer_features
from model_training import train_model, evaluate_model, save_model
from anomaly_detector import AnomalyDetector
from visualization import plot_tsne, plot_feature_importance, plot_anomaly_scores
from utils import setup_logging, calculate_performance_metrics, save_results
from config import ANOMALY_THRESHOLD

def main():
    """
    Main function to orchestrate the entire anomaly detection process.
    
    This function performs the following steps:
    1. Set up logging
    2. Load and prepare data
    3. Engineer features
    4. Train the model
    5. Evaluate the model
    6. Generate visualizations
    7. Perform anomaly detection
    8. Calculate and save performance metrics
    """
    setup_logging()
    logging.info("Starting the anomaly detection process")

    # Load and prepare data
    logging.info("Loading and preparing data")
    df, df_edgelist = load_data()
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    # Engineer features
    logging.info("Engineering features")
    X_train_eng, X_val_eng, X_test_eng = engineer_features(X_train, X_val, X_test, train_df, val_df, test_df, df_edgelist)

    # Train model
    logging.info("Training model")
    model = train_model(X_train_eng, y_train)

    # Evaluate model
    logging.info("Evaluating model")
    evaluate_model(model, X_train_eng, y_train, "Training Set")
    evaluate_model(model, X_val_eng, y_val, "Validation Set")
    evaluate_model(model, X_test_eng, y_test, "Test Set")

    # Save model
    save_model(model)

    # Visualizations
    logging.info("Generating visualizations")
    plot_tsne(X_test_eng[:, :model.n_features_in_], y_test, "t-SNE of Test Transactions")
    feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
    plot_feature_importance(model, feature_names)

    # Anomaly detection
    logging.info("Performing anomaly detection")
    detector = AnomalyDetector()
    is_anomaly, anomaly_scores = detector.detect(X_test_eng[:, :model.n_features_in_], df_edgelist)

    plot_anomaly_scores(anomaly_scores, ANOMALY_THRESHOLD)

    # Calculate and save performance metrics
    metrics = calculate_performance_metrics(y_test, anomaly_scores)
    save_results([metrics], "final_results.csv")

    logging.info("Anomaly detection process completed")

if __name__ == "__main__":
    main()