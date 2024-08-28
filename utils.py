import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging


def setup_logging():
    """
    Set up the logging configuration for the project.

    This function configures the logging module with the following settings:
    - Log level: INFO
    - Log format: timestamp - log level - message
    - Date format: YYYY-MM-DD HH:MM:SS
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def calculate_performance_metrics(y_true, y_pred_proba):
    """
    Calculate various performance metrics for the model.

    This function computes the following metrics:
    1. ROC AUC (Receiver Operating Characteristic Area Under the Curve)
    2. PR AUC (Precision-Recall Area Under the Curve)
    3. Best F1 Score and its corresponding threshold

    Args:
        y_true (np.array): True labels
        y_pred_proba (np.array): Predicted probabilities for the positive class

    Returns:
        dict: A dictionary containing the calculated metrics
    """
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Calculate F1 score at different thresholds
    f1_scores = []
    thresholds = np.arange(0.1, 1.0, 0.1)
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-10)
        recall = np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    return {
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Best F1 Score": best_f1,
        "Best Threshold": best_threshold
    }


def save_results(results, filename="results.csv"):
    """
    Save results to a CSV file.

    Args:
        results (list): A list of dictionaries containing the results
        filename (str, optional): The name of the file to save the results. Defaults to "results.csv".
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    logging.info(f"Results saved to {filename}")


if __name__ == "__main__":
    # Test the functions
    setup_logging()
    logging.info("Testing utils functions")

    # Generate some dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.random(1000)

    metrics = calculate_performance_metrics(y_true, y_pred_proba)
    logging.info("Performance metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")

    save_results([metrics], "test_results.csv")