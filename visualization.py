import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from config import TSNE_PERPLEXITY, TSNE_N_COMPONENTS


def plot_tsne(X, y, title="t-SNE Visualization"):
    """
    Create and display a t-SNE plot of the data.

    Args:
        X (np.array): The feature matrix
        y (np.array): The label vector
        title (str, optional): The title of the plot. Defaults to "t-SNE Visualization".
    """
    # Create t-SNE model and fit it to the data
    tsne = TSNE(n_components=TSNE_N_COMPONENTS, perplexity=TSNE_PERPLEXITY, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Create and display a bar plot of feature importances.

    Args:
        model: The trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        top_n (int, optional): Number of top features to display. Defaults to 20.
    """
    # Get feature importances from the model
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Feature Importances")
    plt.bar(range(top_n), importances[indices][:top_n])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()


def plot_anomaly_scores(anomaly_scores, threshold):
    """
    Create and display a histogram of anomaly scores.

    Args:
        anomaly_scores (np.array): Array of anomaly scores
        threshold (float): Threshold for classifying anomalies
    """
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores, bins=50, edgecolor='black')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    from data_loader import load_data, prepare_features
    from feature_engineering import engineer_features
    from model_training import load_model
    from anomaly_detector import AnomalyDetector

    # Load and prepare data
    df, _ = load_data()
    X, y = prepare_features(df)
    X_eng, _, _ = engineer_features(X, X, X)

    # Plot t-SNE
    plot_tsne(X_eng, y, "t-SNE of Transactions")

    # Plot feature importance
    model = load_model()
    feature_names = [f"feature_{i}" for i in range(X_eng.shape[1])]
    plot_feature_importance(model, feature_names)

    # Plot anomaly scores
    detector = AnomalyDetector()
    _, anomaly_scores = detector.detect(X)
    plot_anomaly_scores(anomaly_scores, ANOMALY_THRESHOLD)