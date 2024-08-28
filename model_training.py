from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from config import *


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, set_name=""):
    y_pred = model.predict(X)
    print(f"\nClassification Report for {set_name}:")
    print(classification_report(y, y_pred))
    print(f"\nConfusion Matrix for {set_name}:")
    print(confusion_matrix(y, y_pred))


def save_model(model, filename="anomaly_detection_model.joblib"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


def load_model(filename="anomaly_detection_model.joblib"):
    return joblib.load(filename)


if __name__ == "__main__":
    from data_loader import load_data, split_data, prepare_features
    from feature_engineering import engineer_features

    # Load and prepare data
    df = load_data()
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    # Engineer features
    X_train_eng, X_val_eng, X_test_eng = engineer_features(X_train, X_val, X_test)

    # Train model
    model = train_model(X_train_eng, y_train)

    # Evaluate model
    evaluate_model(model, X_train_eng, y_train, "Training Set")
    evaluate_model(model, X_val_eng, y_val, "Validation Set")
    evaluate_model(model, X_test_eng, y_test, "Test Set")

    # Save model
    save_model(model)