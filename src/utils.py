# =============================================================================
# src/utils.py
# Utility functions for the MLOps Wine Quality Prediction project.
# Includes helpers for loading data, evaluating models, and saving artifacts.
# =============================================================================

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def create_directories():
    """
    Ensure that required output directories exist.
    Creates 'models/' and 'reports/' at the project root if missing.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    print("[INFO] Directories ensured: models/, reports/")


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the Wine Quality CSV dataset.

    The UCI Wine Quality dataset uses semicolons (;) as separators.
    Falls back to comma if semicolon parsing yields a single column.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(filepath, sep=";")
        # If only one column was parsed, the separator might be a comma
        if df.shape[1] == 1:
            print("[WARN] Semicolon sep gave 1 column — retrying with comma.")
            df = pd.read_csv(filepath, sep=",")
        print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at '{filepath}'.\n"
            "Please download winequality-red.csv from:\n"
            "  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/\n"
            "and place it in the 'data/' folder."
        )


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained regression model on test data.

    Args:
        model: A fitted scikit-learn estimator.
        X_test: Test feature matrix.
        y_test: True target values.

    Returns:
        dict: Dictionary containing 'r2', 'mae', and 'rmse'.
    """
    predictions = model.predict(X_test)
    r2   = r2_score(y_test, predictions)
    mae  = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    return {
        "r2":   round(r2,   4),
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
    }


def save_metrics(metrics: dict, filepath: str = "reports/metrics.json"):
    """
    Save model metrics dictionary to a JSON file.

    Args:
        metrics (dict): Metrics to save (model name, r2, mae, rmse).
        filepath (str): Destination path for the JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved to '{filepath}'")
