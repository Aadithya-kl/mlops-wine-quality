# =============================================================================
# src/predict.py
# Prediction helper for the Wine Quality Prediction project.
#
# Loads the saved best model and provides a clean predict() function
# that returns both a numeric score and a quality category label.
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd

# Path to the saved model (relative to project root)
MODEL_PATH = os.path.join("models", "wine_model.pkl")

# Ordered list of feature columns — must match training order
FEATURE_COLS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_model():
    """
    Load the trained model from disk.

    Returns:
        Fitted scikit-learn estimator.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Model not found at '{MODEL_PATH}'.\n"
            "Please run training first:\n"
            "    python src/train.py\n"
        )
    model = joblib.load(MODEL_PATH)
    return model


def get_quality_category(score: float) -> str:
    """
    Convert a numeric quality score to a human-readable category.

    Args:
        score (float): Predicted wine quality score.

    Returns:
        str: 'Low quality', 'Medium quality', or 'High quality'.
    """
    if score < 5:
        return "Low quality"
    elif score < 7:
        return "Medium quality"
    else:
        return "High quality"


def predict(feature_values: list) -> dict:
    """
    Predict wine quality given a list of feature values.

    Args:
        feature_values (list): Numeric values in the same order as FEATURE_COLS:
            [fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
             chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
             density, pH, sulphates, alcohol]

    Returns:
        dict: {
            'score'    : float — predicted quality score (rounded to 2 dp),
            'category' : str   — 'Low quality' / 'Medium quality' / 'High quality',
            'features' : dict  — feature name → value mapping
        }
    """
    if len(feature_values) != len(FEATURE_COLS):
        raise ValueError(
            f"Expected {len(FEATURE_COLS)} features, got {len(feature_values)}."
        )

    model = load_model()

    # Build a single-row DataFrame with proper column names (suppresses sklearn warning)
    X = pd.DataFrame([feature_values], columns=FEATURE_COLS)
    raw_score = model.predict(X)[0]
    score     = round(float(raw_score), 2)
    category  = get_quality_category(score)

    return {
        "score":    score,
        "category": category,
        "features": dict(zip(FEATURE_COLS, feature_values)),
    }


# ---------------------------------------------------------------------------
# Quick self-test — run directly to verify the module works
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example feature values (approximate average wine)
    sample = [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    result = predict(sample)
    print(f"\nPrediction Result")
    print(f"  Score    : {result['score']}")
    print(f"  Category : {result['category']}")
    print(f"\n  Feature values:")
    for feat, val in result["features"].items():
        print(f"    {feat:<25} : {val}")
