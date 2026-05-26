# =============================================================================
# src/train.py
# Main training script for the Wine Quality Prediction MLOps project.
#
# Steps performed:
import sys, io
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
#   1. Load dataset
#   2. Split features and target
#   3. Train-test split
#   4. Train multiple regression models
#   5. Evaluate and select best model (highest R2)
#   6. Save best model as models/wine_model.pkl
#   7. Save metrics to reports/metrics.json
#   8. Generate comparison and prediction plots
#   9. Log all runs to MLflow
# =============================================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for headless/CI)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ── Make sure src/ imports work when running from project root ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import create_directories, load_dataset, evaluate_model, save_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH    = os.path.join("data", "winequality-red.csv")
MODEL_PATH   = os.path.join("models", "wine_model.pkl")
METRICS_PATH = os.path.join("reports", "metrics.json")
COMPARISON_PLOT_PATH    = os.path.join("reports", "model_comparison.png")
PRED_PLOT_PATH          = os.path.join("reports", "actual_vs_predicted.png")

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# List of (name, model) tuples to train and compare
MODELS = [
    ("LinearRegression",       LinearRegression()),
    ("DecisionTreeRegressor",  DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ("RandomForestRegressor",  RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    ("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)),
]

# Feature columns (all except target)
FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]
TARGET_COL = "quality"


def main():
    print("=" * 60)
    print("  Wine Quality Prediction - Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Ensure output directories exist
    # ------------------------------------------------------------------
    create_directories()

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    df = load_dataset(DATA_PATH)

    # ------------------------------------------------------------------
    # 3. Split features and target
    # ------------------------------------------------------------------
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    print(f"\n[INFO] Features shape : {X.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts().sort_index()}")

    # ------------------------------------------------------------------
    # 4. Train-test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[INFO] Train samples : {X_train.shape[0]}")
    print(f"[INFO] Test  samples : {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # 5. Train models, evaluate, and log to MLflow
    # ------------------------------------------------------------------
    results = {}   # {model_name: {r2, mae, rmse, model_object}}

    mlflow.set_experiment("wine-quality-prediction")
    print("\n[INFO] Starting MLflow experiment: wine-quality-prediction")

    for model_name, model in MODELS:
        print(f"\n" + "-"*50)
        print(f"[TRAIN] {model_name}")

        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            print(f"  R2   : {metrics['r2']}")
            print(f"  MAE  : {metrics['mae']}")
            print(f"  RMSE : {metrics['rmse']}")

            # Log to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("r2",   metrics["r2"])
            mlflow.log_metric("mae",  metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Store result
        results[model_name] = {**metrics, "model": model}

    # ------------------------------------------------------------------
    # 6. Select best model (highest R2)
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_metrics = results[best_name]
    best_model   = best_metrics["model"]

    print("\n" + "="*60)
    print(f"  Best Model : {best_name}")
    print(f"  R2         : {best_metrics['r2']}")
    print(f"  MAE        : {best_metrics['mae']}")
    print(f"  RMSE       : {best_metrics['rmse']}")
    print("="*60)

    # ------------------------------------------------------------------
    # 7. Save best model
    # ------------------------------------------------------------------
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[INFO] Best model saved to '{MODEL_PATH}'")

    # ------------------------------------------------------------------
    # 8. Save metrics JSON
    # ------------------------------------------------------------------
    final_metrics = {
        "best_model": best_name,
        "r2":   best_metrics["r2"],
        "mae":  best_metrics["mae"],
        "rmse": best_metrics["rmse"],
        "all_models": {
            k: {"r2": v["r2"], "mae": v["mae"], "rmse": v["rmse"]}
            for k, v in results.items()
        },
    }
    save_metrics(final_metrics, METRICS_PATH)

    # ------------------------------------------------------------------
    # 9. Plot 1 — Model Comparison (R2 bar chart)
    # ------------------------------------------------------------------
    model_names = list(results.keys())
    r2_scores   = [results[m]["r2"] for m in model_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars    = ax.bar(model_names, r2_scores, color=colors, edgecolor="white", width=0.5)

    # Highlight best model bar
    best_idx = model_names.index(best_name)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

    for bar, score in zip(bars, r2_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Model Comparison - R2 Score", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("R2 Score")
    ax.set_ylim(0, max(r2_scores) + 0.1)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"[INFO] Model comparison plot saved to '{COMPARISON_PLOT_PATH}'")

    # ------------------------------------------------------------------
    # 10. Plot 2 — Actual vs Predicted (best model)
    # ------------------------------------------------------------------
    y_pred = best_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.55, edgecolors="white",
               linewidths=0.3, color="#4C72B0", s=50, label="Predictions")

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min()) - 0.2
    max_val = max(y_test.max(), y_pred.max()) + 0.2
    ax.plot([min_val, max_val], [min_val, max_val],
            "r--", linewidth=1.5, label="Perfect Prediction")

    ax.set_xlabel("Actual Quality", fontsize=12)
    ax.set_ylabel("Predicted Quality", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {best_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PRED_PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"[INFO] Actual vs Predicted plot saved to '{PRED_PLOT_PATH}'")

    print("\n[DONE] Training complete. Run 'mlflow ui' to explore experiment results.")


if __name__ == "__main__":
    main()
