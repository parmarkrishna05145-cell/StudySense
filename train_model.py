"""
StudySense - Exam Score Predictor (Training)
============================================
Trains a StandardScaler + Random Forest regressor on
data/students_dataset.csv, prints MAE / RMSE / R2 and feature
importances, saves the pipeline to model/model.joblib and the
metrics to model/metrics.json.

Built as part of the IBM SkillsBuild x AICTE Internship 2026
(Masterclass 1 - IBM Bob project). Developed with IBM Bob as the
AI SDLC partner: Plan -> Code -> Debug -> Refactor.
"""

import json
import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "students_dataset.csv")
MODEL_PATH = os.path.join("model", "model.joblib")
METRICS_PATH = os.path.join("model", "metrics.json")

FEATURES = [
    "hours_studied",
    "sleep_hours",
    "attendance_pct",
    "previous_score",
    "extracurricular_hours",
]
TARGET = "final_score"


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (works across sklearn versions)."""
    try:
        from sklearn.metrics import root_mean_squared_error

        return float(root_mean_squared_error(y_true, y_pred))
    except ImportError:  # older sklearn
        from sklearn.metrics import mean_squared_error

        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_pipeline() -> Pipeline:
    """Scaled features -> Random Forest regressor."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(n_estimators=200, random_state=42),
            ),
        ]
    )


def train(verbose: bool = True):
    """Load data, train, evaluate, save model + metrics."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].to_numpy()
    y = df[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": rmse(y_test, preds),
        "r2": float(r2_score(y_test, preds)),
    }

    importances = model.named_steps["rf"].feature_importances_
    ranked = sorted(zip(FEATURES, importances), key=lambda t: -t[1])

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    if verbose:
        print("=" * 60)
        print(
            f"StudySense model trained | MAE {metrics['mae']:.2f} marks | "
            f"RMSE {metrics['rmse']:.2f} | R2 {metrics['r2']:.3f}"
        )
        print("=" * 60)
        print("What drives the predicted score (feature importance):")
        for name, imp in ranked:
            bar = "\u2588" * int(round(imp * 40))
            print(f"  {name:<22} {imp:>6.1%}  {bar}")
        print(f"\nModel saved  -> {MODEL_PATH}")
        print(f"Metrics saved -> {METRICS_PATH}")

    return model


if __name__ == "__main__":
    train()
