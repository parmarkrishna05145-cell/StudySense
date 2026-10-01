"""
StudySense - chart generator
============================
Creates feature_importance.png (a horizontal bar chart of what drives
the predicted score). Requires matplotlib; run after training.
"""

import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_model import FEATURES

OUT_PATH = "feature_importance.png"
LABELS = {
    "hours_studied": "Study hours / day",
    "sleep_hours": "Sleep hours / day",
    "attendance_pct": "Attendance %",
    "previous_score": "Previous exam score",
    "extracurricular_hours": "Extracurricular hrs / day",
}


def make_chart() -> None:
    model = joblib.load("model/model.joblib")
    importances = model.named_steps["rf"].feature_importances_

    pairs = sorted(zip(FEATURES, importances), key=lambda t: t[1])
    labels = [LABELS[f] for f, _ in pairs]
    values = [imp for _, imp in pairs]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(labels, values, color="#4F6DF5")
    ax.set_xlabel("Importance (share of prediction)")
    ax.set_title("StudySense - What Drives the Predicted Exam Score?")
    ax.set_xlim(0, max(values) * 1.25)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.0%}",
                va="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Chart saved -> {OUT_PATH}")


if __name__ == "__main__":
    make_chart()
