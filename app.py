"""
StudySense - Exam Score Predictor (CLI)
=======================================
Predict a student's final exam score from their study habits.

Usage:
    python app.py --hours 6 --sleep 7 --attendance 85 --prev 72 --activities 2
    python app.py            # runs a demo on 3 built-in student profiles

Built as part of the IBM SkillsBuild x AICTE Internship 2026
(Masterclass 1 - IBM Bob project). Developed with IBM Bob as the
AI SDLC partner: Plan -> Code -> Debug -> Refactor.
"""

import argparse
import json
import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib

from train_model import FEATURES, MODEL_PATH, train

GRADE_BANDS = [
    (90, "\U0001F31F Excellent"),
    (75, "\u2705 Very Good"),
    (60, "\U0001F44D Good"),
    (40, "\u26A0\uFE0F Average"),
    (0, "\U0001F6A8 Needs Improvement"),
]

FLAG_TO_FEATURE = {
    "hours": "hours_studied",
    "sleep": "sleep_hours",
    "attendance": "attendance_pct",
    "prev": "previous_score",
    "activities": "extracurricular_hours",
}

DEMO_PROFILES = [
    ("Riya  - the consistent topper", 6.5, 7.5, 94, 81, 1.5),
    ("Arjun - last-minute crammer", 8.0, 5.0, 68, 55, 1.0),
    ("Sam   - balanced but distracted", 2.5, 8.0, 72, 62, 4.5),
]


def load_model():
    """Load the saved model, retraining automatically if missing."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found - training a fresh model first...\n")
        train(verbose=False)
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    path = os.path.join("model", "metrics.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def predict(model, profile: dict) -> float:
    """Return the predicted final score (0-100) for a student profile."""
    vector = [[profile[f] for f in FEATURES]]
    return float(model.predict(vector)[0])


def grade_band(score: float) -> str:
    for threshold, label in GRADE_BANDS:
        if score >= threshold:
            return label
    return ""


def show(title: str, profile: dict, score: float, mae: float) -> None:
    bar_len = 20
    filled = int(round(max(0.0, min(100.0, score)) / 100 * bar_len))
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    print("-" * 60)
    print(f"Student : {title}")
    for feat in FEATURES:
        print(f"  {feat:<22} {profile[feat]}")
    print(f"Predicted final score : {score:.1f} / 100  {grade_band(score)}")
    print(f"Score meter: [{bar}]")
    print(f"(model's typical error: about +/- {mae:.1f} marks)")


def run_demo(model, mae: float) -> None:
    print(f"Running StudySense demo on {len(DEMO_PROFILES)} student profiles...\n")
    for title, *vals in DEMO_PROFILES:
        profile = dict(zip(FEATURES, vals))
        show(title, profile, predict(model, profile), mae)
    print("-" * 60)
    print("Tip: predict your own score ->")
    print('  python app.py --hours 6 --sleep 7 --attendance 85 --prev 72 --activities 2')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StudySense - predict the final exam score from study habits"
    )
    parser.add_argument("--hours", type=float, help="study hours per day")
    parser.add_argument("--sleep", type=float, help="sleep hours per day")
    parser.add_argument("--attendance", type=float, help="attendance percentage")
    parser.add_argument("--prev", type=float, help="previous exam score")
    parser.add_argument("--activities", type=float, help="extracurricular hours per day")
    args = parser.parse_args()

    print("=" * 60)
    print(" StudySense | Exam Score Predictor | built with IBM Bob")
    print("=" * 60 + "\n")

    values = {FLAG_TO_FEATURE[f]: getattr(args, f) for f in FLAG_TO_FEATURE}
    if all(v is None for v in values.values()):
        model = load_model()
        run_demo(model, load_metrics().get("mae", 0.0))
    elif any(v is None for v in values.values()):
        parser.error(
            "provide ALL five values: --hours --sleep --attendance --prev --activities "
            "(or none to run the demo)"
        )
    else:
        model = load_model()
        show("Your profile", values, predict(model, values), load_metrics().get("mae", 0.0))


if __name__ == "__main__":
    main()
