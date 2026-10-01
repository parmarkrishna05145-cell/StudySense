"""
StudySense - Exam Score Predictor
=================================
Generates the bundled synthetic student dataset (220 records) used for
training. Deterministic (seed 42) so anyone can reproduce it.

Built as part of the IBM SkillsBuild x AICTE (BharatCares / CSRBOX)
Machine Learning & Applied AI Internship 2026 - Masterclass 1 IBM Bob project.
Developed with IBM Bob as the AI SDLC partner (Plan -> Code -> Debug -> Refactor).
"""

import os

import numpy as np
import pandas as pd

N_STUDENTS = 220
DATA_PATH = os.path.join("data", "students_dataset.csv")


def generate() -> pd.DataFrame:
    """Create a plausible synthetic dataset of study habits vs exam scores."""
    rng = np.random.default_rng(42)

    hours = rng.uniform(0.0, 10.0, N_STUDENTS).round(1)        # study hrs/day
    sleep = rng.uniform(4.0, 9.0, N_STUDENTS).round(1)         # sleep hrs/day
    attendance = rng.uniform(55, 100, N_STUDENTS).round(0)     # %
    previous = rng.uniform(35, 95, N_STUDENTS).round(0)        # last exam score
    extracurricular = rng.uniform(0.0, 6.0, N_STUDENTS).round(1)  # hrs/day

    noise = rng.normal(0, 3.2, N_STUDENTS)
    score = (
        8
        + 0.30 * previous
        + 2.60 * hours
        + 0.18 * attendance
        + 1.20 * sleep
        - 0.80 * extracurricular
        + noise
    )
    score = np.clip(score, 18, 98).round(0)

    return pd.DataFrame(
        {
            "hours_studied": hours,
            "sleep_hours": sleep,
            "attendance_pct": attendance,
            "previous_score": previous,
            "extracurricular_hours": extracurricular,
            "final_score": score,
        }
    )


def main() -> None:
    df = generate()
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} student records -> {DATA_PATH}")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
