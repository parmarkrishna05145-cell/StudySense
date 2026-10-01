"""
StudySense - sanity tests
=========================
Run:  python test_model.py
Plain asserts, no test framework needed. Every check prints PASS.
"""

import json
import os

import joblib
import pandas as pd

from app import predict
from train_model import DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGET


def run_tests() -> None:
    # 1. dataset checks
    df = pd.read_csv(DATA_PATH)
    assert set(FEATURES + [TARGET]).issubset(df.columns), "dataset columns missing"
    assert len(df) >= 200, "dataset too small"
    assert df.isna().sum().sum() == 0, "dataset contains NaN values"
    print("test 1: dataset has 220 clean rows with all 6 columns ......... PASS")

    # 2. model + metrics files exist
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as fh:
        metrics = json.load(fh)
    assert 0 < metrics["r2"] < 1, "unexpected R2"
    print(
        f"test 2: model loads, metrics sane (R2={metrics['r2']:.3f}) ......... PASS"
    )

    # 3. predictions stay in a valid 0-100 range
    base = dict(zip(FEATURES, [5.0, 7.0, 85.0, 70.0, 2.0]))
    p = predict(model, base)
    assert 0.0 <= p <= 100.0, f"prediction out of range: {p}"
    print(f"test 3: prediction in valid range ({p:.1f}/100) ................. PASS")

    # 4. more study hours -> higher predicted score
    low = predict(model, dict(base, hours_studied=2.0))
    high = predict(model, dict(base, hours_studied=8.0))
    assert high > low, "model does not reward more study hours"
    print(f"test 4: 8h study ({high:.1f}) beats 2h study ({low:.1f}) ........ PASS")

    # 5. better attendance -> higher predicted score
    att_low = predict(model, dict(base, attendance_pct=60.0))
    att_high = predict(model, dict(base, attendance_pct=96.0))
    assert att_high > att_low, "model does not reward attendance"
    print(
        f"test 5: 96% attendance ({att_high:.1f}) beats 60% ({att_low:.1f}) ... PASS"
    )

    print("\nALL 5 TESTS PASSED")


if __name__ == "__main__":
    run_tests()
