"""Unit tests for the dataset adapters (synthetic rows; no Kaggle files needed)."""
import unittest

import numpy as np
import pandas as pd

from heartml.datasets import DatasetRejected, DatasetSkipped, _clean, _detect, combine, sanity_check
from heartml.schema import FEATURES


def synthetic_canonical(n=200, seed=0):
    """Canonical frame where disease really goes with the clinical signs."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "age": 50 + 8 * y + rng.normal(0, 6, n), "sex": rng.integers(0, 2, n),
        "cp": np.where(y == 1, rng.choice([0, 1, 2, 3], n, p=[.1, .1, .1, .7]), rng.choice([0, 1, 2, 3], n)),
        "trestbps": rng.normal(130, 15, n), "chol": rng.normal(240, 40, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": 160 - 25 * y + rng.normal(0, 12, n),
        "exang": (rng.random(n) < (0.15 + 0.5 * y)).astype(int), "oldpeak": np.abs(0.3 + 1.5 * y + rng.normal(0, .6, n)),
        "slope": rng.integers(0, 3, n), "ca": np.where(y == 1, rng.integers(0, 4, n), rng.choice([0, 0, 0, 1], n)),
        "thal": rng.integers(0, 3, n), "target": y})
    # Real files store whole numbers (oldpeak: 1 decimal), so overlapping rows match exactly.
    for col in ("age", "trestbps", "chol", "thalach"):
        df[col] = df[col].round()
    df["oldpeak"] = df["oldpeak"].round(1)
    return df


def to_heart_failure(df):
    return pd.DataFrame({
        "Age": df.age.round(), "Sex": df.sex.map({1: "M", 0: "F"}),
        "ChestPainType": df.cp.map({0: "TA", 1: "ATA", 2: "NAP", 3: "ASY"}), "RestingBP": df.trestbps.round(),
        "Cholesterol": df.chol.round(), "FastingBS": df.fbs, "RestingECG": df.restecg.map({0: "Normal", 1: "ST", 2: "LVH"}),
        "MaxHR": df.thalach.round(), "ExerciseAngina": df.exang.map({1: "Y", 0: "N"}), "Oldpeak": df.oldpeak.round(1),
        "ST_Slope": df.slope.map({0: "Up", 1: "Flat", 2: "Down"}), "HeartDisease": df.target})


def to_kaggle(df):
    """Inverse of the heart.csv encoding (inverted label, renumbered categories)."""
    k = df.copy()
    k["cp"] = df.cp.map({0: 3, 1: 1, 2: 2, 3: 0})
    k["restecg"] = df.restecg.map({0: 1, 1: 2, 2: 0})
    k["slope"] = df.slope.map({0: 2, 1: 1, 2: 0})
    k["thal"] = df.thal.map({0: 2, 1: 1, 2: 3})
    k["target"] = 1 - df.target
    return k


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.canon = synthetic_canonical()

    def test_kaggle_roundtrip_restores_canonical(self):
        kind, out = _detect(to_kaggle(self.canon))
        self.assertEqual(kind, "kaggle_uci")
        pd.testing.assert_frame_equal(out[FEATURES + ["target"]].astype(float),
                                      self.canon[FEATURES + ["target"]].astype(float), check_dtype=False)

    def test_heart_failure_adapter(self):
        kind, out = _detect(to_heart_failure(self.canon))
        self.assertEqual(kind, "heart_failure")
        cleaned = _clean(out)
        self.assertTrue(cleaned[["ca", "thal"]].isna().all().all())
        self.assertEqual(sanity_check(cleaned, "hf").get("exang", 0) > 0, True)

    def test_cleveland_zero_based(self):
        df = self.canon.rename(columns={"target": "condition"})
        self.assertEqual(_detect(df)[0], "cleveland_zero_based")

    def test_inverted_label_is_rejected(self):
        bad = self.canon.copy()
        bad["target"] = 1 - bad["target"]
        with self.assertRaises(DatasetRejected):
            sanity_check(_clean(bad), "bad")

    def test_cardio_dataset_skipped(self):
        with self.assertRaises(DatasetSkipped):
            _detect(pd.DataFrame({"ap_hi": [120], "ap_lo": [80], "cardio": [0]}))

    def test_cross_source_dedupe(self):
        a = _clean(self.canon)
        a["source"] = "a"
        b = _clean(_detect(to_heart_failure(self.canon))[1])
        b["source"] = "b"
        data, rep = combine([(a, {"file": "a"}), (b, {"file": "b"})])
        self.assertLessEqual(len(data), len(a) + 5)       # copy of the same patients adds ~nothing
        self.assertEqual(data["source"].iloc[0], "a")      # most complete copy kept


if __name__ == "__main__":
    unittest.main()
