"""Serving side: load the trained bundle once and score a patient."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .schema import FEATURES

ARTIFACT_DIR = Path(os.environ.get("HEART_ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "artifacts"))


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_bundle():
    path = ARTIFACT_DIR / "model.joblib"
    if not path.exists():
        raise ModelNotTrained("No trained model found. Run:  python -m heartml.train")
    return joblib.load(path)


def load_metrics() -> dict | None:
    path = ARTIFACT_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def predict_risk(values: dict) -> dict:
    bundle = load_bundle()
    row = pd.DataFrame([{f: values.get(f, np.nan) for f in FEATURES}], dtype=float)
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {"probability": probability, "model_name": bundle["model_name"]}
