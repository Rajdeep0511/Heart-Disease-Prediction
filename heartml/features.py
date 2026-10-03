import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schema import FEATURES

ENGINEERED = ["hr_reserve_pct", "age_x_oldpeak", "high_bp", "high_chol", "ischemia_flag"]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds a few clinically motivated derived columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES).astype(float) if not isinstance(X, pd.DataFrame) else X[FEATURES].astype(float)
        out = X.copy()
        # Share of the age-predicted maximum heart rate (220 - age) that was reached.
        out["hr_reserve_pct"] = X["thalach"] / (220.0 - X["age"]).clip(lower=1.0)
        out["age_x_oldpeak"] = X["age"] * X["oldpeak"] / 100.0
        out["high_bp"] = np.where(X["trestbps"].isna(), np.nan, (X["trestbps"] >= 140).astype(float))
        out["high_chol"] = np.where(X["chol"].isna(), np.nan, (X["chol"] >= 240).astype(float))
        out["ischemia_flag"] = ((X["exang"] == 1) | (X["oldpeak"] >= 1.0)).astype(float)
        return out
