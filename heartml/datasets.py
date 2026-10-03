
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FEATURES, TARGET


class DatasetSkipped(Exception):
    """File is not a supported schema (not an error)."""


class DatasetRejected(Exception):
    """File looked supported but failed validation."""


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------
_KAGGLE_RENAME = {
    "trtbps": "trestbps", "thalachh": "thalach", "exng": "exang",
    "slp": "slope", "caa": "ca", "thall": "thal", "output": "target",
    "condition": "target",
}


def _adapt_kaggle_uci(df: pd.DataFrame) -> pd.DataFrame:
    """heart.csv-style files. Mappings derived from the data itself:
    cp 0/1/2/3 -> asymptomatic/atypical/non-anginal/typical, restecg 0/1/2 ->
    LVH/normal/ST-T, slope 0/1/2 -> down/flat/up, thal 1/2/3 ->
    fixed/normal/reversible, and target is inverted (1 = healthy)."""
    out = df.copy()
    out["cp"] = out["cp"].map({3: 0, 1: 1, 2: 2, 0: 3})
    out["restecg"] = out["restecg"].map({1: 0, 2: 1, 0: 2})
    out["slope"] = out["slope"].map({2: 0, 1: 1, 0: 2})
    out["thal"] = out["thal"].map({2: 0, 1: 1, 3: 2})      # 0 == missing -> NaN
    out["ca"] = out["ca"].where(out["ca"].between(0, 3))   # 4 == missing -> NaN
    out[TARGET] = 1 - out[TARGET]
    return out


def _adapt_cleveland_zero_based(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ca"] = out["ca"].where(out["ca"].between(0, 3))
    out["thal"] = out["thal"].where(out["thal"].between(0, 2))
    out[TARGET] = (out[TARGET] > 0).astype(int)
    return out


def _adapt_heart_failure(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age"] = df["Age"]
    out["sex"] = df["Sex"].astype(str).str.upper().map({"M": 1, "F": 0})
    out["cp"] = df["ChestPainType"].map({"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3})
    out["trestbps"] = df["RestingBP"]
    out["chol"] = df["Cholesterol"]            # 0 means "not recorded"
    out["fbs"] = df["FastingBS"]
    out["restecg"] = df["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2})
    out["thalach"] = df["MaxHR"]
    out["exang"] = df["ExerciseAngina"].astype(str).str.upper().map({"Y": 1, "N": 0})
    out["oldpeak"] = df["Oldpeak"]
    out["slope"] = df["ST_Slope"].map({"Up": 0, "Flat": 1, "Down": 2})
    out["ca"] = np.nan                          # not recorded in this dataset
    out["thal"] = np.nan
    out[TARGET] = df["HeartDisease"]
    return out


def _detect(df: pd.DataFrame):
    cols = set(df.columns)
    if "ap_hi" in cols and "cardio" in cols:
        raise DatasetSkipped("Cardiovascular Disease dataset has a different feature space "
                             "(no ECG / max heart rate / ST data); not used by this model.")
    if {"HeartDisease", "ChestPainType", "ST_Slope"} <= cols:
        return "heart_failure", _adapt_heart_failure(df)
    lower = df.rename(columns={c: c.strip().lower() for c in df.columns})
    lower = lower.rename(columns=_KAGGLE_RENAME)
    if set(FEATURES + [TARGET]) <= set(lower.columns):
        if "condition" in cols:
            return "cleveland_zero_based", _adapt_cleveland_zero_based(lower)
        return "kaggle_uci", _adapt_kaggle_uci(lower)
    raise DatasetSkipped(f"Unrecognised columns: {sorted(cols)[:8]}...")


# --------------------------------------------------------------------------
# Cleaning + validation
# --------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=[TARGET])
    # Physiologically impossible zeros are really "not recorded".
    for col in ("trestbps", "chol", "thalach"):
        out.loc[out[col] <= 0, col] = np.nan
    out["oldpeak"] = out["oldpeak"].clip(lower=0)
    out[TARGET] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


_EXPECTED_SIGN = {"exang": +1, "oldpeak": +1, "ca": +1, "age": +1, "thalach": -1, "cp_asymptomatic": +1}


def sanity_check(df: pd.DataFrame, name: str) -> dict:
    """Disease must go with older age, exercise angina, ST depression, more
    blocked vessels, asymptomatic chest pain and a LOWER max heart rate."""
    tmp = df.copy()
    tmp["cp_asymptomatic"] = (tmp["cp"] == 3).astype(float).where(tmp["cp"].notna())
    results = {}
    for col, sign in _EXPECTED_SIGN.items():
        if tmp[col].notna().sum() < 30 or tmp[col].nunique() < 2:
            continue
        corr = tmp[col].corr(tmp[TARGET])
        results[col] = round(float(corr), 3)
    passed = sum(1 for c, r in results.items() if r * _EXPECTED_SIGN[c] > 0)
    if results and passed < math.ceil(0.7 * len(results)):
        raise DatasetRejected(
            f"{name}: label/encoding sanity check failed ({passed}/{len(results)} expected "
            f"correlation signs hold: {results}). The file probably uses a different encoding."
        )
    return results


def load_one(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    raw = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]
    kind, adapted = _detect(raw)
    df = _clean(adapted)
    checks = sanity_check(df, path.name)
    df["source"] = path.stem
    info = {"file": path.name, "adapter": kind, "rows_raw": int(len(raw)), "label_checks": checks}
    return df, info


# --------------------------------------------------------------------------
# Combine + de-duplicate
# --------------------------------------------------------------------------
# Several Kaggle datasets are copies of the same ~300 Cleveland patients (one
# is even repeated ~3x). Left in, duplicates leak between train and test and
# inflate accuracy, so they are removed.
_CROSS_KEY = ["age", "sex", "cp", "trestbps", "thalach", "exang", "oldpeak"]


def combine(frames: list[tuple[pd.DataFrame, dict]]) -> tuple[pd.DataFrame, dict]:
    # Most complete sources first, so duplicates keep the richest copy.
    frames = sorted(frames, key=lambda f: f[0][FEATURES].isna().mean().mean())
    kept, seen, report = [], pd.DataFrame(columns=_CROSS_KEY), []
    for df, info in frames:
        n0 = len(df)
        df = df.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
        n1 = len(df)
        if len(seen):
            merged = df.merge(seen.drop_duplicates(), on=_CROSS_KEY, how="left", indicator=True)
            df = df[(merged["_merge"] == "left_only").to_numpy()].reset_index(drop=True)
        n2 = len(df)
        seen = pd.concat([seen, df[_CROSS_KEY]], ignore_index=True)
        kept.append(df)
        report.append({**info, "rows_after_exact_dedupe": n1, "rows_final": n2,
                       "dropped_exact_duplicates": n0 - n1, "dropped_overlap_with_other_files": n1 - n2})
    data = pd.concat(kept, ignore_index=True)
    return data, {"sources": report, "total_rows": int(len(data))}


def load_all(data_dir: str | Path = "data") -> tuple[pd.DataFrame, dict]:
    frames, skipped = [], []
    for path in sorted(Path(data_dir).glob("*.csv")):
        try:
            frames.append(load_one(path))
        except DatasetSkipped as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
        except DatasetRejected as exc:
            skipped.append({"file": path.name, "reason": f"REJECTED - {exc}"})
    if not frames:
        raise SystemExit(f"No usable CSV files found in {data_dir}/. See data/README.md.")
    data, report = combine(frames)
    report["skipped_files"] = skipped
    return data, report
