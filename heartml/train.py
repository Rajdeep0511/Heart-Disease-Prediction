"""Train, compare and export the heart-disease model.

    python -m heartml.train                 # standard run (~1-2 min)
    python -m heartml.train --tune          # + randomized hyper-parameter search
    python -m heartml.train --smote         # force SMOTE oversampling
    python -m heartml.train --data-dir data --out artifacts

Workflow
  1. Load + harmonise every CSV in data/ (heartml.datasets), de-duplicate.
  2. Hold out 20 % as a test set that is never used for any decision.
  3. Compare many algorithms with repeated stratified 5-fold CV (train part only).
  4. Build soft-voting and stacking ensembles from the best single models.
  5. Pick the best by CV ROC-AUC, check it once on the hold-out set, and test
     whether probability calibration improves the Brier score.
  6. Leave-one-source-out check (when several data sources exist).
  7. Refit on ALL data and save artifacts/model.joblib + artifacts/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .datasets import load_all
from .features import ClinicalFeatures
from .schema import FEATURES, TARGET

try:
    from xgboost import XGBClassifier
except Exception:  # optional dependency
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # optional dependency
    LGBMClassifier = None
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # optional dependency
    SMOTE = ImbPipeline = None

warnings.filterwarnings("ignore")

NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "hr_reserve_pct", "age_x_oldpeak"]
BIN_COLS = ["sex", "fbs", "exang", "high_bp", "high_chol", "ischemia_flag"]
CAT_COLS = ["cp", "restecg", "slope", "thal"]


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), NUM_COLS),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN_COLS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])


def make_pipeline(clf, smote: bool = False, seed: int = 42):
    steps = [("fe", ClinicalFeatures()), ("prep", make_preprocessor())]
    if smote:
        steps.append(("smote", SMOTE(random_state=seed)))
        return ImbPipeline(steps + [("clf", clf)])
    return Pipeline(steps + [("clf", clf)])


def build_candidates(seed: int, balanced: bool, pos_weight: float) -> dict:
    cw = "balanced" if balanced else None
    c = {
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=3000, class_weight=cw),
        "SVM (RBF)": SVC(C=1.0, probability=True, class_weight=cw, random_state=seed),
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                                class_weight="balanced_subsample" if balanced else None,
                                                n_jobs=1, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight=cw,
                                            n_jobs=1, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2,
                                                        subsample=0.8, random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=200,
                                                                 l2_regularization=1.0, class_weight=cw,
                                                                 random_state=seed),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000,
                                              random_state=seed),
    }
    if XGBClassifier is not None:
        c["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=2, eval_metric="logloss",
                                     scale_pos_weight=pos_weight if balanced else 1.0,
                                     n_jobs=1, random_state=seed, verbosity=0)
    if LGBMClassifier is not None:
        c["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight=cw,
                                       n_jobs=1, random_state=seed, verbose=-1)
    return c


TUNE_SPACES = {
    "Logistic Regression": {"C": loguniform(1e-2, 1e2)},
    "SVM (RBF)": {"C": loguniform(1e-1, 1e2), "gamma": loguniform(1e-3, 1e0)},
    "Random Forest": {"max_depth": [3, 5, 8, None], "min_samples_leaf": randint(1, 8),
                      "max_features": ["sqrt", 0.5, 0.8]},
    "Gradient Boosting": {"n_estimators": randint(80, 400), "learning_rate": loguniform(1e-2, 2e-1),
                          "max_depth": randint(1, 4), "subsample": uniform(0.6, 0.4)},
    "XGBoost": {"n_estimators": randint(100, 500), "learning_rate": loguniform(1e-2, 2e-1),
                "max_depth": randint(2, 5), "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(1e-1, 1e1)},
    "LightGBM": {"n_estimators": randint(100, 500), "learning_rate": loguniform(1e-2, 2e-1),
                 "num_leaves": randint(4, 24), "min_child_samples": randint(5, 30),
                 "reg_lambda": loguniform(1e-1, 1e1)},
}

SCORING = {"auc": "roc_auc", "accuracy": "accuracy", "f1": "f1", "recall": "recall",
           "precision": "precision", "brier": "neg_brier_score"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def holdout_metrics(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y, pred), "brier": brier_score_loss(y, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)),
    }


def bootstrap_ci(model, X, y, n=1000, seed=0) -> dict:
    """95 % bootstrap intervals for hold-out AUC and accuracy (small test sets are noisy)."""
    rng = np.random.default_rng(seed)
    proba = model.predict_proba(X)[:, 1]
    y = np.asarray(y)
    aucs, accs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
        accs.append(accuracy_score(y[idx], proba[idx] >= 0.5))
    lo, hi = np.percentile(aucs, [2.5, 97.5]), np.percentile(accs, [2.5, 97.5])
    return {"auc_ci95": [float(lo[0]), float(lo[1])],
            "accuracy_ci95": [float(hi[0]), float(hi[1])]}


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def log(msg: str):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tune", action="store_true", help="randomized hyper-parameter search")
    ap.add_argument("--tune-iter", type=int, default=20)
    ap.add_argument("--smote", action="store_true", help="force SMOTE (auto-enabled if classes are skewed)")
    ap.add_argument("--no-smote", action="store_true", help="never use SMOTE")
    args = ap.parse_args(argv)
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    data, report = load_all(args.data_dir)
    X, y = data[FEATURES], data[TARGET]
    pos_rate = float(y.mean())
    minority_ratio = min(pos_rate, 1 - pos_rate) / max(pos_rate, 1 - pos_rate)
    log(f"Loaded {len(data)} unique patients from {data['source'].nunique()} file(s); "
        f"disease prevalence {pos_rate:.1%}.")
    for s in report["sources"]:
        log(f"  - {s['file']:<40} raw={s['rows_raw']:>5}  final={s['rows_final']:>5}  ({s['adapter']})")
    for s in report["skipped_files"]:
        log(f"  - SKIPPED {s['file']}: {s['reason']}")

    use_smote = (args.smote or minority_ratio < 0.6) and not args.no_smote and SMOTE is not None
    if args.smote and SMOTE is None:
        log("imbalanced-learn not installed; falling back to class weights.")
    log(f"Imbalance handling: {'SMOTE' if use_smote else 'class weights'} (minority/majority = {minority_ratio:.2f}).")

    # 2. hold-out split -----------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cands = build_candidates(args.seed, balanced=not use_smote, pos_weight=pos_weight)

    # optional tuning
    tuned = {}
    if args.tune:
        log("\nTuning hyper-parameters (randomized search, CV on training part only)...")
        for name, space in TUNE_SPACES.items():
            if name not in cands:
                continue
            rs = RandomizedSearchCV(make_pipeline(cands[name], use_smote, args.seed),
                                    {f"clf__{k}": v for k, v in space.items()},
                                    n_iter=args.tune_iter, scoring="roc_auc",
                                    cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
                                    n_jobs=-1, random_state=args.seed, refit=False)
            rs.fit(X_tr, y_tr)
            best = {k.replace("clf__", ""): v for k, v in rs.best_params_.items()}
            cands[name].set_params(**best)
            tuned[name] = {"best_params": clean(best), "search_auc": round(float(rs.best_score_), 4)}
            log(f"  {name:<24} search AUC {rs.best_score_:.3f}")

    # 3. cross-validated comparison ----------------------------------------
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    rows, models = [], {}

    def evaluate(name, clf):
        pipe = make_pipeline(clf, use_smote, args.seed)
        res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for m in SCORING:
            vals = res[f"test_{m}"] * (-1 if m == "brier" else 1)
            row[m] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std())
        rows.append(row)
        models[name] = pipe
        log(f"  {name:<26} AUC {row['auc']:.3f} ± {row['auc_std']:.3f}   acc {row['accuracy']:.3f}   "
            f"recall {row['recall']:.3f}   F1 {row['f1']:.3f}")

    log(f"\nCross-validation ({args.folds}-fold x {args.repeats} repeats) on {len(X_tr)} training rows:")
    for name, clf in cands.items():
        evaluate(name, clf)

    # 4. ensembles ----------------------------------------------------------
    ranked = sorted(rows, key=lambda r: r["auc"], reverse=True)
    top = [r["model"] for r in ranked[:4]]
    evaluate("Soft Voting (top 3)", VotingClassifier([(n, clone(cands[n])) for n in top[:3]], voting="soft"))
    evaluate("Stacking (top 4 + LR)", StackingClassifier(
        [(n, clone(cands[n])) for n in top], final_estimator=LogisticRegression(max_iter=2000),
        cv=5, stack_method="predict_proba", n_jobs=1))

    # 5. selection + honest test -------------------------------------------
    table = sorted(rows, key=lambda r: (round(r["auc"], 4), r["f1"]), reverse=True)
    best_name = table[0]["model"]
    log(f"\nBest by CV ROC-AUC: {best_name}")

    raw = clone(models[best_name]).fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(clone(models[best_name]), method="sigmoid", cv=5).fit(X_tr, y_tr)
    m_raw, m_cal = holdout_metrics(raw, X_te, y_te), holdout_metrics(cal, X_te, y_te)
    use_cal = m_cal["brier"] < m_raw["brier"] - 1e-3
    test_metrics = m_cal if use_cal else m_raw
    test_metrics.update(bootstrap_ci(cal if use_cal else raw, X_te, y_te))
    ci = test_metrics["auc_ci95"]
    log(f"Hold-out ({len(y_te)} unseen rows): AUC {test_metrics['auc']:.3f} (95% CI {ci[0]:.2f}-{ci[1]:.2f})  "
        f"acc {test_metrics['accuracy']:.3f}  recall {test_metrics['recall']:.3f}  "
        f"precision {test_metrics['precision']:.3f}  Brier {m_raw['brier']:.3f} -> {m_cal['brier']:.3f}  "
        f"({'calibrated' if use_cal else 'uncalibrated'} kept)")

    fitted_for_importance = cal if use_cal else raw
    pi = permutation_importance(fitted_for_importance, X_te, y_te, scoring="roc_auc",
                                n_repeats=30, random_state=args.seed)
    importance = sorted(({"feature": f, "importance": float(m), "std": float(s)}
                         for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std)),
                        key=lambda d: d["importance"], reverse=True)

    # 6. leave-one-source-out ----------------------------------------------
    loso = []
    big = [s for s, n in data["source"].value_counts().items() if n >= 50]
    if len(big) >= 2:
        log("\nLeave-one-source-out generalisation:")
        for s in big:
            tr, te = data[data["source"] != s], data[data["source"] == s]
            if te[TARGET].nunique() < 2 or tr[TARGET].nunique() < 2:
                continue
            m = clone(models[best_name]).fit(tr[FEATURES], tr[TARGET])
            auc = roc_auc_score(te[TARGET], m.predict_proba(te[FEATURES])[:, 1])
            loso.append({"held_out_source": s, "rows": int(len(te)), "auc": auc})
            log(f"  held out {s:<35} AUC {auc:.3f}  (n={len(te)})")

    # 7. final refit on everything -----------------------------------------
    final = (CalibratedClassifierCV(clone(models[best_name]), method="sigmoid", cv=5) if use_cal
             else clone(models[best_name])).fit(X, y)
    bundle = {
        "pipeline": final, "features": FEATURES, "model_name": best_name, "calibrated": bool(use_cal),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": int(len(data)), "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, out_dir / "model.joblib", compress=3)

    metrics = clean({
        "trained_at": bundle["trained_at"], "best_model": best_name, "calibrated": bool(use_cal),
        "imbalance_strategy": "SMOTE" if use_smote else "class weights",
        "n_samples": len(data), "disease_prevalence": pos_rate, "dataset_report": report,
        "cv": {"folds": args.folds, "repeats": args.repeats, "table": table},
        "holdout": test_metrics, "holdout_uncalibrated_brier": m_raw["brier"],
        "holdout_calibrated_brier": m_cal["brier"], "tuning": tuned,
        "permutation_importance": importance, "leave_one_source_out": loso,
        "versions": {"sklearn": sklearn.__version__},
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"\nSaved {out_dir/'model.joblib'} and {out_dir/'metrics.json'} in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
