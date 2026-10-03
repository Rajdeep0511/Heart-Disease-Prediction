# Heart Risk Check

**Kept (patient side):** register, log in, my details, enter measurements and get a prediction, history of past checks.
**Removed:** doctor and admin roles, add/view doctor, dataset upload, feedback, notifications, doctor search, intro video, Django admin.
**Added:** model comparison page, probability with risk band, reference-range flags, edit/delete, per-user privacy, tests.

## Run it

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver                            # http://127.0.0.1:8000
```

A trained model is included (`artifacts/`), so the site works immediately.

## Retrain / add datasets

```bash
python -m heartml.download_data      # optional: fetch the Kaggle datasets into data/ (needs internet)
python -m heartml.train              # ~2 min on 300 rows; add --tune for hyper-parameter search
python manage.py test && python -m unittest heartml.tests
```

Or download CSVs by hand into `data/` (see `data/README.md`). Every file is auto-detected, converted to
one encoding, sanity-checked and de-duplicated. The site's **About the model** page shows the results.

## What the ML pipeline does

| Step | Detail |
|---|---|
| Harmonise | Adapters convert each dataset to one clinical encoding. A label/encoding sanity check rejects a file whose disease direction looks wrong. |
| De-duplicate | Exact duplicates and cross-file overlap are removed (several Kaggle files are the same Cleveland patients), so train and test never share patients. |
| Feature engineering | Heart-rate reserve %, age x ST depression, high-BP, high-cholesterol and ischemia flags, all inside the sklearn pipeline so serving matches training. |
| Algorithms | Logistic Regression, SVM, kNN, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, MLP, XGBoost, LightGBM, plus Soft Voting and Stacking ensembles. |
| Imbalance | Class weights; SMOTE is auto-enabled when the minority class is under 60% of the majority (`--smote` to force). |
| Evaluation | 20% untouched hold-out; repeated stratified 5-fold CV for selection; bootstrap 95% CIs; leave-one-source-out when several sources exist. |
| Calibration | Sigmoid calibration is kept only if it improves hold-out Brier score. |
| Explainability | Permutation importance (global); reference-range flags on each result. |

The original project used Gradient Boosting with `learning_rate=1.0, max_depth=1` and retrained it from the database on every request.
This version trains once, saves the model, and loads it at startup.



## Layout

```
heartsite/   Django settings + urls          templates/, static/css/   simple, offline UI (no CDN)
patients/    models, forms, views, tests     heartml/                  datasets, features, train, predictor
data/        training CSVs                   artifacts/                model.joblib, metrics.json
```

Production: set `DJANGO_DEBUG=0`, `DJANGO_SECRET_KEY` and `DJANGO_ALLOWED_HOSTS`, and run `collectstatic` behind a real web server.
