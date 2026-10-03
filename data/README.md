# Training data

`heart.csv` (303 Cleveland patients) ships with the project. Add more CSVs here and run
`python -m heartml.train` - every file is auto-detected, converted to one encoding,
checked, and de-duplicated.

| Dataset | Used? | Notes |
|---|---|---|
| johnsmith88/heart-disease-dataset | yes | Same ~300 Cleveland patients repeated ~3x (1025 rows). Duplicates are removed. |
| rashikrahmanpritom/heart-attack-analysis-prediction-dataset | yes | Same 303 Cleveland patients, renamed columns. |
| cherngs/heart-disease-cleveland | yes | Cleveland again, zero-based codes. |
| fedesoriano/heart-failure-prediction | yes | Adds Hungarian, Swiss and Long Beach patients (918 rows; no `ca`/`thal`). The only dataset here with substantially new patients. |
| sulianova/cardiovascular-disease-dataset | **no** | Different features (no ECG / max heart rate / ST data). |

Automatic download: `python -m heartml.download_data`
Links: https://www.kaggle.com/datasets/<slug above>
