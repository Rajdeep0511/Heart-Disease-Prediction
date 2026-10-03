"""Canonical feature schema shared by training and serving.

Every dataset is converted to this encoding (see datasets.py), so the
website form, the trained model and the training data all agree.

Canonical encoding (UCI documentation order, zero-based):
    sex      0 female, 1 male
    cp       0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic
    fbs      1 if fasting blood sugar > 120 mg/dl
    restecg  0 normal, 1 ST-T wave abnormality, 2 left ventricular hypertrophy
    exang    1 if exercise-induced angina
    slope    0 upsloping, 1 flat, 2 downsloping   (peak-exercise ST segment)
    ca       0-3 major vessels coloured by fluoroscopy
    thal     0 normal, 1 fixed defect, 2 reversible defect
    target   1 = heart disease present
"""

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"

CP_LABELS = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}
RESTECG_LABELS = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}
SLOPE_LABELS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_LABELS = {0: "Normal", 1: "Fixed defect", 2: "Reversible defect"}
