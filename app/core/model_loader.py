# import joblib

# model = joblib.load("model/ckd_model.joblib")
# scaler = joblib.load("model/scaler.joblib")

from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "ckd_model.joblib"
)

SCALER_PATH = (
    BASE_DIR
    / "model"
    / "scaler.joblib"
)


model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)