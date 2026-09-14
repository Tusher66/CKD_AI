from pathlib import Path

import joblib

from app.core.config import settings


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / settings.MODEL_PATH
SCALER_PATH = BASE_DIR / settings.SCALER_PATH


model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)


print("Model loaded successfully")
print("Scaler loaded successfully")