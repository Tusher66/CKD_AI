from pathlib import Path
import joblib
import shap

from app.core.config import settings


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / settings.MODEL_PATH
SCALER_PATH = BASE_DIR / settings.SCALER_PATH


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

explainer = shap.TreeExplainer(model)


print("================================")
print("Model loaded successfully")
print("Model:", model)
print("Model classes:", model.classes_)
print("Scaler:", scaler)
print("SHAP explainer loaded successfully")
print("================================")