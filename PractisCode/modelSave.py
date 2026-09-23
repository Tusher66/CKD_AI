import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE_DIR / "dataset" / "patients.csv")

X = df[["Age", "BP", "Creatinine"]]
y = df["CKD"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(model, MODEL_DIR / "ckd_model.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

predictions = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Model saved to: {MODEL_DIR / 'ckd_model.joblib'}")
print(f"Scaler saved to: {MODEL_DIR / 'scaler.joblib'}")