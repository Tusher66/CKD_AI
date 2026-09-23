import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "patients.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "ckd_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"

FEATURE_COLUMNS = ["Age", "BP", "Creatinine"]
TARGET_COLUMN = "CKD"


def train_and_save_model():
    print(f"1. Loading dataset from: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"   Loaded {len(df)} records. Columns: {list(df.columns)}")

    # Validate columns
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in dataset")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"   Class distribution:\n{y.value_counts().to_dict()}")

    # 2. Train / Test Split
    print("\n2. Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Scale Features
    print("3. Fitting StandardScaler on training set and transforming data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Model
    print("4. Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate Model
    print("\n5. Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 45)
    print("           MODEL EVALUATION RESULTS          ")
    print("=" * 45)
    print(f"Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature Importance
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    print("Feature Importance:")
    print(importance_df.to_string(index=False))

    # 6. Save Model and Scaler
    print("\n6. Saving model and scaler artifacts...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"   Saved model to: {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"   Saved scaler to: {SCALER_PATH}")

    print("\n Retraining and saving completed successfully!")
    return model, scaler


if __name__ == "__main__":
    train_and_save_model()
