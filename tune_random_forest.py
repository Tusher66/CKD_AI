import os
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "patients.csv"
MODEL_PATH = BASE_DIR / "model" / "ckd_model.joblib"
SCALER_PATH = BASE_DIR / "model" / "scaler.joblib"


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("\n========== DATASET ==========")

print("Total rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["CKD"].value_counts())


# ==========================================
# 3. FEATURES & TARGET
# ==========================================

X = df[
    [
        "Age",
        "BP",
        "Creatinine"
    ]
]

y = df["CKD"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training:", X_train.shape)

print("Testing :", X_test.shape)


# ==========================================
# 5. SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n========== SCALING ==========")

print("Scaler fitted successfully.")


# ==========================================
# 6. RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    random_state=42
)


# ==========================================
# 7. PARAMETER GRID
# ==========================================

param_grid = {

    "n_estimators": [
        50,
        100,
        200
    ],

    "max_depth": [
        None,
        5,
        10,
        20
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]
}


# ==========================================
# 8. STRATIFIED K-FOLD
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 9. GRID SEARCH
# ==========================================

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=cv,

    scoring="f1",

    n_jobs=-1,

    verbose=1
)


# ==========================================
# 10. START TUNING
# ==========================================

print("\n======================================")

print("Starting Hyperparameter Tuning...")

print("======================================")

print("Total parameter combinations:", 108)

print("Total CV fits:", 540)


grid_search.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 11. BEST PARAMETERS
# ==========================================

print("\n========== BEST PARAMETERS ==========")

print(
    grid_search.best_params_
)


# ==========================================
# 12. BEST CV SCORE
# ==========================================

print("\n========== BEST CV SCORE ==========")

print(
    f"{grid_search.best_score_:.4f}"
)


# ==========================================
# 13. BEST MODEL
# ==========================================

best_model = grid_search.best_estimator_


print("\n========== BEST MODEL ==========")

print(best_model)


# ==========================================
# 14. MODEL CLASSES
# ==========================================

print("\n========== MODEL CLASSES ==========")

print(best_model.classes_)


# ==========================================
# 15. TEST PREDICTION
# ==========================================

predictions = best_model.predict(
    X_test_scaled
)

probabilities = best_model.predict_proba(
    X_test_scaled
)[:, 1]


# ==========================================
# 16. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ==========================================
# 17. FINAL TEST RESULTS
# ==========================================

print("\n========== FINAL TEST EVALUATION ==========")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ==========================================
# 18. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ==========================================
# 19. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 20. SAVE MODEL DIRECTORY
# ==========================================

os.makedirs(
    "../model",
    exist_ok=True
)


# ==========================================
# 21. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    MODEL_PATH
)


# ==========================================
# 22. SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    SCALER_PATH
)


# ==========================================
# 23. VERIFY SAVED FILES
# ==========================================

print("\n========== MODEL SAVING ==========")

print("Model saved to:")
print(MODEL_PATH)

print("\nScaler saved to:")
print(SCALER_PATH)


# ==========================================
# 24. VERIFY MODEL LOADING
# ==========================================

print("\n========== VERIFY SAVED MODEL ==========")

loaded_model = joblib.load(
    MODEL_PATH
)

loaded_scaler = joblib.load(
    SCALER_PATH
)


print("\nLoaded Model:")
print(loaded_model)

print("\nLoaded Model Classes:")
print(loaded_model.classes_)

print("\nLoaded Scaler:")
print(loaded_scaler)


# ==========================================
# 25. VERIFY PREDICTION
# ==========================================

sample_patient = pd.DataFrame({
    "Age": [65],
    "BP": [100],
    "Creatinine": [2.8]
})


sample_scaled = loaded_scaler.transform(
    sample_patient
)


sample_prediction = loaded_model.predict(
    sample_scaled
)

sample_probability = loaded_model.predict_proba(
    sample_scaled
)


print("\n========== SAMPLE PATIENT TEST ==========")

print("\nPatient:")
print(sample_patient)

print("\nPrediction:")
print(sample_prediction[0])

print("\nProbability:")
print(sample_probability[0])

print(
    f"\nNo CKD Probability : "
    f"{sample_probability[0][0]:.4f}"
)

print(
    f"CKD Probability    : "
    f"{sample_probability[0][1]:.4f}"
)


# ==========================================
# 26. FINAL MESSAGE
# ==========================================

print("\n==========================================")

print("Random Forest training completed.")

print("Best model saved successfully.")

print("Scaler saved successfully.")

print("==========================================")


sample_patient = pd.DataFrame({
    "Age": [55],
    "BP": [150],
    "Creatinine": [1.1]
})

sample_scaled = loaded_scaler.transform(
    sample_patient
)

sample_prediction = loaded_model.predict(
    sample_scaled
)

sample_probability = loaded_model.predict_proba(
    sample_scaled
)