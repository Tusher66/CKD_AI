import pandas as pd

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
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("../dataset/patients.csv")

print("========== DATASET ==========")

print("Total rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["CKD"].value_counts())


# ==========================================
# 2. FEATURES & TARGET
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
# 3. TRAIN / TEST SPLIT
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
# 4. SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    random_state=42
)


# ==========================================
# 6. PARAMETER GRID
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
# 7. STRATIFIED K-FOLD
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 8. GRID SEARCH
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
# 9. START TUNING
# ==========================================

print("\n======================================")

print("Starting Hyperparameter Tuning...")

print("======================================")


grid_search.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 10. BEST PARAMETERS
# ==========================================

print("\n========== BEST PARAMETERS ==========")

print(
    grid_search.best_params_
)


# ==========================================
# 11. BEST CV SCORE
# ==========================================

print("\n========== BEST CV SCORE ==========")

print(
    f"{grid_search.best_score_:.4f}"
)


# ==========================================
# 12. BEST MODEL
# ==========================================

best_model = grid_search.best_estimator_


print("\n========== BEST MODEL ==========")

print(best_model)


# ==========================================
# 13. TEST PREDICTION
# ==========================================

predictions = best_model.predict(
    X_test_scaled
)

probabilities = best_model.predict_proba(
    X_test_scaled
)[:, 1]


# ==========================================
# 14. EVALUATION
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
# 15. PRINT RESULTS
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
# 16. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# 17. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)