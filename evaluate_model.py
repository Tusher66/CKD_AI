import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv(
    "dataset/patients.csv"
)


# =========================
# 2. Features
# =========================

X = df[
    [
        "Age",
        "BP",
        "Creatinine"
    ]
]


# =========================
# 3. Target
# =========================

y = df["CKD"]


# =========================
# 4. Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 5. Create Model
# =========================

model = DecisionTreeClassifier(
    random_state=42
)


# =========================
# 6. Train Model
# =========================

model.fit(
    X_train,
    y_train
)


# =========================
# 7. Prediction
# =========================

predictions = model.predict(
    X_test
)


# =========================
# 8. Probability
# =========================

probabilities = model.predict_proba(
    X_test
)[:, 1]


# =========================
# 9. Metrics
# =========================

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


# =========================
# 10. Print Metrics
# =========================

print("\n========== MODEL EVALUATION ==========\n")

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


# =========================
# 11. Confusion Matrix
# =========================

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# =========================
# 12. Classification Report
# =========================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# =========================
# 13. Feature Importance
# =========================

print("\n========== FEATURE IMPORTANCE ==========\n")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.to_string(
        index=False
    )
)

print("========== DATASET INFO ==========")

print("Total rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget classes:")
print(df["CKD"].value_counts())

print("\nUnique CKD values:")
print(df["CKD"].unique())

print("\nFeature shape:")
print(X.shape)

print("\nTrain shape:")
print(X_train.shape)

print("\nTest shape:")
print(X_test.shape)