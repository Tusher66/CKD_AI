import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("../dataset/patients.csv")

print("========== DATASET ==========")
print("Total rows:", len(df))

print("\nTarget distribution:")
print(df["CKD"].value_counts())


# ==========================================
# 2. FEATURES & TARGET
# ==========================================

X = df[["Age", "BP", "Creatinine"]]
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


# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(random_state=42),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ==========================================
# 6. TRAIN & EVALUATE
# ==========================================

results = []


for name, model in models.items():

    print("\n================================")
    print(f"Training: {name}")
    print("================================")

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

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

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })


# ==========================================
# 7. CREATE RESULT TABLE
# ==========================================

results_df = pd.DataFrame(results)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n\n")
print("==============================================")
print("           MODEL COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)

print("==============================================")