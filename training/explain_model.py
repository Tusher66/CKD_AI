import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("../dataset/patients.csv")

print("========== DATASET ==========")

print("Total rows:", len(df))


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


# ==========================================
# 4. SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. TUNED RANDOM FOREST
# ==========================================

model = RandomForestClassifier(

    n_estimators=50,

    max_depth=10,

    min_samples_leaf=4,

    min_samples_split=10,

    random_state=42
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model.fit(

    X_train_scaled,

    y_train
)


print("\nModel training completed.")


# ==========================================
# 7. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

}).sort_values(

    by="Importance",

    ascending=False
)


print("\n========== FEATURE IMPORTANCE ==========")

print(
    feature_importance.to_string(
        index=False
    )
)


# ==========================================
# 8. SHAP EXPLAINER
# ==========================================

explainer = shap.TreeExplainer(model)


shap_values = explainer.shap_values(
    X_test_scaled
)


print("\nSHAP values calculated.")


# ==========================================
# 9. SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")


shap.summary_plot(

    shap_values,

    X_test_scaled,

    feature_names=X.columns,

    show=False
)


plt.tight_layout()

plt.show()