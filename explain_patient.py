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

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


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
# 7. CREATE PATIENT
# ==========================================

patient = pd.DataFrame({

    "Age": [65],

    "BP": [100],

    "Creatinine": [2.8]

})


print("\n========== PATIENT ==========")

print(patient)


# ==========================================
# 8. SCALE PATIENT
# ==========================================

patient_scaled = scaler.transform(
    patient
)


# ==========================================
# 9. PREDICTION
# ==========================================

prediction = model.predict(
    patient_scaled
)

probability = model.predict_proba(
    patient_scaled
)


print("\n========== PREDICTION ==========")

print(
    "Prediction:",
    int(prediction[0])
)

print(
    "No CKD Probability:",
    round(
        float(probability[0][0]),
        4
    )
)

print(
    "CKD Probability:",
    round(
        float(probability[0][1]),
        4
    )
)


# ==========================================
# 10. SHAP EXPLAINER
# ==========================================

explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    patient_scaled
)


print("\n========== SHAP INFORMATION ==========")

print(
    "SHAP shape:",
    shap_values.shape
)


# ==========================================
# 11. SELECT CKD CLASS
# ==========================================

# Expected shape:
#
# (1, 3, 2)
#
# 1 = patient
# 3 = features
# 2 = classes
#
# Class 0 = No CKD
# Class 1 = CKD

shap_values_ckd = shap_values[
    0,
    :,
    1
]


print(
    "CKD SHAP values:",
    shap_values_ckd
)


# ==========================================
# 12. DISPLAY FEATURE CONTRIBUTIONS
# ==========================================

contribution_df = pd.DataFrame({

    "Feature": X.columns,

    "Value": patient.iloc[0].values,

    "SHAP Contribution": shap_values_ckd

})


print(
    "\n========== FEATURE CONTRIBUTIONS =========="
)

print(
    contribution_df.to_string(
        index=False
    )
)


# ==========================================
# 13. SHAP WATERFALL
# ==========================================

print(
    "\nGenerating SHAP waterfall plot..."
)


# Create SHAP Explanation object

base_value = explainer.expected_value[1]


explanation = shap.Explanation(

    values=shap_values_ckd,

    base_values=base_value,

    data=patient.iloc[0].values,

    feature_names=X.columns.tolist()

)


shap.plots.waterfall(
    explanation,
    show=False
)


plt.tight_layout()

plt.show()