import pandas as pd
import joblib

# Load Model

model = joblib.load(
        "model/ckd_model.joblib"
)

# Load Scaler

scaler = joblib.load(
    "model/scaler.joblib"
)


# New Patient
patient = pd.DataFrame(
    {
        "Age":[6],
        "BP":[95],
        "Creatinine":[3.1]
    }
)

# Scale
patient_scaled = scaler.transform(
patient
)

# Prediction
prediction = model.predict(patient_scaled)

print("Prediction", prediction)