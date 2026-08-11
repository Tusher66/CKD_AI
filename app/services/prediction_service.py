import pandas as pd

from app.core.model_loader import (
    model,
    scaler
)

from app.schemas.patient import Patient


def predict_patient(
    patient: Patient
):

    data = pd.DataFrame(
        {
            "Age": [patient.Age],

            "BP": [patient.BP],

            "Creatinine": [
                patient.Creatinine
            ]
        }
    )

    data_scaled = scaler.transform(
        data
    )

    prediction = model.predict(
        data_scaled
    )

    probability = model.predict_proba(
        data_scaled
    )

    return {
        "prediction": int(
            prediction[0]
        ),

        "probability": float(
            probability[0][1]
        ),

        "model": "Random Forest",

        "version": "1.0.0"
    }