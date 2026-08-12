import pandas as pd

from app.schemas.patient import Patient


def predict_patient(
    patient: Patient,
    model,
    scaler
):

    data = pd.DataFrame(
        {
            "Age": [
                patient.Age
            ],

            "BP": [
                patient.BP
            ],

            "Creatinine": [
                patient.Creatinine
            ]
        }
    )


    scaled_data = scaler.transform(
        data
    )


    prediction = model.predict(
        scaled_data
    )


    probability = model.predict_proba(
        scaled_data
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