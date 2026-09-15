import pandas as pd


def predict_patient(
    patient,
    model,
    scaler
):

    data = pd.DataFrame({
        "Age": [patient.Age],
        "BP": [patient.BP],
        "Creatinine": [patient.Creatinine]
    })

    scaled_data = scaler.transform(data)

    prediction = model.predict(
        scaled_data
    )

    probabilities = model.predict_proba(
        scaled_data
    )

    return {
        "prediction": int(prediction[0]),
        "probability": float(
            probabilities[0][1]
        )
    }