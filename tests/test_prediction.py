import pytest

from app.services.prediction_service import predict_patient

from app.core.model_loader import (
    model,
    scaler,
    explainer
)

def test_prediction():

    patient = type(
        "Patient",
        (),
        {
            "Age": 65,
            "BP": 100,
            "Creatinine": 2.8
        }
    )()

    result = predict_patient(
        patient,
        model,
        scaler,
        explainer
    )

    assert "prediction" in result

    assert "probability" in result

    assert "explanation" in result

    assert result["prediction"] in [0, 1]

    assert 0 <= result["probability"] <= 1