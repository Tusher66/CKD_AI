import logging

from fastapi import (
    APIRouter,
    HTTPException
)

from app.schemas.patient import Patient

from app.schemas.response import (
    PredictionResponse
)

from app.services.prediction_service import (
    predict_patient
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(patient: Patient):

    try:

        logger.info(
            "Prediction started"
        )

        result = predict_patient(
            patient
        )

        logger.info(
            "Prediction completed"
        )

        return result

    except Exception as e:

        logger.error(
            "Prediction failed: %s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Model prediction failed"
        )