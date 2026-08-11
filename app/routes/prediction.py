import logging
from io import StringIO

import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import StreamingResponse

from app.core.model_loader import (
    model,
    scaler
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


# =========================================================
# Single Patient Prediction
# =========================================================

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


# =========================================================
# CSV Batch Prediction
# =========================================================

@router.post(
    "/predict/csv"
)
async def predict_csv(
    file: UploadFile = File(...)
):

    try:

        # ---------------------------------------------
        # 1. Check file
        # ---------------------------------------------

        if not file.filename.lower().endswith(".csv"):

            raise HTTPException(
                status_code=400,
                detail="Only CSV files are allowed"
            )


        # ---------------------------------------------
        # 2. Read CSV
        # ---------------------------------------------

        df = pd.read_csv(
            file.file
        )


        # ---------------------------------------------
        # 3. Required columns
        # ---------------------------------------------

        required_columns = [
            "Age",
            "BP",
            "Creatinine"
        ]


        # ---------------------------------------------
        # 4. Check columns
        # ---------------------------------------------

        for column in required_columns:

            if column not in df.columns:

                raise HTTPException(
                    status_code=400,
                    detail=f"Missing column: {column}"
                )


        # ---------------------------------------------
        # 5. Get features
        # ---------------------------------------------

        features = df[
            required_columns
        ]


        # ---------------------------------------------
        # 6. Scale data
        # ---------------------------------------------

        scaled_data = scaler.transform(
            features
        )


        # ---------------------------------------------
        # 7. Prediction
        # ---------------------------------------------

        predictions = model.predict(
            scaled_data
        )


        # ---------------------------------------------
        # 8. Probability
        # ---------------------------------------------

        probabilities = model.predict_proba(
            scaled_data
        )


        # ---------------------------------------------
        # 9. Add results
        # ---------------------------------------------

        df["Prediction"] = predictions

        df["Probability"] = probabilities[:, 1]


        # ---------------------------------------------
        # 10. Create CSV in memory
        # ---------------------------------------------

        output = StringIO()

        df.to_csv(
            output,
            index=False
        )

        output.seek(0)


        # ---------------------------------------------
        # 11. Return CSV file
        # ---------------------------------------------

        return StreamingResponse(
            iter([
                output.getvalue()
            ]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=prediction_result.csv"
            }
        )


    except HTTPException:

        raise


    except Exception as e:

        logger.error(
            "CSV prediction failed: %s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail="CSV prediction failed"
        )