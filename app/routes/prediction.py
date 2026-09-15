import logging

from io import StringIO

import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends
)

from fastapi.responses import (
    StreamingResponse
)

from app.core.dependencies import (
    get_current_admin,
    get_model,
    get_scaler,
    get_current_user
)

from app.schemas.patient import (
    Patient
)

from app.schemas.response import (
    PredictionResponse
)

from sqlalchemy.orm import Session

from app.core.database import get_db

from app.models.user import User

from app.models.prediction import (
    PredictionHistory
)

from app.services.prediction_service import (
    predict_patient
)


logger = logging.getLogger(
    __name__
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)


# =========================================================
# SINGLE PATIENT PREDICTION
# =========================================================

@router.post("/predict")
def predict(
    patient: Patient,
    db: Session = Depends(get_db),
    model=Depends(get_model),
    scaler=Depends(get_scaler),
    current_user: User = Depends(get_current_user)
):

    result = predict_patient(
        patient,
        model,
        scaler
    )

    prediction_history = PredictionHistory(

        user_id=current_user.id,

        age=patient.Age,

        bp=patient.BP,

        creatinine=patient.Creatinine,

        prediction=result["prediction"],

        probability=result["probability"]

    )

    db.add(
        prediction_history
    )

    db.commit()

    db.refresh(
        prediction_history
    )

    return {
        "message": "Prediction completed successfully",

        "prediction": result["prediction"],

        "probability": result["probability"],

        "prediction_id": prediction_history.id
    }


# =========================================================
# CSV BATCH PREDICTION
# =========================================================

@router.post(
    "/predict/csv"
)
async def predict_csv(

    file: UploadFile = File(...),

    model=Depends(
        get_model
    ),

    scaler=Depends(
        get_scaler
    ),
    
    current_user=Depends(
        get_current_user
    )

):

    try:

        # -------------------------------------------------
        # 1. Check file
        # -------------------------------------------------

        if not file.filename:

            raise HTTPException(

                status_code=400,

                detail="File name is required"

            )


        if not file.filename.lower().endswith(
            ".csv"
        ):

            raise HTTPException(

                status_code=400,

                detail="Only CSV files are allowed"

            )


        # -------------------------------------------------
        # 2. Read CSV
        # -------------------------------------------------

        df = pd.read_csv(
            file.file
        )


        # -------------------------------------------------
        # 3. Check empty CSV
        # -------------------------------------------------

        if df.empty:

            raise HTTPException(

                status_code=400,

                detail="CSV file is empty"

            )


        # -------------------------------------------------
        # 4. Required columns
        # -------------------------------------------------

        required_columns = [

            "Age",

            "BP",

            "Creatinine"

        ]


        # -------------------------------------------------
        # 5. Validate columns
        # -------------------------------------------------

        missing_columns = [

            column

            for column in required_columns

            if column not in df.columns

        ]


        if missing_columns:

            raise HTTPException(

                status_code=400,

                detail={
                    "message": "Missing required columns",
                    "columns": missing_columns
                }

            )


        # -------------------------------------------------
        # 6. Select features
        # -------------------------------------------------

        features = df[
            required_columns
        ]


        # -------------------------------------------------
        # 7. Validate numeric data
        # -------------------------------------------------

        for column in required_columns:

            features[column] = pd.to_numeric(

                features[column],

                errors="coerce"

            )


        if features.isnull().any().any():

            raise HTTPException(

                status_code=400,

                detail="CSV contains invalid or empty numeric values"

            )


        # -------------------------------------------------
        # 8. Scale data
        # -------------------------------------------------

        scaled_data = scaler.transform(
            features
        )


        # -------------------------------------------------
        # 9. Predict
        # -------------------------------------------------

        predictions = model.predict(
            scaled_data
        )


        # -------------------------------------------------
        # 10. Probability
        # -------------------------------------------------

        probabilities = model.predict_proba(
            scaled_data
        )


        # -------------------------------------------------
        # 11. Add results
        # -------------------------------------------------

        df["Prediction"] = predictions


        df["Probability"] = probabilities[:, 1]


        # -------------------------------------------------
        # 12. Create CSV in memory
        # -------------------------------------------------

        output = StringIO()


        df.to_csv(

            output,

            index=False

        )


        output.seek(0)


        # -------------------------------------------------
        # 13. Return CSV
        # -------------------------------------------------

        logger.info(

            "CSV prediction completed. Patients: %s",

            len(df)

        )


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

@router.get("/prediction/history")
def get_prediction_history(

    db: Session = Depends(get_db),

    current_user: User = Depends(
        get_current_user
    )

):

    history = db.query(
        PredictionHistory
    ).filter(
        PredictionHistory.user_id
        == current_user.id
    ).order_by(
        PredictionHistory.created_at.desc()
    ).all()

    return {
        "count": len(history),

        "data": [
            {
                "id": item.id,

                "age": item.age,

                "bp": item.bp,

                "creatinine": item.creatinine,

                "prediction": item.prediction,

                "probability": item.probability,

                "created_at": item.created_at

            }

            for item in history
        ]
    }

@router.get("/admin/predictions")
def get_all_predictions(

    db: Session = Depends(get_db),

    current_admin: User = Depends(
        get_current_admin
    )

):

    history = db.query(
        PredictionHistory
    ).order_by(
        PredictionHistory.created_at.desc()
    ).all()

    return {
        "count": len(history),

        "data": [
            {
                "id": item.id,

                "user_id": item.user_id,

                "age": item.age,

                "bp": item.bp,

                "creatinine": item.creatinine,

                "prediction": item.prediction,

                "probability": item.probability,

                "created_at": item.created_at

            }
            for item in history
        ]
    }