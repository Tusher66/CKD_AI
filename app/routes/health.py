from fastapi import APIRouter

from app.core.config import settings


router = APIRouter(
    tags=["Health"]
)


@router.get("/health")
def health():

    return {
        "status": "UP"
    }


@router.get("/model-info")
def model_info():

    return {

        "application": settings.APP_NAME,

        "model": "Random Forest",

        "version": settings.APP_VERSION,

        "features": [
            "Age",
            "BP",
            "Creatinine"
        ]

    }