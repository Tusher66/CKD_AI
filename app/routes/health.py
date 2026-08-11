from fastapi import APIRouter


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
        "model": "Random Forest",

        "version": "1.0.0",

        "features": [
            "Age",
            "BP",
            "Creatinine"
        ]
    }