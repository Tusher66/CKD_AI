from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.prediction import router as prediction_router


app = FastAPI(
    title="CKD Prediction API",
    description="AI-based Chronic Kidney Disease Prediction API",
    version="1.0.0"
)


app.include_router(
    health_router
)

app.include_router(
    prediction_router
)