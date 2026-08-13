from fastapi import FastAPI

from app.core.config import settings

from app.routes.health import router as health_router
from app.routes.prediction import router as prediction_router
from app.routes.auth import router as auth

app = FastAPI(

    title=settings.APP_NAME,

    version=settings.APP_VERSION

)

app.include_router(
    health_router
)

app.include_router(
    prediction_router
)

app.include_router(
    auth
)