from fastapi import FastAPI

from app.core.config import settings
from app.core.database import Base, engine

from app.models.user import User

from app.routes.health import router as health_router
from app.routes.prediction import router as prediction_router
from app.routes.auth import router as auth_router
from app.routes.admin import router as admin_router

# Create database tables
Base.metadata.create_all(bind=engine)


print("Application Started")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI based Chronic Kidney Disease Prediction API"
)


app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(auth_router)
app.include_router(admin_router)