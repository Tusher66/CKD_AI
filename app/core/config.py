from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str

    APP_VERSION: str

    MODEL_PATH: str

    SCALER_PATH: str

    SECRET_KEY: str

    ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    DATABASE_URL: str


    class Config:

        env_file = ".env"

        extra = "ignore"


settings = Settings()