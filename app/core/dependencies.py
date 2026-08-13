from fastapi import (
    Depends,
    HTTPException,
    status
)

from jose import (
    JWTError,
    jwt
)

from fastapi.security import (
    OAuth2PasswordBearer
)

from app.core.config import settings

from app.core.model_loader import (
    model,
    scaler
)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


def get_model():

    return model


def get_scaler():

    return scaler


def get_current_user(
    token: str = Depends(
        oauth2_scheme
    )
):

    credentials_exception = HTTPException(

        status_code=status.HTTP_401_UNAUTHORIZED,

        detail="Could not validate credentials",

        headers={
            "WWW-Authenticate": "Bearer"
        }

    )


    try:

        payload = jwt.decode(

            token,

            settings.SECRET_KEY,

            algorithms=[
                settings.ALGORITHM
            ]

        )


        username = payload.get(
            "sub"
        )


        if username is None:

            raise credentials_exception


        return username


    except JWTError:

        raise credentials_exception