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

from sqlalchemy.orm import Session

from app.core.config import settings

from app.core.database import get_db

from app.core.model_loader import (
    model,
    scaler
)

from app.models.user import User


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
    ),

    db: Session = Depends(
        get_db
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


    except JWTError:

        raise credentials_exception


    user = db.query(
        User
    ).filter(
        User.username == username
    ).first()


    if user is None:

        raise credentials_exception


    if not user.is_active:

        raise HTTPException(

            status_code=403,

            detail="User account is inactive"

        )


    return user


def get_current_admin(
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "ADMIN":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return current_user