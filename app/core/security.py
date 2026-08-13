from datetime import (
    datetime,
    timedelta,
    timezone
)

import bcrypt

from jose import (
    jwt,
    JWTError
)

from app.core.config import settings


def hash_password(
    password: str
):

    password_bytes = password.encode(
        "utf-8"
    )

    hashed_password = bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt()
    )

    return hashed_password.decode(
        "utf-8"
    )


def verify_password(
    plain_password: str,
    hashed_password: str
):

    plain_password_bytes = plain_password.encode(
        "utf-8"
    )

    hashed_password_bytes = hashed_password.encode(
        "utf-8"
    )

    return bcrypt.checkpw(
        plain_password_bytes,
        hashed_password_bytes
    )


def create_access_token(
    data: dict
):

    to_encode = data.copy()

    expire = datetime.now(
        timezone.utc
    ) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

    return encoded_jwt