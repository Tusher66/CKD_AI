from fastapi import (
    APIRouter,
    HTTPException,
    Depends
)

from fastapi.security import OAuth2PasswordRequestForm

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token
)

from app.schemas.auth import (
    TokenResponse
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# Temporary user
fake_user = {

    "username": "admin",

    "password": hash_password(
        "admin123"
    )

}


@router.post(
    "/login",
    response_model=TokenResponse
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):

    if form_data.username != fake_user["username"]:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )


    if not verify_password(

        form_data.password,

        fake_user["password"]

    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )


    access_token = create_access_token({

        "sub": form_data.username

    })


    return {

        "access_token": access_token,

        "token_type": "bearer"

    }