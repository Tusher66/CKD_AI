from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from sqlalchemy.orm import Session

from fastapi.security import (
    OAuth2PasswordRequestForm
)

from app.core.database import get_db

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token
)

from app.models.user import User

from app.schemas.auth import (
    RegisterRequest,
    TokenResponse
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post("/register")
def register(

    request: RegisterRequest,

    db: Session = Depends(
        get_db
    )

):

    existing_user = db.query(
        User
    ).filter(
        User.username == request.username
    ).first()


    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )


    existing_email = db.query(
        User
    ).filter(
        User.email == request.email
    ).first()


    if existing_email:

        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )


    user = User(

        username=request.username,

        email=request.email,

        password_hash=hash_password(
            request.password
        ),

        role="USER",

        is_active=True

    )


    db.add(user)

    db.commit()

    db.refresh(user)


    return {

        "message":
        "User registered successfully",

        "username":
        user.username,

        "email":
        user.email

    }


@router.post(
    "/login",
    response_model=TokenResponse
)
def login(

    form_data: OAuth2PasswordRequestForm = Depends(),

    db: Session = Depends(
        get_db
    )

):

    user = db.query(
        User
    ).filter(
        User.username == form_data.username
    ).first()


    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )


    if not verify_password(

        form_data.password,

        user.password_hash

    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )


    if not user.is_active:

        raise HTTPException(
            status_code=403,
            detail="User account is inactive"
        )


    access_token = create_access_token({

        "sub": user.username,

        "role": user.role

    })


    return {

        "access_token":
        access_token,

        "token_type":
        "bearer"

    }