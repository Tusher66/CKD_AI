from pydantic import BaseModel


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str