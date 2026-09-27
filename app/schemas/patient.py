from pydantic import (
    BaseModel,
    Field
)


class Patient(BaseModel):

    Age: int = Field(
        ...,
        ge=1,
        le=120,
        description="Patient age"
    )

    BP: int = Field(
        ...,
        ge=40,
        le=250,
        description="Blood pressure"
    )

    Creatinine: float = Field(
        ...,
        ge=0.1,
        le=20.0,
        description="Serum creatinine"
    )