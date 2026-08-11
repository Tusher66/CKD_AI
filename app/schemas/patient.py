from pydantic import BaseModel


class Patient(BaseModel):

    Age: int

    BP: int

    Creatinine: float