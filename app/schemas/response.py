from pydantic import BaseModel


class PredictionExplanation(BaseModel):

    Age: float

    BP: float

    Creatinine: float


class PredictionResponse(BaseModel):

    prediction: int

    probability: float

    explanation: PredictionExplanation