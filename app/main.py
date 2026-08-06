from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from app.core.model_loader import model, scaler


app = FastAPI()

@app.get("/")
def home():

    return {
        "message":"Hello AI"
    }


# model = joblib.load("model/ckd_model.joblib")

# scaler = joblib.load("model/scaler.joblib")

class Patient(
    BaseModel
):
    Age:int
    BP:int
    Creatinine:float

@app.post("/predict")
def predict(
predict:Patient
):

    data = pd.DataFrame({
        "Age":[predict.Age],
        "BP":[predict.BP],
        "Creatinine":[predict.Creatinine]
    })

    # data = scaler.transform(
    #     data
    # )

    # prediction = model.predict(
    #     data
    # )

    data = scaler.transform(data)

    prediction = model.predict(data)

    return{
        "Prediction":int(
            prediction[0]
        )
    }
    
    # return {
    #     "Age":predict.Age,
    #     "BP":predict.BP,
    #     "Creatinine":predict.Creatinine
    # }