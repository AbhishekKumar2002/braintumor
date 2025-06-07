# app.py
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import tempfile

MODEL_URL = "https://drive.google.com/uc?export=download&id=1MBXA6SDfhauS7P7qjfBJhlwqzYVLpj4O"

app = FastAPI()

def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download model")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        f.write(response.content)
        model_path = f.name

    model = tf.keras.models.load_model(model_path)
    return model

model = load_model_from_url(MODEL_URL)

class InputData(BaseModel):
    pregnancies: int
    glucose: float
    bloodpressure: float
    skinthickness: float
    insulin: float
    bmi: float
    diabetespedigreefunction: float
    age: int

@app.post("/predict")
async def predict(data: InputData):
    input_data = [[
        data.pregnancies,
        data.glucose,
        data.bloodpressure,
        data.skinthickness,
        data.insulin,
        data.bmi,
        data.diabetespedigreefunction,
        data.age
    ]]
    prediction = model.predict(input_data)
    result = int(prediction[0][0] > 0.5)
    return {"prediction": result}
