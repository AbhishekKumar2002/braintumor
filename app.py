import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import tempfile

app = FastAPI()

MODEL_URL = "https://drive.google.com/uc?export=download&id=1MBXA6SDfhauS7P7qjfBJhlwqzYVLpj4O"

def load_model_from_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(response.content)
            model_path = tmp.name
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print("Model load error:", str(e))
        raise RuntimeError("Failed to load the model")

try:
    model = load_model_from_url(MODEL_URL)
except Exception:
    model = None  # fallback to safe state if model failed to load

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
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
