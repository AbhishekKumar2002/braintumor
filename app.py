from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from io import BytesIO
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749284032628.bin"

# Load model from URL (in-memory, no file I/O)
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Model could not be fetched from the URL.")
    return load_model(BytesIO(response.content))

# Load model once at app start
model = load_model_from_url(MODEL_URL)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = data.get("input")

        if input_data is None:
            return jsonify({"error": "Missing input"}), 400

        # Convert and reshape the input data as required
        input_array = np.array(input_data).reshape(1, -1)  # Update shape as per model input
        prediction = model.predict(input_array)
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
