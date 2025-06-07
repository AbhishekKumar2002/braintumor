from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
import tempfile
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749284032628.bin"

def load_model_from_url(url):
    # Download model bytes
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception("Could not fetch model")

    # Create a temporary file to save the model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
        tmp_path = tmp_file.name

    # Load the model from the temp file
    model = tf.keras.models.load_model(tmp_path)

    # Remove temp file after loading
    os.remove(tmp_path)

    return model

# Load model once globally at startup (this will download and load)
model = load_model_from_url(MODEL_URL)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = data.get("input")

        if input_data is None:
            return jsonify({"error": "Missing input"}), 400

        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # fallback for local
    app.run(host="0.0.0.0", port=port)
