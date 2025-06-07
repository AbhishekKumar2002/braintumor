from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from PIL import Image
import tensorflow as tf
from io import BytesIO

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749284032628.bin"

# Load model from BytesIO (no need for h5py)
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Could not fetch model from URL.")
    return tf.keras.models.load_model(BytesIO(response.content))

model = load_model_from_url(MODEL_URL)

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read uploaded file from form-data
        file = request.files['file']
        image = Image.open(file).convert("RGB")
        image = image.resize((150, 150))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)

        # Predict
        preds = model.predict(img_array)
        label_index = np.argmax(preds)
        label = class_names[label_index]
        confidence = float(preds[0][label_index])

        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
