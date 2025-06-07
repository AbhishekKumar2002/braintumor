from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# URL of your model file (must be .h5 format if using tf.keras.models.load_model)
MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749284032628.bin"

def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Could not fetch model from URL.")
    model_path = BytesIO(response.content)
    model = tf.keras.models.load_model(model_path)
    return model

# Load model once globally
model = load_model_from_url(MODEL_URL)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # example classes

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize((150, 150))  # use the size your model was trained on
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 150, 150, 3)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        image_array = preprocess_image(file)
        prediction = model.predict(image_array)[0]
        label_idx = np.argmax(prediction)
        label = class_names[label_idx]
        confidence = float(prediction[label_idx])

        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=10000)
