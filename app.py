from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'Model/animal_detector_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Bear", "Cat", "Cow", "Deer", "Dog", "Dolphins", "Elephants", "Horse", "Lion", "Panda", "Tiger", "Zebra"]

def classify_image(img_path):
    """Classify the image and return predictions with confidence above 20%."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_classes = np.argsort(predictions[0])[::-1]
    confidence_values = predictions[0][predicted_classes]

    top_predictions = []
    for i, class_idx in enumerate(predicted_classes):
        if confidence_values[i] > 0.2:
            top_predictions.append((class_names[class_idx], float(confidence_values[i]) * 100))
        else:
            break 

    if top_predictions and top_predictions[0][1] < 40:
        return {"prediction": "Not recognized", "predictions": []}

    if not top_predictions:
        return {"prediction": "Not recognized", "predictions": []}

    return {"prediction": top_predictions[0][0], "predictions": top_predictions}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    result = classify_image(file_path)
    
    if result["prediction"] == "Not recognized":
        return jsonify({'prediction': "Not recognized", 'predictions': []})

    return jsonify({
        'prediction': result["prediction"],
        'predictions': [{"class_name": pred[0], "probability": pred[1]} for pred in result["predictions"]]
    })
    
if __name__ == '__main__':
    app.run(debug=True)
