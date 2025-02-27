from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import werkzeug

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = os.path.abspath("skin_disease_model.keras")
model = load_model(model_path)

# Print model summary to verify input shape
model.summary()

# Get expected input shape dynamically
expected_shape = model.input_shape  # Expected format: (None, height, width, channels)
_, img_height, img_width, img_channels = expected_shape  # Extract expected dimensions

# Define class names (adjust based on your model)
class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma',
    'nevus', 'pigmented benign keratosis', 'seborrheic keratosis',
    'squamous cell carcinoma', 'vascular lesion'
]

# Ensure upload directory exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Secure filename and save the uploaded file
        filename = werkzeug.utils.secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(img_height, img_width), color_mode="rgb")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.astype('float32') / 255.0  # Normalize pixel values

        print(f"Preprocessed image shape: {img_array.shape}")

        # Ensure input matches model's expected shape
        if img_array.shape[1:] != (img_height, img_width, img_channels):
            return jsonify({'error': f'Invalid image shape: {img_array.shape[1:]}, expected: {(img_height, img_width, img_channels)}'}), 400

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        print(f"Predictions: {predictions}")
        print(f"Predicted class index: {predicted_class}")
        print(f"Predicted label: {predicted_label}")

        # Optionally remove uploaded file after processing
        #os.remove(img_path)

        return jsonify({'predicted_class': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=4000, threaded=False)
