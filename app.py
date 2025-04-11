# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
import werkzeug
import tensorflow as tf

# Configure TensorFlow for memory efficiency
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure TensorFlow to use less memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploadimage'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model(os.path.join(os.path.dirname(__file__), "sd_model.keras"))

# Define the classes (adjust according to your model's output)
# Should match your notebook's classes
#class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_names=['actinic keratosis', 'basal cell carcinoma','pigmented benign keratosis', 'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesion']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        imagefile = request.files['file']
        if imagefile.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image to a temporary location
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        imagefile.save(temp_path)

        try:
            # Preprocess the image
            img = image.load_img(temp_path, target_size=(32, 32), color_mode="rgb")
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0

            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names[predicted_class]

            return jsonify({
                'predicted_class': predicted_label,
                'confidence': float(predictions[0][predicted_class])
            })

        finally:
            # Clean up the uploaded image
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error removing file: {str(e)}")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=10000)
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
import werkzeug
import tensorflow as tf

# Configure TensorFlow for memory efficiency
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure TensorFlow to use less memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploadimage'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model(os.path.join(os.path.dirname(__file__), "sd_model.keras"))

# Define the classes (adjust according to your model's output)
# Should match your notebook's classes
#class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_names=['actinic keratosis', 'basal cell carcinoma','pigmented benign keratosis', 'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesion']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        imagefile = request.files['file']
        if imagefile.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image to a temporary location
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        imagefile.save(temp_path)

        try:
            # Preprocess the image
            img = image.load_img(temp_path, target_size=(32, 32), color_mode="rgb")
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0

            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names[predicted_class]

            return jsonify({
                'predicted_class': predicted_label,
                'confidence': float(predictions[0][predicted_class])
            })

        finally:
            # Clean up the uploaded image
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error removing file: {str(e)}")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=10000)
