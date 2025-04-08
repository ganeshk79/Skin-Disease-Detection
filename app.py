# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os
import werkzeug

from dotenv import load_dotenv  # Add this import
load_dotenv()  # Load environment variables

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Even more suppressed logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model("sd_model.keras")

# Define the classes (adjust according to your model's output)
# Should match your notebook's classes
class_names=['actinic keratosis', 'basal cell carcinoma','pigmented benign keratosis', 'dermatofibroma', 'melanoma', 'nevus', 'vascular lesion']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if (request.method =="POST"):    
        # Save the uploaded image to a temporary location
        # img_path = os.path.join('uploads', file.filename)
        # file.save(img_path)
        imagefile=request.files['file']
        filename =werkzeug.utils.secure_filename(imagefile.filename)
	UPLOAD_FOLDER = "uploadimage"
	os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists
	img_path = os.path.join(UPLOAD_FOLDER, filename)
	imagefile.save(img_path)

        img_path="./uploadimage/"+filename

        # Preprocess the image img = image.load_img(img_path, target_size=(192, 256), color_mode="rgb")
        # Change this line in your Flask app:
	img = image.load_img(img_path, target_size=(192, 256), color_mode="rgb")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.astype('float32') / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]
        print("Predictions:", predictions)
        print("Predicted class index:", predicted_class)
        print("Predicted label:", predicted_label)


        # Clean up the uploaded image
        #os.remove(img_path)

        return jsonify({'predicted_class': predicted_label})

if __name__ == '__main__':
    # Run with Waitress for production
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
