from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load your model
model = load_model('best_model_weights.keras', compile=False)  #path

# Define the allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import cv2

def predict_image(file_path):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    # Load the image and convert to grayscale
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use thresholding to detect hand region
    _, thresholded = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the hand
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img_gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply a Gaussian blur to the background only
        blurred = cv2.GaussianBlur(img_rgb, (21, 21), 0)
        img_rgb = np.where(mask[:, :, None] == 255, img_rgb, blurred)

    # Resize and normalize for prediction
    img_rgb = cv2.resize(img_rgb, (32, 32))
    img_array = np.array(img_rgb) / 255.0
    img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    print('Predicted class:', predicted_class)
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Save and predict on the uploaded frame
        file_path = os.path.join('uploads', secure_filename(file.filename))
        file.save(file_path)

        # Run prediction
        predicted_class = predict_image(file_path)
        return jsonify({'prediction': predicted_class})

@app.route('/predict-manual', methods=['POST'])
def predict_manual():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return jsonify({'error': 'No file part'})

        files = request.files.getlist('files[]')
        print('Number of files:', len(files))  # Log the number of files
        print('Received files:', request.files)


        if len(files) == 1:  # Single image prediction
            file = files[0]

            if file.filename == '':
                return jsonify({'error': 'No selected file'})

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                file.save(file_path)

                predicted_class = predict_image(file_path)
                return jsonify({'prediction': f'Your image represents {predicted_class}'})

        elif len(files) > 1:  # Multiple images prediction
            predictions = []

            for i, file in enumerate(files):
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join('uploads', f'{i}_{filename}')  # Add an index as a prefix
                    file.save(file_path)

                    predicted_class = predict_image(file_path)
                    predictions.append(predicted_class)
                    
            predicted_word = ''.join(predictions)
        
            return jsonify({'prediction': f'Your images represent {predicted_word}'})

if __name__ == '__main__':
    app.run(debug=False, threaded=False) 