from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define function to extract features from image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.flatten()

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    file_path = 'temp_image.jpg'
    file.save(file_path)

    # Extract features from the image
    features = extract_features(file_path)

    # Reshape features array to match the expected input shape
    features = np.array(features).reshape(1, -1)

    # Perform prediction using the loaded model
    prediction = model.predict(features)[0]

    # Delete the temporary file
    os.remove(file_path)

    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(debug=True)
