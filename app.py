from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load class names from class_names.json file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Load the trained model (best_model.keras or .h5)
model = tf.keras.models.load_model('best_model.keras')  # Or .h5

@app.route('/')
def index():
    return "Welcome to Tomato Leaf Disease Classifier!"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Convert the file to an image object
        img = image.load_img(BytesIO(file.read()), target_size=(224, 224))  # Resize image to match model input size
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get prediction from the model
        predictions = model.predict(img_array)
        
        # Get the index of the predicted class
        class_idx = np.argmax(predictions, axis=1)
        
        # Get the predicted class label from class_names
        predicted_class = class_names[class_idx[0]]

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        # If there's an error, return the error message
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
