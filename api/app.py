from flask import Flask, request, render_template
import numpy as np
import cv2
from flask_cors import CORS, cross_origin
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="../mask_detection_lite.tflite")
interpreter.allocate_tensors()

# Define the classes
mask_label = {0: 'MASK', 1: 'UNCOVERED CHIN', 2: 'UNCOVERED NOSE', 3: 'UNCOVERED NOSE AND MOUTH', 4: "NO MASK"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
@cross_origin(allow_headers=['Content-Type'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        result = process_image(file)
        return result
    return None

def process_image(file):
    filestr = file.read()  # Read uploaded file from request
    file_bytes = np.frombuffer(filestr, np.uint8)  # Convert to numpy array
    face = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)  # Decode image
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = np.expand_dims(face, axis=0)
    face = (face.astype(np.float32) - 127.5) / 127.5  # Normalize the image
    return perform_inference(face)

def perform_inference(face):
    # Perform inference
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], face)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    # Interpret the prediction
    output_class = np.argmax(output)
    predicted_class = mask_label[output_class]
    confidence = float(np.max(output))
    return {'class': predicted_class, 'confidence': confidence}

if __name__ == '__main__':
    app.run(debug=True)
