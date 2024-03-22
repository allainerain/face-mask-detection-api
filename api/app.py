from flask import Flask, request, render_template
import numpy as np
import cv2
from flask_cors import CORS, cross_origin
import tensorflow as tf
from flask import jsonify

#Keras imports
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input

app = Flask(__name__)
CORS(app)

# loead the tflite model
interpreter = tf.lite.Interpreter(model_path="../models/mask_detection_lite.tflite")
interpreter.allocate_tensors()

#load keras model
MODEL_PATH = '../models/mask_detection.h5'
model = load_model(MODEL_PATH)     

# define the classes
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
        files = request.files.getlist('file') #obtain the files in the request
        results = [] 
        for file in files:
            print(1)
            result = process_image(file)
            results.append(result)
        return jsonify(results)
    return None

#process the files upload
def process_image(file):
    results = []

    filestr = file.read()  # read the uploaded file given to the function
    file_bytes = np.frombuffer(filestr, np.uint8)  

    #processing the image
    face = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)  
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = np.expand_dims(face, axis=0)

    keras_face = preprocess_input(face)
    results.append(perform_inference_keras(keras_face))

    tflite_face = (face.astype(np.float32) - 127.5) / 127.5  # normalizing
    results.append(perform_inference_tflite(tflite_face))

    return results

#use the model to infer
def perform_inference_tflite(face):
    # interpret using the tflite interpreter
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], face)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    # process the output
    output_class = np.argmax(output)
    predicted_class = mask_label[output_class]
    confidence = float(np.max(output))
    return {
        'model':'tflite', 
        'class': predicted_class, 
        'confidence': confidence
    }

def perform_inference_keras(face):
    # getting the prediction
    prediction = model.predict(face, batch_size=32)
    output = np.argmax(prediction)

    #interpreting the prediction
    predicted_class = mask_label[output]
    max_prob = np.max(prediction)

    #returning the value
    return {
        'model': 'keras',
        'class': predicted_class,
        'confidence': float(max_prob)
    }

if __name__ == '__main__':
    app.run(debug=True)
