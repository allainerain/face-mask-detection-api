from flask import Flask, render_template, request, redirect, url_for, send_from_directory,flash
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
import numpy as np
import cv2
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# loading the model
MODEL_PATH = '../mask_detection.h5'
model = load_model(MODEL_PATH)     

#defining the classes
mask_label = {0: 'MASK', 1: 'UNCOVERED CHIN', 2: 'UNCOVERED NOSE', 3: 'UNCOVERED NOSE AND MOUTH', 4: "NO MASK"}
         
@app.route('/')
def home():
    return "Mask Patrol API"

@app.route('/hello',methods = ['GET'])
def hello():
    return "Hello"

# predict end point
@app.route('/predict', methods=['GET', 'POST'])
@cross_origin(allow_headers=['Content-Type'])
def upload():
    if request.method == 'POST':

        #processing
        # Access the uploaded file
        filestr = request.files['file'].read() # Read uploaded file from request
        file_bytes = np.frombuffer(filestr, np.uint8) # Convert to numpy array
        face = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED) # Decode image

        #processing
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)

        # getting the prediction
        prediction = model.predict(face, batch_size=32)
        output = np.argmax(prediction)

        #interpreting the prediction
        predicted_class = mask_label[output]
        max_prob = np.max(prediction)

        #returning the value
        return {
            'class' : predicted_class,
            'confidence' : float(max_prob)
        }
        
    return None

if __name__ == '__main__':
   app.run(debug = True)
