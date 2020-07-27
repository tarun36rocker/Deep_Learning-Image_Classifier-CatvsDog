from __future__ import division, print_function
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
UPLOAD_FOLDER ='UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = pickle.load(open('model.pkl', 'rb'))

def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image=image.img_to_array(test_image) #converts it into 3d array
    test_image=np.expand_dims(test_image,axis=0)
    prediction = model.predict(test_image)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path=file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        f.save(file_path)
        print("FILE PATH IS : ",file_path)
        # Make prediction
        prediction = model_predict(file_path, model)
        if(prediction[0]==0):
            output="Cat!!"
        else:
            output="Dog!!"

    return render_template('index.html', prediction_text='Your animal is a : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
