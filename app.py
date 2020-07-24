import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    from keras.preprocessing import image
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        
        test_image=image.load_img(request.FILES['Picture'].name,target_size = (64, 64))
        test_image=image.img_to_array(test_image) #converts it into 3d array
        test_image=np.expand_dims(test_image,axis=0)
   
        prediction = model.predict(test_image)
        if(prediction[0]==0):
            output="Cat!!"
        else:
            output="Dog!!"
    

    return render_template('index.html', prediction_text='Your animal is a : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
