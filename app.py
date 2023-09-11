from tkinter.ttk import Style
from turtle import color
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle
import joblib
from flask import *


app = Flask(__name__)

# model = tf.keras.models.load_model('lr_heart.h5')

# with open('ash_model_1.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

model = joblib.load('lr_heart.h5')


@app.route('/')
def home():
    #result =''
    return render_template('index.html')



@app.route('/predict', methods=['POST', 'GET'])
def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trtbps = float(request.form['trtbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalachh = float(request.form['thalachh'])
    exng = float(request.form['exng'])
    caa = float(request.form['caa'])
    feature_vector = [[age, sex, cp, trtbps, chol, fbs, restecg, thalachh,exng, caa]]
    # model_input = tf.strings.to_number([age, sex, Cp, trtbps, chol, fbs, restecg, thalachh,exng, caa])
    # model_input = tf.expand_dims(model_input, axis=0)
    # model_input = tf.expand_dims(model_input, axis=1)
    prediction = model.predict(feature_vector)
    return render_template('index.html',prediction = f'Output is : {prediction[0]}')


if __name__ == "__main__":
    app.run(debug=True)

