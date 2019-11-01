# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 01:22:46 2019

@author: ASUS
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#import os
#os.getcwd()
app = Flask(__name__)
model = pickle.load(open('employee_attrition2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('GUI.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('GUI.html', prediction ='1 for staying 0 for leaving $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)