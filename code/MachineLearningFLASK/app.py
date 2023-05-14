# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:47:01 2023

@author: user
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
import joblib
# I use joblib save the model

# Load model
app = Flask(__name__)
model = joblib.load(open('model/bayesian.pkl', 'rb'))

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict')
def predict():
    # 傳入測試資料
    
    return

if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
    
    
