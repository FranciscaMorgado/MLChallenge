# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:42:39 2019

@author: Francisca
"""

from flask import Flask, render_template,url_for,request

import numpy as np
import pickle 
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    enc = pickle.load(open('encoder.sav', 'rb'))
    stats = np.load('data_stats.npy')
    pca = pickle.load(open('pca.sav', 'rb'))
    filename = 'model.sav'
    model = pickle.load(open(filename, 'rb'))
    
    if request.method == 'POST':
        file = request.files["FileUpload"]  
        data = pd.read_csv(file)
        
        ### SPLIT CATEGORICAL DATA FROM NUMERICAL DATA ###
        categorical = []
        numerical = []
        for name in data.columns:
            if name=='LABEL':
                continue
            if data[name].dtype=='object':
                categorical.append(data[name].values.tolist())
            else:
                numerical.append(data[name].values.tolist())   
        categorical = np.swapaxes(np.array(categorical), 0,1)
        numerical = np.swapaxes(np.array(numerical), 0,1)
        categorical = enc.transform(categorical).toarray()
        data = np.concatenate((categorical, numerical), axis=1)
        
        ### STANDARDIZATION ###
        data_standardized = data.copy()
        for i in range(len(data[0])):
            col = data[:,i]
            col_mean = stats[0][i]
            col_std = stats[1][i]
            data_standardized[:,i] = (col-col_mean)/col_std
        
        ### PCA ###
        data_standardized = pca.fit_transform(data_standardized)
        
        my_prediction = list(model.predict(data_standardized))
    return render_template('result.html', prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
