# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:58:53 2025

@author: Affan
"""

import pickle
import streamlit as st
import numpy as np

loaded_model = pickle.load(open('C:/Users/Affan/Desktop/ml_model_deployment/rainfall_trained_model.pkl','rb'))
model = loaded_model["model"]
feature_name = loaded_model["feature_name"]

def rainfall_prediction(input) :
    np_as_array = np.asarray(input)
    input_reshaped = np_as_array.reshape(1,-1)
    prediction = model.predict(input_reshaped)
    if (prediction[0] == 0):
      return 'no rainfall'
    else:
      return 'rainfall'

def main():
    
    st.title("Rainfall Predictor")
    
    Pressure = st.text_input('Enter the Pressure')
    Dewpoint = st.text_input('Enter the Dewpoint')
    Humidity = st.text_input('Enter the Humidity')
    Cloud = st.text_input('Enter the Cloud')
    Sunshine = st.text_input('Enter the Sunshine')
    Winddirection = st.text_input('Enter the Winddirection')
    Windspeed = st.text_input('Enter the Windspeed')
     
    prediction = ''
    
    if st.button("Predict"):
        try:
            input_data = [float(Pressure), float(Dewpoint), float(Humidity),
                          float(Cloud), float(Sunshine), float(Winddirection),
                          float(Windspeed)]
            prediction = rainfall_prediction(input_data)
        except ValueError:
            prediction = "Please enter valid numeric values!"
         
    st.success(prediction)

    
if __name__ == '__main__':
    main()