# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import sklearn
loaded_model = pickle.load(open('C:/Users/Affan/Desktop/ml_model_deployment/rainfall_trained_model.pkl','rb'))
model = loaded_model["model"]
feature_name = loaded_model["feature_name"]
input = (1022.2,14.1,78	,90	,0,30.0,28.5 )
np_as_array = np.asarray(input)
input_reshaped = np_as_array.reshape(1,-1)
prediction = model.predict(input_reshaped)
# print(prediction)
if (prediction[0] == 0):
  print('no rainfall')
else:
  print('rainfall')