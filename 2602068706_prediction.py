# -*- coding: utf-8 -*-
#### DTSC6012001 - Model Deployment
"""Nama: Vira Fitriyani<br>
NIM: 2602068706<br>
MID Exam<br>
"""

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import joblib

model = joblib.load('XGB_model.pkl')
features_encoded = joblib.load('features_encoded.pkl')

def main():
    st.title("Churn Prediction - Model Deployment")
    Gender = st.radio("Gender", ['Male', 'Female'])
    Age = st.number_input("Age", 0, 100)
    Geography = st.radio("Country", ['France', 'Spain', 'Germany'])
    Tenure = st.number_input("The period of time you holds a position (in years)", 0, 100)
    IsActiveMember = st.radio("Choose status member", ['Active', 'Inactive'])
    HasCrCard = st.radio('Do you have a Credit Card', ['Yes', 'No'])
    CreditScore = st.number_input('Total Credit Score', 0, 1000)
    EstimatedSalary = st.number_input('Number of your estimated salary', 0, 10000000000)
    Balance = st.number_input('Total Balance', 0, 10000000000)
    NumOfProducts = st.number_input('Number of products', 0, 100)
    
    data = {'Gender': Gender, 'Age': int(Age), 'Geography': Geography, 'Tenure': int(Tenure), 
         'IsActiveMember': IsActiveMember, 'HasCrCard': HasCrCard, 'CreditScore': int(CreditScore),
         'EstimatedSalary': int(EstimatedSalary), 'Balance': int(Balance), 
          'NumOfProducts': int(NumOfProducts)}

    df = pd.DataFrame([list(data.values())], columns=['Gender', 'Age', 'Geography', 'Tenure', 
                                                      'IsActiveMember', 'HasCrCard', 'CreditScore', 
                                                     'EstimatedSalary', 'Balance', 'NumOfProducts'])
    df = df.replace(features_encoded)
    
    if st.button('Make Prediction'):
        features = df
        result = make_prediction(features)
        st.success(f'The Prediction is: {result}')

def make_prediction(features):
  # Convert features to a 1D array if necessary
    if isinstance(features, dict):
        features = list(features.values())
    
    # Extract numeric values
    numeric_values = [val for val in features if isinstance(val, (int, float))]
    input_array = np.array(numeric_values).reshape(1, -1)
    
    # Check for NaN or infinite values
    if np.isnan(input_array).any() or np.isinf(input_array).any():
        raise ValueError("Input array contains NaN or infinite values")
    
    # Make prediction
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
  main()
