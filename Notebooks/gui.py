#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import os

# Correct path to the model file
model_path = '/workspaces/Fraud-Detection-ML-Project/Notebooks/random_forest_model.pkl'


# Load the model
try:
    model = joblib.load(model_path)
    st.success("Model successfully loaded!")
except FileNotFoundError:
    st.error("Model file not found. Please make sure the 'random_forest_model.pkl' file is in the correct location.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Title
st.title("Financial Fraud Detection Application")

# Get input data from the user
st.subheader("Please enter the transaction features:")
input_data = {}

# Input fields for V1 to V28 data
for i in range(1, 29):
    feature = f"V{i}"
    input_data[feature] = st.number_input(f"Enter the value for {feature}:", value=0.0, format="%.4f")

# Input for Amount data
input_data['Amount'] = st.number_input("Enter the Amount value:", value=0.0, format="%.2f")

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Make a prediction
if st.button("Make Prediction"):
    try:
        prediction = model.predict(input_df)
        result = "Fraud" if prediction[0] == 1 else "Normal Transaction"
        st.success(f"Prediction Result: {result}")
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {e}")
