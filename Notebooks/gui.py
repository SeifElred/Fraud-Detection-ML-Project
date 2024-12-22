#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import os

# Correct path to the model file
model_path = os.path.join(os.getcwd(), "Notebooks", "random_forest_model.pkl")

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

# Sidebar Instructions
st.sidebar.title("Instructions")
st.sidebar.write("1. Enter the transaction details in the fields below.\n"
                 "2. Click 'Make Prediction' to check if the transaction is fraud or normal.\n"
                 "3. Ensure all input values are realistic and within valid ranges.")

# Subheader for user input
st.subheader("Please enter the transaction features:")
input_data = {}

# Input fields for V1 to V28
for i in range(1, 29):
    feature = f"V{i}"
    input_data[feature] = st.number_input(
        f"Enter the value for {feature}:", 
        value=0.0, 
        min_value=-50.0, 
        max_value=50.0, 
        format="%.4f"
    )

# Input for Amount
input_data['Amount'] = st.number_input(
    "Enter the Amount value:", 
    value=0.0, 
    min_value=0.0, 
    max_value=100000.0, 
    format="%.2f"
)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Display input data for debugging
st.write("Input DataFrame for prediction:", input_df)

# Make a prediction
if st.button("Make Prediction"):
    try:
        # Ensure the DataFrame has the correct data type
        input_df = input_df.astype('float64')

        # Perform prediction
        prediction = model.predict(input_df)
        result = "Fraud" if prediction[0] == 1 else "Normal Transaction"
        st.success(f"Prediction Result: {result}")

        # Display probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write(f"Probability of Fraud: {probabilities[1]:.2%}")
            st.write(f"Probability of Normal Transaction: {probabilities[0]:.2%}")
    except ValueError as ve:
        st.error(f"ValueError: {ve}. Please check your input.")
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {e}")
