import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('C:\Users\DELL\Downloads\prediction_model (1).h5')

# Load the scaler
with open('C:\Users\DELL\Downloads\standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to preprocess data and make predictions
def predict_churn(data):
    # Your preprocessing code here
    scaled_data = scaler.transform(data)
    confidence = model.predict(scaled_data)[0][0]
    return confidence

# Main Streamlit app
st.title("Customer Churn Prediction App")

# Input form for user to enter data
st.header("Enter Customer Data")
# Example inputs; replace with appropriate input fields for your data
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")
# Add more input fields as needed

if st.button("Predict Churn"):
    # Create a DataFrame with user-entered data
    user_data = pd.DataFrame({
        'Contract': [contract],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        # Add more columns as needed
    })

    # Make predictions
    confidence = predict_churn(user_data)

    # Display the prediction result
    st.success(f"The predicted churn confidence is: {confidence:.2%}")
