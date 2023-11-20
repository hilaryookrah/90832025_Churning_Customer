import streamlit as st
import pickle as pkl
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

# Define your relevant features
relevant_features = ['contract_encoded', 'tenure', 'total_charges', 'monthly_charges', 'payment_method_encoded']

def create_model(activation='relu', optimizer='adam'):
    input_layer = Input(shape=(len(relevant_features),))
    hidden1 = Dense(10, activation=activation)(input_layer)
    hidden2 = Dense(20, activation=activation)(hidden1)
    hidden3 = Dense(10, activation=activation)(hidden2)
    output_layer = Dense(1, activation='sigmoid')(hidden3)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load encoding dictionaries and scaler
with open('encoding_dicts.pkl', 'rb') as file:
    encoding_dicts = pkl.load(file)
    contract_options = encoding_dicts['contract']
    payment_method_options = encoding_dicts['payment_method']

with open('standard_scaler.pkl', 'rb') as file:
    scaler = pkl.load(file)

# Load the trained model
model = tf.keras.models.load_model('prediction_model.h5')

# Function to preprocess input data
def preprocess_input(input_data):
    contract, tenure, total_charges, monthly_charges, payment_method = input_data

    # Encode 'Contract' and 'Payment Method'
    contract_encoded = contract_options[contract]
    payment_method_encoded = payment_method_options[payment_method]

    input_scaled = scaler.transform([[contract_encoded, tenure, total_charges, monthly_charges, payment_method_encoded]])
    return input_scaled

# Streamlit web app
def main():
    st.title("Churn Prediction Web App")

    st.sidebar.header("User Input")
    contract = st.sidebar.selectbox("Contract Type", list(contract_options.keys()))
    monthly_charges = st.sidebar.number_input("Monthly Charges")
    total_charges = st.sidebar.number_input("Total Charges")
    tenure = st.sidebar.number_input("Tenure")
    payment_method = st.sidebar.selectbox("Payment Method", list(payment_method_options.keys()))

    # Encode user input
    input_data = [contract, tenure, total_charges, monthly_charges, payment_method]
    input_scaled = preprocess_input(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        churn_prob = prediction[0][0]
        churn_confidence = churn_prob * 100  # Convert probability to percentage

        st.subheader("Prediction:")
        if churn_prob > 0.5:
            st.write(f"The customer is likely to churn with a probability of {churn_prob:.2f} and a confidence of {churn_confidence:.2f}%.")
        else:
            st.write(f"The customer is not likely to churn with a probability of {1 - churn_prob:.2f} and a confidence of {100 - churn_confidence:.2f}%.")

if __name__ == '__main__':
    main()
