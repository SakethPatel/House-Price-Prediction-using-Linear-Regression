# app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler using pickle
with open('house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the app
st.title('House Price Prediction')

# Create input fields for the user to enter data
house_age = st.number_input('House Age (years)', min_value=0.0, value=10.0, step=0.1)
distance_to_mrt = st.number_input('Distance to the nearest MRT station (meters)', min_value=0.0, value=1000.0, step=0.1)
num_convenience_stores = st.number_input('Number of Convenience Stores', min_value=0, value=5, step=1)


# Predict button
if st.button('Predict'):
    # Prepare the input for prediction
    input_data = np.array([house_age, distance_to_mrt, num_convenience_stores]).reshape(1, -1)
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(input_data_scaled)[0]

    # Display the result
    st.success(f'The predicted house price is {prediction:.4f}')

    r2_score = model.score(input_data_scaled, [prediction])
    st.write(f'R² Score for the prediction: {r2_score:.4f}')
