import streamlit as st
import pickle
import numpy as np
from sklearn.metrics import r2_score

# Load the trained model and scaler using pickle
with open('house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize or load data storage
if 'input_data_list' not in st.session_state:
    st.session_state['input_data_list'] = []
if 'predicted_list' not in st.session_state:
    st.session_state['predicted_list'] = []
if 'actual_price_list' not in st.session_state:  # To store actual prices
    st.session_state['actual_price_list'] = []

# Title of the app
st.title('House Price Prediction')

# Create input fields for the user to enter data
house_age = st.number_input('House Age (years)', min_value=0.0, value=10.0, step=0.1)
distance_to_mrt = st.number_input('Distance to the nearest MRT station (meters)', min_value=0.0, value=1000.0, step=0.1)
num_convenience_stores = st.number_input('Number of Convenience Stores', min_value=0, value=5, step=1)
actual_price = st.number_input('Actual House Price (for R² calculation)', min_value=0.0, value=50.0, step=0.1)  # New input for actual price

# Predict button
if st.button('Predict'):
    # Prepare the input for prediction
    input_data = np.array([house_age, distance_to_mrt, num_convenience_stores]).reshape(1, -1)
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(input_data_scaled)[0]

    # Append input data, prediction, and actual price to session state for future R² calculations
    st.session_state['input_data_list'].append(input_data_scaled.flatten().tolist())
    st.session_state['predicted_list'].append(prediction)
    st.session_state['actual_price_list'].append(actual_price)  # Store actual price

    # Display the result
    st.success(f'The predicted house price is {prediction:.4f}')

# Calculate and display R² score if there are multiple predictions
if len(st.session_state['predicted_list']) > 1:
    # Convert lists to arrays for calculation
    y_pred_all = np.array(st.session_state['predicted_list'])
    y_actual_all = np.array(st.session_state['actual_price_list'])  # Use actual prices

    # Calculate R² score using actual prices and corresponding predictions
    r2 = r2_score(y_actual_all, y_pred_all)
    
    # Display the R² score
    st.write(f'R² Score for all predictions: {r2:.4f}')
else:
    st.info('Enter more data points to calculate the R² score.')
