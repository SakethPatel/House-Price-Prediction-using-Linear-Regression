# model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data_raw = pd.read_csv(file_path)
    df = pd.DataFrame(data_raw)
    
    df.loc[(df['Distance to the nearest MRT station'] > 1000) & 
           (df['Distance to the nearest MRT station'] <= 2000) & 
           (df['House price of unit area'] == 0), 'House price of unit area'] = 28

    df.loc[(df['Distance to the nearest MRT station'] > 2000) & 
           (df['House price of unit area'] == 0), 'House price of unit area'] = 15

    

    df.drop(columns=['Transaction date'], axis=1, inplace=True)

    return df

# Train the model
def train_model(df):
    features = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores']
    X = df[features]
    y = df['House price of unit area']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the scaler and scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler

# Load data, train model, and initialize scaler
df = load_and_preprocess_data('Real_Estate.csv')
model, scaler = train_model(df)

# Save the trained model and scaler using pickle
with open('house_price_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
