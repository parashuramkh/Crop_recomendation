# app.py
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')  # Ensure this matches the actual model filename

# Load the datasets
pin_data = pd.read_csv('PIN.csv', encoding='latin1')
apc_data = pd.read_csv('APC.csv', encoding='latin1')

# Initialize the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(apc_data['Crop'].unique())  # Ensure 'Crop' column exists in apc_data

def fetch_weather_data(lat, lon, api_key):
    weather_api_query = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{datetime.today().strftime("%Y-%m-%d")}?unitGroup=us&key={api_key}&contentType=json'
    try:
        response = urllib.request.urlopen(weather_api_query)
        data = response.read()
        return json.loads(data.decode('utf-8'))
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

@app.route('/')
def index():
    return "Welcome to the Crop Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_pincode = data['pincode']
    land_size = data['land_size']

    if user_pincode in pin_data['Pincode'].astype(str).values:
        row = pin_data[pin_data['Pincode'] == int(user_pincode)].iloc[0]
        latitude = row['Latitude']
        longitude = row['Longitude']

        # Fetch weather data
        api_key = os.getenv('API_KEY')  # Get the API key from environment variables
        weather_data = fetch_weather_data(latitude, longitude, api_key)

        if weather_data is None:
            return jsonify({"error": "Error fetching weather data."}), 500

        # Analyze weather data
        current_day = weather_data['days'][0]
        temperature = current_day['temp']
        humidity = current_day['humidity']

        # Prepare features for prediction
        features = np.array([[latitude, longitude, temperature, humidity]])
        predicted_index = model.predict(features)
        predicted_crop = label_encoder.inverse_transform(predicted_index)

        # Get crop information
        crop_data = apc_data[apc_data['Crop'] == predicted_crop[0]]
        average_yield = crop_data['Average_Yield'].values[0]
        estimated_production = land_size * average_yield

        return jsonify({
            "predicted_crop": predicted_crop[0],
            "average_yield": average_yield,
            "estimated_production": estimated_production
        })
    else:
        return jsonify({"error": "Pincode not found."}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
