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

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')  # Ensure this matches the actual model filename

# Load the datasets
pin_data = pd.read_csv('PIN.csv', encoding='latin1')
apc_data = pd.read_csv('APC.csv', encoding='latin1')

# Initialize the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(apc_data['Crop'].unique())  # Ensure 'Crop' column exists in apc_data

# Directly define your API key
API_KEY = 'DCDVS6HGLC8S6F657B22M9NNM'  # Replace with your actual API key

def fetch_weather_data(lat, lon, start_date):
    weather_api_query = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}?unitGroup=us&key={API_KEY}&contentType=json'
    print(f"Fetching weather data from: {weather_api_query}")  # Log the URL
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
    try:
        data = request.json
        user_pincode = data['pincode']
        land_size = data['land_size']

        if user_pincode in pin_data['Pincode'].astype(str).values:
            row = pin_data[pin_data['Pincode'] == int(user_pincode)].iloc[0]
            latitude = row['Latitude']
            longitude = row['Longitude']

            # Fetch weather data
            start_date = datetime.today().strftime('%Y-%m-%d')  # Use the current date
            weather_data = fetch_weather_data(latitude, longitude, start_date)

            if weather_data is None:
                return jsonify({"error": "Error fetching weather data."}), 500

            # Analyze weather data
            current_day = weather_data['days'][0]
            temperature = current_day.get('temp', 0)  # Default to 0 if not found
            humidity = current_day.get('humidity', 0)  # Default to 0 if not found

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
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
