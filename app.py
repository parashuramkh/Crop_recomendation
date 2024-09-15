from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the datasets
pin_data = pd.read_csv('PIN.csv', encoding='latin1')
apc_data = pd.read_csv('APC.csv', encoding='latin1')

# Load the model
model = joblib.load('model.pkl')  # Ensure the correct path

# Initialize the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(apc_data['Crop'].unique())  # Ensure 'Crop' column exists

# API key for the weather service
API_KEY = 'DCDVS6HGLC8S6F657B22M9NNM'  # Replace with your actual API key

# Fetch weather data function
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

# Home route
@app.route('/')
def index():
    return '''
        <html>
        <head>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    width: 100%;
                    padding: 0 20px;
                }
                .container img {
                    height: 150px;
                    width: auto;
                }
                .center-text {
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Left image -->
                <div class="left-image">
                    <img src="https://static.wixstatic.com/media/7d5a65_9525c98a8a2c4c7e9193e78a3039d6bd~mv2.png/v1/fill/w_398,h_186,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Deenabandh_edited_edited.png" alt="Deenabandh Logo">
                </div>

                <!-- Center text -->
                <div class="center-text">
                    WELCOME TO THE CROP OPTIMIZATION SYSTEM
                </div>

                <!-- Right image -->
                <div class="right-image">
                    <img src="https://race.reva.edu.in/wp-content/uploads/2020/11/Race_Logo.png" alt="Race Logo">
                </div>
            </div>
        </body>
        </html>
    '''

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate request data
        user_pincode = request.form.get('pincode')
        land_size = request.form.get('land_size')

        if not user_pincode or not land_size:
            return jsonify({"error": "Pincode and land size are required."}), 400
        
        try:
            land_size = float(land_size)
        except ValueError:
            return jsonify({"error": "Invalid land size format. It must be a number."}), 400
        
        if user_pincode in pin_data['Pincode'].astype(str).values:
            row = pin_data[pin_data['Pincode'] == int(user_pincode)].iloc[0]
            latitude = row['Latitude']
            longitude = row['Longitude']
            place_name = row['Placename']
            district = row['District']
            state_name = row['StateName']

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

            # Prepare output message with a button to try another pincode
            return render_template_string('''
                <h1>Thank you for Using the Crop Recommendation system results for your Pincode: {{ user_pincode }} is below </h1>
                <h2>Location Details</h2>
                <p>Pincode: {{ user_pincode }}</p>
                <p>Place Name: {{ place_name }}</p>
                <p>District: {{ district }}</p>
                <p>State Name: {{ state_name }}</p>
                <h2>The Weather Information for your Pincode: {{ user_pincode }} is below </h2>
                <p>Temperature: {{ temperature }} °F</p>
                <p>Humidity: {{ humidity }} %</p>
                <h2>Optimized Crop information for your Pincode: {{ user_pincode }} is below</h2>
                <p>Predicted Crop: {{ predicted_crop }}</p>
                <p>The Average Yield per acre from this can be : {{ average_yield }} kgs/acre</p>
                <p>The Total Estimated Production for {{ land_size }} acres: {{ estimated_production }} kgs</p>
                <h2>Thank You for using the crop optimization system</h2>
                <p> If you have any feedback or comments, please do write to parashuramsu@gmail.com</p>
                <form action="/" method="get">
                    <button type="submit">Try Another Pincode</button>
                </form>
            ''', user_pincode=user_pincode, place_name=place_name, district=district, state_name=state_name,
            temperature=temperature, humidity=humidity, predicted_crop=predicted_crop[0],
            average_yield=average_yield, land_size=land_size, estimated_production=estimated_production)

        else:
            return jsonify({"error": "Pincode not found."}), 404
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)
