from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the datasets from the cloned GitHub repository
pin_data = pd.read_csv('PIN.csv', encoding='latin1')
apc_data = pd.read_csv('APC.csv', encoding='latin1')

# Load the model
model = joblib.load('model.pkl')  # Ensure this path is correct

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
    return '''
        <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,500" rel="stylesheet" crossorigin="anonymous">
        <div class="header">
            <h1>Welcome to the Crop Optimization System</h1>
            <div class="header-images">
                <img src="https://static.wixstatic.com/media/7d5a65_9525c98a8a2c4c7e9193e78a3039d6bd~mv2.png/v1/fill/w_398,h_186,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Deenabandh_edited_edited.png" alt="Deenabandhu Logo" class="header-image left">
                <img src="https://race.reva.edu.in/wp-content/uploads/2020/11/Reva-logo-1-1.png" alt="Reva University Logo" class="header-image right">
            </div>
        </div>
        <div class="container">
            <div class="input-section">
                <div>
                    <label for="name_input">Your Name:</label>
                    <input type="text" id="name_input" name="name_input" placeholder="Enter your name" required />
                </div>
                <div>
                    <label for="pincode_input">Your PIN Code:</label>
                    <input type="text" id="pincode_input" name="pincode" placeholder="Enter your PIN code" required />
                </div>
                <div>
                    <label for="land_size_input">Your Land Size (in acres):</label>
                    <input type="number" id="land_size_input" name="land_size" placeholder="Enter land size" required />
                </div>
                <button id="submit_button" type="submit">Get Optimized Crop</button>
            </div>
        </div>
        <div class="footer">
            <img src="https://static.wixstatic.com/media/7d5a65_9525c98a8a2c4c7e9193e78a3039d6bd~mv2.png/v1/fill/w_398,h_186,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Deenabandh_edited_edited.png" alt="Deenabandhu Logo" class="footer-image">
            <div class="footer-text">
                <p>All rights reserved (2024)</p>
                <p>For any grievances, please reach out to:</p>
                <p>Parashuram Hadimain, Email: <a href="mailto:parashuramsu@gmail.com">parashuramsu@gmail.com</a></p>
                <p>Deenabandhu Social Service Organization, Email: <a href="mailto:harsh@deenabandhu.org.in">harsh@deenabandhu.org.in</a></p>
            </div>
        </div>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_name = request.form['name_input']
        user_pincode = request.form['pincode']
        land_size = float(request.form['land_size'])

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
            ph = current_day.get('humidity', 0)  # Assuming pH is provided as humidity; adjust if necessary
            rainfall = current_day.get('precip', 0)  # Default to 0 if not found

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
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Crop Recommendation</title>
                    <link href="https://fonts.googleapis.com/css?family=Roboto:300,400,500" rel="stylesheet" crossorigin="anonymous">
                    <style>
                        body {
                            background-color: #d4edda; /* Light green background for the body */
                            font-family: 'Roboto', sans-serif;
                        }
                        .header {
                            text-align: center;
                            background-color: white; /* White header */
                            padding: 10px 0;
                        }
                        .header h1 {
                            color: #155724; /* Dark green title */
                            margin: 0;
                        }
                        .header-images {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin: 10px 0;
                        }
                        .header-image {
                            max-height: 50px; /* Optimize image size */
                            margin: 0 10px;
                        }
                        .container {
                            max-width: 600px;
                            margin: 20px auto;
                            padding: 20px;
                            border: 1px solid #ccc;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                            background-color: white; /* White background for the container */
                        }
                        .output-section {
                            margin-top: 20px;
                            padding: 10px;
                            border: 1px solid #007bff;
                            border-radius: 4px;
                            background-color: #f8f9fa; /* Light background for output */
                        }
                        button {
                            padding: 10px 15px;
                            background-color: #28a745;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        }
                        button:hover {
                            background-color: #218838;
                        }
                        .footer {
                            display: flex;
                            align-items: center;
                            justify-content: space-between; /* Align footer content to the left and right */
                            margin-top: 20px;
                            padding: 10px;
                            background-color: #f1f1f1;
                            border-top: 1px solid #ccc;
                        }
                        .footer-image {
                            max-height: 100px; /* Make the Deenabandhu logo bigger */
                            margin-right: 20px; /* Space between logo and text */
                        }
                        .footer-text {
                            text-align: left; /* Align footer text to the left */
                        }
                        .footer p {
                            margin: 5px 0;
                            font-size: 14px;
                        }
                        .footer a {
                            color: #007bff;
                            text-decoration: none;
                        }
                        .footer a:hover {
                            text-decoration: underline;
                        }
                    </style>
                </head>
                <body>
                    <h1>Your weather for the pincode {{ user_pincode }}:</h1>
                    <p>Place Name: {{ place_name }}</p>
                    <p>District: {{ district }}</p>
                    <p>State Name: {{ state_name }}</p>
                    <p>Temperature: {{ temperature }} °F</p>
                    <p>Humidity: {{ humidity }} %</p>
                    <p>pH: {{ ph }}</p>
                    <p>Rainfall: {{ rainfall }} inches</p>

                    <h2>Top Recommended crops:</h2>
                    <p>{{ predicted_crop }}</p>
                    <p>Best season for the crop {{ predicted_crop }}: Kharif</p>
                    <p>Estimated per acre yield for {{ predicted_crop }}: {{ average_yield }} kgs/acre</p>
                    <p>For your land size of {{ land_size }} acres, the estimated production for {{ predicted_crop }} is {{ estimated_production }} kgs.</p>

                    <p>Thank you, {{ user_name }}!</p>
                    <form action="/" method="get">
                        <button type="submit">Try Another Pincode</button>
                    </form>
                    <div class="footer">
                        <img src="https://static.wixstatic.com/media/7d5a65_9525c98a8a2c4c7e9193e78a3039d6bd~mv2.png/v1/fill/w_398,h_186,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/Deenabandh_edited_edited.png" alt="Deenabandhu Logo" class="footer-image">
                        <div class="footer-text">
                            <p>All rights reserved (2024)</p>
                            <p>For any grievances, please reach out to:</p>
                            <p>Parashuram Hadimain, Email: <a href="mailto:parashuramsu@gmail.com">parashuramsu@gmail.com</a></p>
                            <p>Deenabandhu Social Service Organization, Email: <a href="mailto:harsh@deenabandhu.org.in">harsh@deenabandhu.org.in</a></p>
                        </div>
                    </div>
                </body>
                </html>
            ''', user_name=user_name, user_pincode=user_pincode, place_name=place_name, district=district, state_name=state_name,
            temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall,
            predicted_crop=predicted_crop[0], average_yield=average_yield, land_size=land_size, estimated_production=estimated_production)

        else:
            return jsonify({"error": "Pincode not found."}), 404
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)
