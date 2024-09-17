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
        <h1>Welcome to Crop Optimization System</h1>
        <p>Please provide your Pincode and Land Size area to predict the optimized crop and estimation of the yield</p>
        <form method="post" action="/predict">
            <label for="pincode">Enter Pincode:</label>
            <input type="text" id="pincode" name="pincode" required>
            <label for="land_size">Enter Land Size (acres):</label>
            <input type="number" id="land_size" name="land_size" step="0.1" required>
            <button type="submit">Submit</button>
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
                <h1>Thank you for using the Crop Recommendation system! Results for your Pincode: {{ user_pincode }} are below:</h1>
                <h2>Location Details</h2>
                <p>Pincode: {{ user_pincode }}</p>
                <p>Place Name: {{ place_name }}</p>
                <p>District: {{ district }}</p>
                <p>State Name: {{ state_name }}</p>
                <h2>The Weather Information for your Pincode: {{ user_pincode }} is below:</h2>
                <p>Temperature: {{ temperature }} °F</p>
                <p>Humidity: {{ humidity }} %</p>
                <h2>Optimized Crop Information for your Pincode: {{ user_pincode }} is below:</h2>
                <p>Predicted Crop: {{ predicted_crop }}</p>
                <p>The Average Yield per acre from this can be: {{ average_yield }} kgs/acre</p>
                <p>The Total Estimated Production for {{ land_size }} acres: {{ estimated_production }} kgs</p>
                <h2>Thank You for using the crop optimization system!</h2>
                <form action="/" method="get">
                    <button type="submit">Try Another Pincode</button>
                </form>
            ''', user_pincode=user_pincode, place_name=place_name, district=district, state_name=state_name,
            temperature=temperature, humidity=humidity, predicted_crop=predicted_crop[0],
            average_yield=average_yield, land_size=land_size, estimated_production=estimated_production)

        else:
            error_message = (
                "Oops, sorry, the PINCODE you entered is incorrect or we currently do not have details for this PINCODE. "
                "If you would like to report this issue, please email us at parashuramsu@gmail.com. "
                "Please try again with another PINCODE."
            )
            return render_template_string('''
                <h1>Error</h1>
                <p>{{ error_message }}</p>
                <form action="/" method="get">
                    <button type="submit">Try Another Pincode</button>
                </form>
            ''', error_message=error_message)
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)
