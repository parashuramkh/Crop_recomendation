@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Print the form data received
        print("Form data received:", request.form)

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

            # Prepare output message
            return render_template_string('''
                <h1>Crop Recommendation</h1>
                <h2>Location Details</h2>
                <p>Pincode: {{ user_pincode }}</p>
                <p>Place Name: {{ place_name }}</p>
                <p>District: {{ district }}</p>
                <p>State Name: {{ state_name }}</p>
                <h2>Weather Information</h2>
                <p>Temperature: {{ temperature }} °F</p>
                <p>Humidity: {{ humidity }} %</p>
                <h2>Recommended Crop</h2>
                <p>Predicted Crop: {{ predicted_crop }}</p>
                <p>Average Yield: {{ average_yield }} kgs/acre</p>
                <p>Estimated Production for {{ land_size }} acres: {{ estimated_production }} kgs</p>
            ''', user_pincode=user_pincode, place_name=place_name, district=district, state_name=state_name,
            temperature=temperature, humidity=humidity, predicted_crop=predicted_crop[0],
            average_yield=average_yield, land_size=land_size, estimated_production=estimated_production)

        else:
            return jsonify({"error": "Pincode not found."}), 404
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Internal server error."}), 500
