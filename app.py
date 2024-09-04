from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from fuzzywuzzy import process

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('model.pkl')

# Define known icons
icons = {
    "Airport shuttle": "fa-shuttle-van",
    "Bar": "fa-cocktail",
    "Beachfront": "fa-umbrella-beach",
    "Non-smoking rooms": "fa-smoking-ban",
    "Private beach area": "fa-umbrella-beach",
    "Garden": "fa-tree",
    "Outdoor furniture": "fa-chair",
    "Picnic area": "fa-picnic-table",
    "Terrace": "fa-table-tennis",
    "Badminton equipment": "fa-basketball-ball",
    "Beach": "fa-umbrella-beach",
    "Canoeing": "fa-water",
    "Diving": "fa-diving-mask",
    "Evening entertainment": "fa-theater-masks",
    "Happy hour": "fa-hourglass-start",
    "Live music/performance": "fa-microphone",
    "Movie nights": "fa-film",
    "Tennis court": "fa-table-tennis",
    "Tennis equipment": "fa-tennis-ball",
    "Windsurfing": "fa-wind",
    "Telephone": "fa-phone",
    "Kid-friendly buffet": "fa-utensils",
    "Restaurant": "fa-utensils",
    "Snack bar": "fa-burger",
    "Wine/champagne": "fa-wine-glass",
    "24-hour front desk": "fa-clock",
    "Currency exchange": "fa-exchange-alt",
    "Daily housekeeping": "fa-broom",
    "Ironing service": "fa-iron",
    "Laundry": "fa-tshirt",
    "Tour desk": "fa-map-signs",
    "24-hour security": "fa-shield-alt",
    "CCTV outside property": "fa-camera",
    "Fire extinguishers": "fa-fire-extinguisher",
    "Key access": "fa-key",
    "Safe": "fa-lock",
    "Security alarm": "fa-bell",
    "Smoke alarms": "fa-smoke",
    "Air conditioning": "fa-snowflake",
    "Designated smoking area": "fa-smoking",
    "Family rooms": "fa-users",
    "Outdoor swimming pool": "fa-swimming-pool",
    "Pool with view": "fa-eye",
    "Pool/beach towels": "fa-towel",
    "Fitness": "fa-dumbbell",
    "Foot bath": "fa-foot",
    "Foot massage": "fa-hands",
    "Massage": "fa-hands",
    "Open-air bath": "fa-bath",
    "Spa": "fa-spa",
    "Spa facilities": "fa-spa",
    "Spa lounge/relaxation area": "fa-couch",
    "Spa/wellness packages": "fa-box",
    "Airport": "fa-plane-departure",
    "Cafes/Bars": "fa-cocktail",
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Find the closest match using fuzzy matching
        closest_match, _ = process.extractOne(text, icons.keys())
        
        if closest_match:
            icon_class = icons.get(closest_match, 'fa-question')
        else:
            # If no close match found, use the model to predict
            predicted_icon = model.predict([text])[0]
            icon_class = icons.get(predicted_icon, 'fa-question')

        return jsonify({'icon_class': icon_class})

    except Exception as e:
        # Print the exception to the server logs
        print(f"Error: {e}")
        # Return a 500 error with the exception message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)