from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Font Awesome icon mappings (if needed for extra functionality)
icon_mappings = {
    # Add your mappings here, if different icons are needed
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # Use the model to predict the icon
    icon = model.predict([text])[0]
    
    # Return Font Awesome class
    icon_class = icon_mappings.get(icon, 'fa-question')  # Default icon if not found
    
    return jsonify({'icon_class': icon_class})

if __name__ == '__main__':
    app.run(debug=True)
