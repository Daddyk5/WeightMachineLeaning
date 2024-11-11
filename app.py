from flask import Flask, request, jsonify
import numpy as np
import joblib
import random
import requests
import json
import threading
import time
import os
import signal

# Flask App Setup
app = Flask(__name__)

# Load the trained model
model = None
try:
    model = joblib.load("weight_model.pkl")
except FileNotFoundError:
    model = None
    print("Model file 'weight_model.pkl' not found. Train and save the model first.")

# Flask API Endpoints
@app.route('/', methods=['GET'])
def home():
    """
    Default route for the API.
    """
    return """
    <html>
        <head><title>Weight Prediction API</title></head>
        <body>
            <h1>Welcome to the Weight Prediction API</h1>
            <p>Use the <strong>/predict</strong> endpoint with a POST request to make predictions.</p>
        </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict weight in kilograms.
    """
    if model is None:
        return jsonify({"error": "Model not found. Please train and save the model first."}), 500

    print("Request received at /predict")

    # Check if the request contains JSON
    if not request.is_json:
        return jsonify({"error": "Invalid input. Please send a JSON payload."}), 400

    # Parse JSON data
    data = request.get_json()
    print(f"Received data: {data}")

    if "weight_in_pounds" not in data:
        return jsonify({"error": "Missing 'weight_in_pounds' key in JSON payload."}), 400

    try:
        # Convert weight from pounds to kg and make prediction
        weight_in_pounds = float(data["weight_in_pounds"])
        weight_in_kg = weight_in_pounds * 0.453592
        prediction = model.predict(np.array([[weight_in_kg]]))
        return jsonify({"predicted_weight": prediction[0]})
    except ValueError:
        return jsonify({"error": "Invalid weight value. Please provide a numeric value."}), 400

@app.errorhandler(404)
def page_not_found(error):
    """
    Handle invalid routes with a custom message.
    """
    return jsonify({"error": "The requested resource was not found. Please check the URL and try again."}), 404

# Function to send a POST request to the Flask API for testing
def test_post_request():
    # Pre-request handling: Generate weight in pounds between 100 and 200
    weight_in_pounds = random.uniform(100, 200)  # Random weight

    # URL of the Flask API
    url = 'http://127.0.0.1:5001/predict'

    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    # Data to send
    data = {
        'weight_in_pounds': weight_in_pounds
    }

    # Log the pre-request data (for debugging)
    print(f"Sending request to {url} with payload: {data}")

    # Send POST request
    success = False
    for _ in range(5):  # Retry up to 5 times
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))

            # Post-response handling
            if response.status_code == 200:
                response_data = response.json()
                # Post-response handling: Log the successful prediction
                print("Status: Success")
                print(f"Predicted Weight: {response_data['predicted_weight']}")
                success = True
                break
            else:
                # Log the error
                print(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            print("Error: Unable to connect to the Flask API. Retrying...")
            time.sleep(1)  # Wait before retrying

    if not success:
        print("Error: Unable to connect to the Flask API after multiple attempts.")

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    def run_flask_app():
        app.run(port=5001, use_reloader=False, debug=False)

    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Test the POST request after a short delay to allow the server to start
    time.sleep(5)  # Increase wait time for the server to be up and running

    # Run the test request
    test_post_request()

    # Terminate the Flask server
    os.kill(os.getpid(), signal.SIGTERM)
