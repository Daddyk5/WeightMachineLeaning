import cv2
import requests
import json
import numpy as np
import mediapipe as mp

# Setup Mediapipe for body detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to send a POST request to the Flask API for prediction
def send_post_request(weight_in_kg):
    # URL of the Flask API
    url = 'http://127.0.0.1:5001/predict'

    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    # Data to send
    data = {
        'weight_in_kg': weight_in_kg
    }

    # Log the pre-request data (for debugging)
    print(f"Sending request to {url} with payload: {data}")

    # Send POST request
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Post-response handling
        if response.status_code == 200:
            response_data = response.json()
            # Post-response handling: Log the successful prediction
            print("Status: Success")
            print(f"Predicted Weight: {response_data['predicted_weight']}")
            if response_data['predicted_weight'] < 18.5:
                print("Classification: Underweight")
            elif 18.5 <= response_data['predicted_weight'] < 24.9:
                print("Classification: Healthy Weight")
            elif 25 <= response_data['predicted_weight'] < 29.9:
                print("Classification: Overweight")
            else:
                print("Classification: Obese")
        else:
            # Log the error
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the Flask API. Ensure the server is running.")

# Open the webcam using OpenCV
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    # If landmarks are detected, calculate an estimated weight
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        # Estimate weight based on detected body landmarks (for demonstration purposes)
        # Here we use a simple approach based on detected body size proportions
        height_estimate = np.linalg.norm(np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]) -
                                         np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]))
        weight_estimate_kg = height_estimate * 100  # Simplified weight estimation

        # Display the weight estimate on the frame
        cv2.putText(frame, f"Estimated Weight: {weight_estimate_kg:.2f} kg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Send the estimated weight to the Flask API for further prediction/classification
        send_post_request(weight_estimate_kg)

    # Display the resulting frame
    cv2.imshow('Webcam - Press "q" to Quit', frame)

    # Wait for user action
    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, break the loop and quit
    if key == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
