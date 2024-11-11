# Camera-Based Weight Prediction Application

## Overview
This project is a camera-based weight prediction application that uses computer vision to estimate the weight of a person in real-time. The application utilizes OpenCV for camera input, Mediapipe for body detection, and a Flask API to make predictions based on the captured data.

## Project Components
1. **Flask API**: A RESTful API built using Flask that handles weight prediction. It receives input (estimated weight in kilograms) and returns a prediction about whether the user's weight is classified as underweight, healthy, overweight, or obese.

2. **Camera Application**: The camera application, built using OpenCV, detects a person's body using Mediapipe and estimates their weight. The estimated weight is then sent to the Flask API for further classification.

## Problem Description
The main problem this project addresses is the real-time estimation and classification of an individual's weight. Using a non-invasive approach like a webcam, the application attempts to automatically determine whether a person falls into categories such as underweight, healthy weight, overweight, or obese based on body proportions detected by Mediapipe.

## How It Works
- **Body Detection**: Mediapipe is used to detect key body landmarks from a live webcam feed.
- **Weight Estimation**: Based on the body landmarks detected (e.g., shoulder to ankle length), the application estimates the user's weight. This is a simplified calculation used for demonstration purposes.
- **Classification via Flask API**: The estimated weight is sent to the Flask API, which processes the data and returns a prediction of weight classification (e.g., healthy, obese).

## Installation and Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure the following are installed:
   - Flask
   - OpenCV (`opencv-python`)
   - Mediapipe
   - Requests
   - Numpy

4. **Run the Flask API**:
   ```bash
   python flask_prediction_component.py
   ```
   This starts the Flask server on `http://127.0.0.1:5001`.

5. **Run the Camera Application**:
   ```bash
   python camera_open_predict.py
   ```
   This will open the webcam to begin detecting and predicting the user's weight in real-time.

## Usage
- When the webcam application is running, it will show a live feed.
- The user can press **'p'** to start the weight estimation process.
- The weight estimate is displayed on the video feed, and the classification result is printed in the console (e.g., Underweight, Healthy Weight, Overweight, Obese).
- Press **'q'** to quit the application.

## Issues with Flask API
If you encounter issues with the Flask API not working properly, here are some common problems and solutions:
1. **Connection Error**: Make sure the Flask server is running properly before launching the camera application. You can do this by checking that the Flask console output shows the server is running on `http://127.0.0.1:5001`.
2. **CORS Issues**: If you are testing from a web page or other tool, you might need to enable CORS. Install `flask-cors` using `pip install flask-cors` and add it to the Flask app:
   ```python
   from flask_cors import CORS
   CORS(app)
   ```
3. **Port Issues**: Ensure that the port (`5001`) is not being used by another application. You can change the port in the script if needed.
4. **File Not Found**: Make sure the trained model (`weight_model.pkl`) exists in the directory. Train and save the model before running the Flask API.

## Future Improvements
- **Improved Weight Estimation**: The current weight estimation is a rough approximation based on height. In future versions, a more accurate model that uses additional features could be trained to improve the precision of predictions.
- **User Interface**: Develop a more user-friendly interface to make it accessible for general users without technical knowledge.
- **Health Recommendations**: Provide personalized health recommendations based on the weight classification to help users achieve a healthy weight.

## License
This project is open-source and available under the MIT License.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

