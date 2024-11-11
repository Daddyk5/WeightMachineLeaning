import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Convert weights from pounds to kilograms
def convert_to_kg(weight_in_pounds):
    return weight_in_pounds * 0.453592

# Train the machine learning model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    return predictions

# Main function
def main():
    # Load the dataset
    file_path = "weights.csv"  # Update with your file path
    data = load_data(file_path)

    # Convert input weights to kilograms
    data['input'] = data['input'].apply(convert_to_kg)

    # Split the dataset into training and testing sets
    X = data[['input']].values
    y = data['output'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    predictions = evaluate_model(model, X_test, y_test)

    # Plot predictions vs actual values
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Weights")
    plt.show()

    # Save the trained model
    joblib.dump(model, "weight_model.pkl")
    print("Model saved as weight_model.pkl")

# Run the main function
if __name__ == "__main__":
    main()
