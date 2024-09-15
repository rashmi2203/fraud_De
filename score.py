import joblib
import numpy as np

# Load the saved model
model = joblib.load("fraud_detection_model.pkl")

# Function to make predictions
def predict(data):
    # Assuming input data is a list of feature values
    data = np.array(data)
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Example: Predicting for new data
    test_data = [[0.5, 1.2, -0.3, 0.8, -0.9, 1.1, -1.2, 1.3, -0.8, 0.6]]  # Example input
    result = predict(test_data)
    print("Prediction:", result)
