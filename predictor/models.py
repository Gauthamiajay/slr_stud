import pandas as pd
import numpy as np
import joblib  # For saving the model
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Function to train and save model
def train_model():
    try:
        # Load dataset
        csv_path = os.path.join(os.path.dirname(__file__), 'study_hours.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # Handle missing values
        df = df.dropna()

        # Split dataset
        X = df[['Hours']]
        y = df['Score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Model evaluation
        accuracy = r2_score(y_test, y_pred) * 100
        error = mean_absolute_error(y_test, y_pred)

        # Save the model
        model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')
        joblib.dump(model, model_path)

        print(f"Model trained successfully! Accuracy: {accuracy:.2f}%")
        return model, accuracy, error

    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None


# Function to load the model and make predictions
def predict_score(hours):
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pkl')

        # Train if model is missing
        if not os.path.exists(model_path):
            print("Model not found, training now...")
            train_model()

        # Load the model
        model = joblib.load(model_path)

        # Validate input
        if not isinstance(hours, (int, float, list, np.ndarray)):
            raise ValueError("Invalid input: 'hours' must be a number or a list of numbers")

        hours_array = np.array(hours).reshape(-1, 1)
        predicted_score = model.predict(hours_array)

        return predicted_score

    except Exception as e:
        print(f"Error predicting score: {e}")
        return None
