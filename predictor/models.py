import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Function to train model
def train_model():
    # Load dataset
    df = pd.read_csv('predictor/study_hours.csv')
    
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

    return model, accuracy, error


# Create your models here.
