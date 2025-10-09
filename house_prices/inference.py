"""Model inference module for house prices prediction"""
import pandas as pd
import numpy as np
import joblib


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the trained model.

    Args:
        input_data: Input dataframe for prediction

    Returns:
        Numpy array with predictions
    """
    try:
        from house_prices.preprocess import preprocess_data

        # Load model
        model = joblib.load('../models/model.joblib')

        # Preprocess data using fitted transformers
        processed_data = preprocess_data(input_data, fit=False)

        # Make predictions
        predictions = model.predict(processed_data)
        return predictions

    except Exception as e:
        print(f"Prediction error: {e}")
        return np.array([])
