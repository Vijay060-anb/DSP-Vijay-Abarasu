"""Model training module for house prices prediction"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from typing import Dict
import os


def compute_rmsle(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute Root Mean Squared Log Error.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        RMSLE score
    """
    # Ensure no negative values for log
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    return rmsle


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """
    Build and train the house prices prediction model.

    Args:
        data: Training dataframe with features and target

    Returns:
        Dictionary with model performance metrics
    """
    try:
        from house_prices.preprocess import preprocess_data

        # Check if required target column exists
        if 'SalePrice' not in data.columns:
            return {'error': 'SalePrice column not found in data'}

        # Split data
        X = data.drop('SalePrice', axis=1)
        y = data['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

        # Preprocess training data (fit transformers)
        print("Preprocessing training data...")
        X_train_processed = preprocess_data(X_train, fit=True)

        # Train model
        print("Training model...")
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train_processed, y_train)

        # Preprocess test data (use fitted transformers)
        print("Preprocessing test data...")
        X_test_processed = preprocess_data(X_test, fit=False)

        # Evaluate model
        y_pred = model.predict(X_test_processed)
        rmsle = compute_rmsle(y_test, y_pred)

        # Save model
        os.makedirs('../models', exist_ok=True)
        joblib.dump(model, '../models/model.joblib')
        print("Model saved to models/model.joblib")

        return {'rmsle': rmsle}

    except Exception as e:
        return {'error': str(e)}
