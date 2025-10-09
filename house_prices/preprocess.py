"""Data preprocessing module for house prices prediction"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import List, Tuple
import joblib
import os


# Feature definitions
continuous_features = ['GrLivArea', 'OverallQual']
categorical_features = ['Neighborhood', 'HouseStyle']


def preprocess_data(data: pd.DataFrame, fit: bool = False) -> np.ndarray:
    """
    Preprocess the house prices data.

    Args:
        data: Input dataframe
        fit: Whether to fit the transformers

    Returns:
        Preprocessed numpy array
    """
    # Use absolute path to models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    # Select only the features we need
    data_subset = data[continuous_features + categorical_features].copy()

    # Handle continuous features
    cont_data = data_subset[continuous_features]

    if fit:
        cont_imputer = SimpleImputer(strategy='median')
        cont_data_imputed = cont_imputer.fit_transform(cont_data)
        scaler = StandardScaler()
        cont_data_scaled = scaler.fit_transform(cont_data_imputed)

        # Save transformers
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(cont_imputer,
                    os.path.join(models_dir, 'cont_imputer.joblib'))
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    else:
        cont_imputer = joblib.load(
            os.path.join(models_dir, 'cont_imputer.joblib'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        cont_data_imputed = cont_imputer.transform(cont_data)
        cont_data_scaled = scaler.transform(cont_data_imputed)

    # Handle categorical features
    cat_data = data_subset[categorical_features]

    if fit:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        cat_data_imputed = cat_imputer.fit_transform(cat_data)
        ohe_encoder = OneHotEncoder(handle_unknown='ignore',
                                    sparse_output=False)
        cat_data_encoded = ohe_encoder.fit_transform(cat_data_imputed)

        joblib.dump(cat_imputer,
                    os.path.join(models_dir, 'cat_imputer.joblib'))
        joblib.dump(ohe_encoder,
                    os.path.join(models_dir, 'ohe_encoder.joblib'))
    else:
        cat_imputer = joblib.load(
            os.path.join(models_dir, 'cat_imputer.joblib'))
        ohe_encoder = joblib.load(
            os.path.join(models_dir, 'ohe_encoder.joblib'))
        cat_data_imputed = cat_imputer.transform(cat_data)
        cat_data_encoded = ohe_encoder.transform(cat_data_imputed)

    # Combine features
    processed_data = np.hstack([cont_data_scaled, cat_data_encoded])
    return processed_data


def get_feature_names() -> Tuple[List[str], List[str]]:
    """
    Get the feature names used in preprocessing.

    Returns:
        Tuple of (continuous_features, categorical_features)
    """
    return continuous_features, categorical_features
