"""Feature engineering, preprocessing pipeline, and data splitting for DriftGuard."""

import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

RAW_NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

ENGINEERED_NUMERICAL_FEATURES = [
    "tenure_years",
    "monthly_to_total_ratio",
    "total_services_subscribed",
]
ENGINEERED_CATEGORICAL_FEATURES = [
    "has_tech_or_security",
    "is_high_risk_combo",  # Month-to-month + Fiber optic
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for customer churn domain feature engineering."""

    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Tenure in years
        df["tenure_years"] = df["tenure"] / 12.0

        # 2. Ratio of monthly charges to total charges
        safe_total = np.where(df["TotalCharges"] <= 0, df["MonthlyCharges"], df["TotalCharges"])
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (safe_total + 1e-5)

        # 3. Total active services count
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        active_services = np.zeros(len(df), dtype=int)
        for col in service_cols:
            if col in df.columns:
                active_services += (df[col] == "Yes").astype(int)
        if "MultipleLines" in df.columns:
            active_services += (df["MultipleLines"] == "Yes").astype(int)
        if "PhoneService" in df.columns:
            active_services += (df["PhoneService"] == "Yes").astype(int)
        df["total_services_subscribed"] = active_services

        # 4. Tech support or Online security presence
        has_tech = (df.get("TechSupport", "No") == "Yes") | (df.get("OnlineSecurity", "No") == "Yes")
        df["has_tech_or_security"] = np.where(has_tech, "Yes", "No")

        # 5. High risk combination: Month-to-month contract + Fiber optic internet
        is_m2m = df.get("Contract", "") == "Month-to-month"
        is_fiber = df.get("InternetService", "") == "Fiber optic"
        df["is_high_risk_combo"] = np.where(is_m2m & is_fiber, "Yes", "No")

        # Convert SeniorCitizen to string category if needed
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

        return df


def build_preprocessor() -> ColumnTransformer:
    """Create deterministic ColumnTransformer with standard scaling and one-hot encoding."""
    all_num_cols = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
    all_cat_cols = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_num_cols),
            ("cat", categorical_transformer, all_cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Perform leak-free stratified split into Train, Validation, and Test sets."""
    clean_df = df.copy()
    
    # Standardize target to binary 0/1
    if clean_df[target_col].dtype == object or isinstance(clean_df[target_col].iloc[0], str):
        y = clean_df[target_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    else:
        y = clean_df[target_col].astype(int)

    feature_cols = [c for c in clean_df.columns if c not in [target_col, "customerID"]]
    X = clean_df[feature_cols]

    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Then split train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=adjusted_val_size,
        stratify=y_train_val,
        random_state=random_state
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def compute_class_weights(y_train: pd.Series) -> Dict[int, float]:
    """Compute balanced class weights to address churn class imbalance."""
    counts = y_train.value_counts()
    n_samples = len(y_train)
    n_classes = len(counts)
    weights = {cls: float(n_samples / (n_classes * count)) for cls, count in counts.items()}
    return weights
