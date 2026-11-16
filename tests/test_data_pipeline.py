"""Tests for data ingestion, schema validation, imputation, and feature engineering."""

import pytest
import numpy as np
import pandas as pd
from ml.dataset import generate_synthetic_telco_data, generate_drifted_data
from ml.validation import validate_raw_dataset, clean_and_impute_dataset
from ml.pipeline import (
    ChurnFeatureEngineer,
    build_preprocessor,
    split_dataset,
    compute_class_weights,
)


def test_synthetic_data_generation():
    df = generate_synthetic_telco_data(num_samples=100)
    assert len(df) == 100
    assert "Churn" in df.columns
    assert "customerID" in df.columns
    assert "MonthlyCharges" in df.columns
    assert "TotalCharges" in df.columns


def test_data_validation_valid():
    df = generate_synthetic_telco_data(num_samples=100)
    report = validate_raw_dataset(df)
    assert report["is_valid"] is True
    assert report["row_count"] == 100
    assert "outliers" in report


def test_data_validation_missing_columns():
    df = pd.DataFrame({"gender": ["Male"], "Churn": ["No"]})
    report = validate_raw_dataset(df)
    assert report["is_valid"] is False
    assert len(report["errors"]) > 0


def test_clean_and_impute_dataset():
    df = generate_synthetic_telco_data(num_samples=50)
    # Inject empty space in TotalCharges
    df.loc[0, "TotalCharges"] = " "
    cleaned = clean_and_impute_dataset(df)
    assert not cleaned["TotalCharges"].isnull().any()
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_feature_engineer_transformer():
    df = generate_synthetic_telco_data(num_samples=30)
    cleaned = clean_and_impute_dataset(df)
    fe = ChurnFeatureEngineer()
    fe.fit(cleaned)
    transformed = fe.transform(cleaned)

    assert "tenure_years" in transformed.columns
    assert "monthly_to_total_ratio" in transformed.columns
    assert "total_services_subscribed" in transformed.columns
    assert "has_tech_or_security" in transformed.columns
    assert "is_high_risk_combo" in transformed.columns


def test_leak_free_split():
    df = generate_synthetic_telco_data(num_samples=100)
    cleaned = clean_and_impute_dataset(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(cleaned, test_size=0.15, val_size=0.15)

    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)
    assert set(y_train.unique()).issubset({0, 1})


def test_class_weights_computation():
    y = pd.Series([0, 0, 0, 1])
    weights = compute_class_weights(y)
    assert weights[1] > weights[0]
