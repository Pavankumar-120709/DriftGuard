"""Data validation, schema checking, and outlier analysis for DriftGuard."""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "tenure": (0, 72),  # min, max
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "MonthlyCharges": (18.0, 150.0),
    "TotalCharges": (0.0, 10000.0),
}

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]
TARGET_COL = "Churn"


def validate_raw_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform data validation, duplicate checks, missing value checks, and outlier analysis."""
    report = {
        "is_valid": True,
        "row_count": len(df),
        "errors": [],
        "warnings": [],
        "missing_values": {},
        "duplicate_count": 0,
        "outliers": {},
    }

    # 1. Check duplicate rows
    if "customerID" in df.columns:
        dupes = df.duplicated(subset=["customerID"]).sum()
    else:
        dupes = df.duplicated().sum()
    report["duplicate_count"] = int(dupes)
    if dupes > 0:
        report["warnings"].append(f"Found {dupes} duplicate customer records.")

    # 2. Check required columns
    required_cols = list(EXPECTED_SCHEMA.keys())
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["errors"].append(f"Missing required columns: {missing_cols}")
        return report

    # 3. Check and clean TotalCharges (handle blank spaces/strings)
    df_clean = df.copy()
    if df_clean["TotalCharges"].dtype == object:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )

    # 4. Missing value checks
    missing_series = df_clean[required_cols].isnull().sum()
    for col, count in missing_series.items():
        if count > 0:
            report["missing_values"][col] = int(count)
            # If missing values exceed 10%, flag as error, otherwise warning
            if count / len(df) > 0.10:
                report["is_valid"] = False
                report["errors"].append(f"Column '{col}' has {count} ({count/len(df):.1%}) missing values.")
            else:
                report["warnings"].append(f"Column '{col}' has {count} missing values ({count/len(df):.1%}).")

    # 5. Schema and Categorical Domain checks
    for col, expected in EXPECTED_SCHEMA.items():
        if isinstance(expected, list):
            # Categorical check
            unique_vals = df_clean[col].dropna().unique().tolist()
            invalid_vals = [v for v in unique_vals if v not in expected]
            if invalid_vals:
                report["warnings"].append(
                    f"Column '{col}' contains unexpected categories: {invalid_vals}"
                )
        elif isinstance(expected, tuple):
            # Numerical bounds & outlier check using IQR
            min_val, max_val = expected
            vals = df_clean[col].dropna()
            out_of_bounds = vals[(vals < min_val) | (vals > max_val)]
            if len(out_of_bounds) > 0:
                report["warnings"].append(
                    f"Column '{col}' has {len(out_of_bounds)} values outside reasonable bounds ({min_val}, {max_val})."
                )

            # IQR Outlier detection
            q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q75 - q25
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr
            iqr_outliers = vals[(vals < lower_fence) | (vals > upper_fence)]
            report["outliers"][col] = {
                "count": int(len(iqr_outliers)),
                "lower_fence": float(lower_fence),
                "upper_fence": float(upper_fence),
            }

    return report


def clean_and_impute_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, types, and remove duplicates deterministically."""
    cleaned = df.copy()

    # Drop customerID duplicates if present
    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["customerID"]).reset_index(drop=True)

    # Convert TotalCharges to float
    if "TotalCharges" in cleaned.columns and cleaned["TotalCharges"].dtype == object:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )

    # Impute missing TotalCharges with MonthlyCharges * tenure or median
    if "TotalCharges" in cleaned.columns:
        missing_mask = cleaned["TotalCharges"].isnull()
        if missing_mask.any():
            fallback_impute = cleaned.loc[missing_mask, "MonthlyCharges"] * cleaned.loc[missing_mask, "tenure"].clip(lower=1)
            cleaned.loc[missing_mask, "TotalCharges"] = fallback_impute.fillna(cleaned["TotalCharges"].median())

    # Impute any remaining numeric columns with median
    for col in NUMERICAL_COLS:
        if col in cleaned.columns and cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Impute categorical columns with mode
    for col in CATEGORICAL_COLS:
        if col in cleaned.columns and cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    return cleaned
