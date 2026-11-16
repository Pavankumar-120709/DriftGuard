"""Tests for Evidently AI data drift detection, feature metrics, and HTML report generation."""

import pytest
from backend.app.services.drift_service import drift_service
from ml.dataset import generate_synthetic_telco_data, generate_drifted_data


def test_drift_analysis_with_production_logs(client, db_session):
    # 1. Send predictions to populate production logs
    ref_df = generate_synthetic_telco_data(num_samples=50)
    drifted_df = generate_drifted_data(ref_df, scenario="gradual_fiber_optic_shift", num_samples=30)

    # Convert rows to prediction payloads
    for _, row in drifted_df.iterrows():
        payload = {
            "customerID": str(row["customerID"]),
            "gender": str(row["gender"]),
            "SeniorCitizen": int(row["SeniorCitizen"]),
            "Partner": str(row["Partner"]),
            "Dependents": str(row["Dependents"]),
            "tenure": int(row["tenure"]),
            "PhoneService": str(row["PhoneService"]),
            "MultipleLines": str(row["MultipleLines"]),
            "InternetService": str(row["InternetService"]),
            "OnlineSecurity": str(row["OnlineSecurity"]),
            "OnlineBackup": str(row["OnlineBackup"]),
            "DeviceProtection": str(row["DeviceProtection"]),
            "TechSupport": str(row["TechSupport"]),
            "StreamingTV": str(row["StreamingTV"]),
            "StreamingMovies": str(row["StreamingMovies"]),
            "Contract": str(row["Contract"]),
            "PaperlessBilling": str(row["PaperlessBilling"]),
            "PaymentMethod": str(row["PaymentMethod"]),
            "MonthlyCharges": float(row["MonthlyCharges"]),
            "TotalCharges": float(row["TotalCharges"]) if str(row["TotalCharges"]).strip() else 0.0,
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200

    # 2. Trigger drift analysis endpoint
    resp = client.post("/monitoring/run", json={"sample_limit": 50, "p_value_threshold": 0.05, "drift_share_threshold": 0.25})
    assert resp.status_code == 200
    data = resp.json()

    assert "id" in data
    assert "dataset_drift_detected" in data
    assert "drift_share" in data
    assert len(data["features"]) > 0
    assert any(f["feature_name"] == "MonthlyCharges" for f in data["features"])

    # 3. Check status endpoint
    status_resp = client.get("/monitoring/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["total_prediction_records"] >= 30
    assert "active_dataset_drift" in status_data
