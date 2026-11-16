"""Tests for FastAPI endpoints: health, predict, batch predict, model info, and validation."""

import pytest
from ml.dataset import generate_synthetic_telco_data
from ml.train import train_and_evaluate_all_models


@pytest.fixture(autouse=True)
def ensure_champion_trained():
    """Ensure at least one champion model is registered before testing API endpoints."""
    from ml.registry import ModelRegistryManager
    reg = ModelRegistryManager()
    if reg.get_champion() is None:
        df = generate_synthetic_telco_data(num_samples=100)
        train_and_evaluate_all_models(df=df)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["champion_available"] is True


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "model_name" in data
    assert "metrics" in data


def test_model_metrics_endpoint(client):
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "val_roc_auc" in data["metrics"]


def test_predict_endpoint_valid(client):
    payload = {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 950.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST-001"
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "model_version" in data
    assert "latency_ms" in data


def test_predict_endpoint_invalid_inputs(client):
    # Invalid gender and negative tenure
    payload = {
        "gender": "InvalidGender",
        "SeniorCitizen": 5,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 950.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction_endpoint(client):
    payload = {
        "customers": [
            {
                "customerID": "BATCH-01",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 45.0,
                "TotalCharges": 90.0,
            },
            {
                "customerID": "BATCH-02",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 48,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 105.0,
                "TotalCharges": 5040.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["predictions"]) == 2
    assert 0.0 <= data["average_churn_probability"] <= 1.0
