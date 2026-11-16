"""Tests for delayed-label ingestion, model performance calculation, retraining lifecycle, and rollback."""

import pytest
from ml.registry import ModelRegistryManager
from backend.app.services.performance_service import performance_service
from backend.app.services.retraining_service import retraining_service


def test_delayed_label_ingestion_and_performance(client, db_session):
    # 1. Create a prediction
    pred_payload = {
        "customerID": "DELAYED-CUST-01",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
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
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
    }
    pred_res = client.post("/predict", json=pred_payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    pred_id = pred_data["prediction_id"]

    # 2. Ingest delayed label
    label_payload = {
        "labels": [
            {
                "customer_id": "DELAYED-CUST-01",
                "actual_churn": 1,
                "prediction_log_id": pred_id
            }
        ]
    }
    label_res = client.post("/labels", json=label_payload)
    assert label_res.status_code == 201
    label_data = label_res.json()
    assert label_data["total_submitted"] == 1
    assert label_data["matched_with_predictions"] == 1

    # 3. Query performance
    perf_res = client.get("/labels/performance")
    assert perf_res.status_code == 200
    perf_data = perf_res.json()
    assert perf_data["total_evaluated"] >= 1
    assert "accuracy" in perf_data
    assert "f1" in perf_data


def test_retraining_quality_gate_rejection():
    """Verify that a challenger with poor metrics is rejected by the quality gate."""
    reg = ModelRegistryManager()
    
    # Register mock weak model
    mock_metrics = {
        "val_roc_auc": 0.65,  # below minimum gate of 0.70
        "val_f1": 0.35,
    }
    reg.register_model(
        version="weak-challenger-v1",
        model_name="Weak_Model",
        pipeline_obj={"dummy": "model"},
        metrics=mock_metrics,
        params={},
        feature_importance=[],
    )

    passed, reason = reg.evaluate_challenger_promotion(
        challenger_version="weak-challenger-v1",
        primary_metric="val_roc_auc",
        min_improvement_threshold=0.005
    )
    assert passed is False
    assert "rejected" in reason.lower()


def test_retraining_and_rollback_lifecycle(client):
    reg = ModelRegistryManager()
    initial_champ = reg.get_champion()
    assert initial_champ is not None
    initial_ver = initial_champ["version"]

    # Trigger retraining through API endpoint
    retrain_res = client.post("/retrain/trigger", json={
        "trigger_reason": "Automated test retraining trigger",
        "min_improvement_threshold": 0.05  # high threshold so it will evaluate gate properly
    })
    assert retrain_res.status_code == 200
    data = retrain_res.json()
    assert data["status"] == "COMPLETED"
    assert data["decision"] in ["PROMOTED", "REJECTED"]

    # Test rollback endpoint
    rollback_res = client.post("/retrain/rollback")
    assert rollback_res.status_code == 200
    r_data = rollback_res.json()
    assert "success" in r_data
