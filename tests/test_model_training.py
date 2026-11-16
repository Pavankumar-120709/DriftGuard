"""Tests for model training, evaluation metrics, and feature importance."""

import pytest
import numpy as np
import pandas as pd
from ml.dataset import generate_synthetic_telco_data
from ml.evaluate import evaluate_classifier, get_curve_points
from ml.explain import compute_feature_importance
from ml.train import train_and_evaluate_all_models, compute_dataset_fingerprint
from ml.registry import ModelRegistryManager


def test_evaluate_classifier():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])

    metrics = evaluate_classifier(y_true, y_pred, y_proba, prefix="val")
    assert "val_accuracy" in metrics
    assert "val_precision" in metrics
    assert "val_recall" in metrics
    assert "val_f1" in metrics
    assert "val_roc_auc" in metrics
    assert "val_true_positives" in metrics
    assert metrics["val_true_positives"] == 2


def test_curve_points_computation():
    y_true = np.array([0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])
    curves = get_curve_points(y_true, y_proba, num_samples=10)
    assert "roc_curve" in curves
    assert "pr_curve" in curves
    assert len(curves["roc_curve"]) > 0


def test_compute_dataset_fingerprint():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df3 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

    fp1 = compute_dataset_fingerprint(df1)
    fp2 = compute_dataset_fingerprint(df2)
    fp3 = compute_dataset_fingerprint(df3)

    assert fp1 == fp2
    assert fp1 != fp3


def test_train_and_evaluate_models_run(tmp_path):
    df = generate_synthetic_telco_data(num_samples=120)
    reg = ModelRegistryManager(storage_dir=tmp_path / "models")
    
    # Train candidate models
    result = train_and_evaluate_all_models(df=df)
    assert result["status"] == "COMPLETED"
    assert "champion_selected" in result
    assert len(result["all_candidates"]) >= 2
    assert "champion_metrics" in result
    assert result["champion_metrics"]["val_roc_auc"] > 0.50
