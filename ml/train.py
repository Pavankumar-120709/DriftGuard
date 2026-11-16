"""Complete model training pipeline with MLflow experiment tracking for DriftGuard."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml.dataset import ingest_telco_data, PROCESSED_DATA_PATH
from ml.evaluate import evaluate_classifier, get_curve_points
from ml.explain import compute_feature_importance
from ml.pipeline import (
    split_dataset,
    build_preprocessor,
    ChurnFeatureEngineer,
    compute_class_weights,
)
from ml.registry import ModelRegistryManager
from ml.validation import validate_raw_dataset, clean_and_impute_dataset

logger = logging.getLogger(__name__)


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Create a deterministic SHA-256 fingerprint of the dataset content and shape."""
    summary_str = f"shape={df.shape}_cols={list(df.columns)}_head={df.head(5).to_dict()}"
    return hashlib.sha256(summary_str.encode("utf-8")).hexdigest()[:16]


def train_and_evaluate_all_models(
    df: Optional[pd.DataFrame] = None,
    experiment_name: str = "customer-churn-prediction",
    mlflow_tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Train Logistic Regression, Random Forest, and XGBoost models, log to MLflow, and select champion."""
    start_time = time.time()

    # 1. Ingest & Validate
    if df is None:
        raw_df = ingest_telco_data()
    else:
        raw_df = df.copy()

    val_report = validate_raw_dataset(raw_df)
    if not val_report["is_valid"]:
        raise ValueError(f"Dataset validation failed: {val_report['errors']}")

    # 2. Clean & Preprocess
    clean_df = clean_and_impute_dataset(raw_df)
    dataset_fingerprint = compute_dataset_fingerprint(clean_df)

    # Save reference data for Evidently drift detection
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    # 3. Train/Val/Test Split (leak-free)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(clean_df)
    class_weights = compute_class_weights(y_train)
    pos_scale = float(class_weights[1] / class_weights[0])

    # 4. Setup MLflow
    use_mlflow = False
    try:
        import mlflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        use_mlflow = True
    except Exception as e:
        logger.warning(f"MLflow setup warning: {e}. Proceeding with local artifact tracking.")

    # 5. Define Candidate Model Architectures & Hyperparameters
    candidates = [
        {
            "name": "Logistic_Regression",
            "model": LogisticRegression(
                max_iter=1000,
                C=0.5,
                class_weight="balanced",
                random_state=42
            ),
            "params": {"max_iter": 1000, "C": 0.5, "class_weight": "balanced", "penalty": "l2"}
        },
        {
            "name": "Random_Forest",
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "params": {"n_estimators": 150, "max_depth": 8, "min_samples_split": 4, "class_weight": "balanced"}
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=pos_scale,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ),
            "params": {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "scale_pos_weight": round(pos_scale, 2),
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }
        }
    ]

    registry = ModelRegistryManager()
    run_results = []
    timestamp_str = int(time.time())

    for candidate in candidates:
        cand_start = time.time()
        c_name = candidate["name"]
        clf = candidate["model"]
        c_params = candidate["params"]

        # Build end-to-end scikit-learn pipeline (Feature engineering -> Preprocessing -> Classifier)
        pipeline = Pipeline([
            ("feature_engineer", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])

        # Train on training split
        pipeline.fit(X_train, y_train)

        # Predict on validation and test sets
        val_preds = pipeline.predict(X_val)
        val_probas = pipeline.predict_proba(X_val)[:, 1]
        test_preds = pipeline.predict(X_test)
        test_probas = pipeline.predict_proba(X_test)[:, 1]

        val_metrics = evaluate_classifier(y_val, val_preds, val_probas, prefix="val")
        test_metrics = evaluate_classifier(y_test, test_preds, test_probas, prefix="test")
        curves = get_curve_points(y_test, test_probas)

        combined_metrics = {**val_metrics, **test_metrics}
        training_duration = round(time.time() - cand_start, 2)
        combined_metrics["training_duration_seconds"] = training_duration

        # Get feature names and calculate feature importance
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_features = list(preprocessor.named_transformers_["cat"].get_feature_names_out())
        num_features = list(preprocessor.transformers[0][2])
        all_feature_names = num_features + cat_features

        feat_imp = compute_feature_importance(clf, all_feature_names)

        # MLflow Logging
        mlflow_run_id = None
        if use_mlflow:
            import mlflow
            try:
                with mlflow.start_run(run_name=f"{c_name}_{timestamp_str}") as run:
                    mlflow_run_id = run.info.run_id
                    mlflow.log_params(c_params)
                    mlflow.log_param("dataset_fingerprint", dataset_fingerprint)
                    mlflow.log_param("model_architecture", c_name)
                    mlflow.log_metrics(combined_metrics)

                    # Save curve artifacts as json
                    curves_file = Path(f"reports/curves_{c_name}.json")
                    curves_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(curves_file, "w") as f:
                        json.dump(curves, f, indent=2)
                    mlflow.log_artifact(str(curves_file))

                    # Save feature importance
                    imp_file = Path(f"reports/feature_importance_{c_name}.json")
                    with open(imp_file, "w") as f:
                        json.dump(feat_imp, f, indent=2)
                    mlflow.log_artifact(str(imp_file))
            except Exception as e:
                logger.warning(f"MLflow run logging error: {e}")

        version = f"{c_name.lower()}-v{timestamp_str}"
        model_entry = registry.register_model(
            version=version,
            model_name=c_name,
            pipeline_obj=pipeline,
            metrics=combined_metrics,
            params=c_params,
            feature_importance=feat_imp,
            mlflow_run_id=mlflow_run_id,
            dataset_fingerprint=dataset_fingerprint,
        )

        run_results.append({
            "name": c_name,
            "version": version,
            "val_roc_auc": val_metrics["val_roc_auc"],
            "val_f1": val_metrics["val_f1"],
            "test_roc_auc": test_metrics["test_roc_auc"],
            "test_f1": test_metrics["test_f1"],
            "metrics": combined_metrics,
            "model_entry": model_entry
        })

    # 6. Select Champion based on Business Metric (Validation ROC-AUC with F1 tie-breaker)
    # Rationale: Customer churn intervention is cost-sensitive. ROC-AUC balances ranking quality across threshold cutoffs,
    # and F1 balances precision and recall for actionable churn prevention.
    run_results.sort(key=lambda r: (r["val_roc_auc"], r["val_f1"]), reverse=True)
    best_candidate = run_results[0]
    registry.promote_model(
        best_candidate["version"],
        reason=f"Selected as initial champion with highest val ROC-AUC ({best_candidate['val_roc_auc']:.4f}) and F1 ({best_candidate['val_f1']:.4f})"
    )

    total_duration = round(time.time() - start_time, 2)
    return {
        "status": "COMPLETED",
        "dataset_rows": len(clean_df),
        "dataset_fingerprint": dataset_fingerprint,
        "total_duration_seconds": total_duration,
        "champion_selected": best_candidate["version"],
        "champion_model_name": best_candidate["name"],
        "champion_metrics": best_candidate["metrics"],
        "all_candidates": [
            {
                "name": r["name"],
                "version": r["version"],
                "val_roc_auc": r["val_roc_auc"],
                "val_f1": r["val_f1"],
                "test_roc_auc": r["test_roc_auc"],
                "test_f1": r["test_f1"],
            }
            for r in run_results
        ]
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running standalone model training pipeline...")
    res = train_and_evaluate_all_models()
    print(json.dumps(res, indent=2))
