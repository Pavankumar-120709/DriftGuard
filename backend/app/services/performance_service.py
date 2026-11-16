"""Delayed-label ingestion and model performance monitoring service for DriftGuard."""

import datetime
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.schema import PredictionLog, GroundTruthLabel
from backend.app.schemas.labels import GroundTruthItem, PerformanceMetricsResponse
from ml.evaluate import evaluate_classifier
from ml.registry import ModelRegistryManager

logger = logging.getLogger(__name__)


class PerformanceMonitoringService:
    def __init__(self):
        self.registry = ModelRegistryManager()

    def ingest_ground_truth_labels(
        self,
        labels: List[GroundTruthItem],
        db: Session
    ) -> Dict[str, Any]:
        """Ingest actual delayed outcomes and link with previous predictions."""
        matched_count = 0
        inserted_count = 0

        for item in labels:
            # Check if this customer or prediction already labeled
            pred_log = None
            if item.prediction_log_id:
                pred_log = db.query(PredictionLog).filter(PredictionLog.id == item.prediction_log_id).first()
            elif item.customer_id:
                pred_log = (
                    db.query(PredictionLog)
                    .filter(PredictionLog.customer_id == item.customer_id)
                    .order_by(PredictionLog.timestamp.desc())
                    .first()
                )

            pred_log_id = pred_log.id if pred_log else None
            if pred_log_id:
                matched_count += 1

            # Check if label already exists
            existing = None
            if pred_log_id:
                existing = db.query(GroundTruthLabel).filter(GroundTruthLabel.prediction_log_id == pred_log_id).first()
            if not existing and item.customer_id:
                existing = db.query(GroundTruthLabel).filter(GroundTruthLabel.customer_id == item.customer_id).first()

            if existing:
                existing.actual_churn = item.actual_churn
                existing.labeled_at = datetime.datetime.utcnow()
            else:
                new_label = GroundTruthLabel(
                    prediction_log_id=pred_log_id,
                    customer_id=item.customer_id,
                    actual_churn=item.actual_churn,
                    labeled_at=datetime.datetime.utcnow()
                )
                db.add(new_label)
                inserted_count += 1

        db.commit()
        return {
            "total_submitted": len(labels),
            "matched_with_predictions": matched_count,
            "new_labels_created": inserted_count,
        }

    def compute_current_performance(
        self,
        db: Session,
        model_version: str = None
    ) -> PerformanceMetricsResponse:
        """Calculate true performance metrics by joining predictions with delayed ground-truth labels."""
        query = (
            db.query(
                PredictionLog.predicted_class,
                PredictionLog.prediction_probability,
                GroundTruthLabel.actual_churn
            )
            .join(GroundTruthLabel, PredictionLog.id == GroundTruthLabel.prediction_log_id)
        )

        if model_version:
            query = query.filter(PredictionLog.model_version == model_version)

        records = query.all()

        if not records:
            # If no labeled pairs yet, return default baseline zeros
            return PerformanceMetricsResponse(
                total_evaluated=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                roc_auc=0.0,
                brier_score=0.0,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                estimated_net_benefit_usd=0.0,
                performance_drop_detected=False,
                evaluation_window="no_labeled_records"
            )

        y_pred = np.array([r[0] for r in records])
        y_proba = np.array([r[1] for r in records])
        y_true = np.array([r[2] for r in records])

        metrics = evaluate_classifier(y_true, y_pred, y_proba, prefix="prod")

        # Check performance drop compared to champion validation baseline
        champion = self.registry.get_champion()
        perf_drop = False
        if champion and "metrics" in champion:
            champ_baseline_roc = champion["metrics"].get("val_roc_auc", 0.80)
            current_roc = metrics["prod_roc_auc"]
            if (champ_baseline_roc - current_roc) > settings.RETRAINING_PERFORMANCE_DROP_THRESHOLD:
                perf_drop = True

        return PerformanceMetricsResponse(
            total_evaluated=len(records),
            accuracy=metrics["prod_accuracy"],
            precision=metrics["prod_precision"],
            recall=metrics["prod_recall"],
            f1=metrics["prod_f1"],
            roc_auc=metrics["prod_roc_auc"],
            brier_score=metrics["prod_brier_score"],
            true_positives=metrics["prod_true_positives"],
            false_positives=metrics["prod_false_positives"],
            true_negatives=metrics["prod_true_negatives"],
            false_negatives=metrics["prod_false_negatives"],
            estimated_net_benefit_usd=metrics["prod_estimated_net_benefit_usd"],
            performance_drop_detected=perf_drop,
            evaluation_window=f"last_{len(records)}_labeled_predictions"
        )


performance_service = PerformanceMonitoringService()
