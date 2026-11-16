"""Automated, controlled model retraining service with quality gates and safe promotion."""

import datetime
import logging
import threading
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import SessionLocal
from backend.app.models.schema import (
    RetrainingJob,
    ModelPromotion,
    PredictionLog,
    GroundTruthLabel,
    ModelVersionEntity,
)
from backend.app.schemas.retraining import RetrainingJobResponse, RollbackResponse
from ml.dataset import ingest_telco_data
from ml.registry import ModelRegistryManager
from ml.train import train_and_evaluate_all_models
from ml.validation import validate_raw_dataset, clean_and_impute_dataset

logger = logging.getLogger(__name__)

# Global lock to prevent concurrent retraining executions
_retraining_lock = threading.Lock()


class RetrainingService:
    def __init__(self):
        self.registry = ModelRegistryManager()

    def is_locked(self) -> bool:
        return _retraining_lock.locked()

    def trigger_retraining(
        self,
        db: Session,
        trigger_reason: str = "Manual trigger",
        min_improvement_threshold: float = 0.005,
        include_production_labels: bool = True
    ) -> RetrainingJobResponse:
        """Execute the end-to-end retraining workflow under concurrency protection."""
        if not _retraining_lock.acquire(blocking=False):
            raise RuntimeError("A model retraining job is already currently running. Duplicate execution prevented.")

        # Create RetrainingJob record
        current_champ = self.registry.get_champion()
        current_champ_ver = current_champ["version"] if current_champ else None

        job = RetrainingJob(
            triggered_at=datetime.datetime.utcnow(),
            trigger_reason=trigger_reason,
            status="RUNNING",
            champion_version=current_champ_ver,
            decision="NONE"
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        try:
            # 1. Assemble Training Data (Reference + Labeled Production)
            base_df = ingest_telco_data()

            if include_production_labels:
                labeled_pairs = (
                    db.query(PredictionLog.input_features, GroundTruthLabel.actual_churn)
                    .join(GroundTruthLabel, PredictionLog.id == GroundTruthLabel.prediction_log_id)
                    .all()
                )
                if labeled_pairs:
                    prod_rows = []
                    for features, churn in labeled_pairs:
                        row = dict(features)
                        row["Churn"] = "Yes" if churn == 1 else "No"
                        prod_rows.append(row)
                    prod_df = pd.DataFrame(prod_rows)
                    combined_df = pd.concat([base_df, prod_df], ignore_index=True)
                    logger.info(f"Incorporated {len(prod_df)} labeled production samples into retraining.")
                else:
                    combined_df = base_df
            else:
                combined_df = base_df

            # 2. Data Validation Gate
            val_report = validate_raw_dataset(combined_df)
            if not val_report["is_valid"]:
                raise ValueError(f"Retraining data validation failed: {val_report['errors']}")

            # 3. Model Training & MLflow Logging
            train_result = train_and_evaluate_all_models(
                df=combined_df,
                experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
                mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI,
            )

            challenger_version = train_result["champion_selected"]
            job.challenger_version = challenger_version
            job.training_metrics = train_result["champion_metrics"]

            # 4. Champion / Challenger Evaluation
            promoted, reason = self.registry.evaluate_challenger_promotion(
                challenger_version=challenger_version,
                primary_metric="val_roc_auc",
                min_improvement_threshold=min_improvement_threshold,
            )

            # 5. Model Promotion or Rejection Decision
            if promoted:
                self.registry.promote_model(challenger_version, reason=reason)
                job.decision = "PROMOTED"
                job.decision_reason = reason

                # Log promotion audit
                promo_audit = ModelPromotion(
                    timestamp=datetime.datetime.utcnow(),
                    promoted_version=challenger_version,
                    previous_version=current_champ_ver,
                    primary_metric="val_roc_auc",
                    challenger_score=train_result["champion_metrics"].get("val_roc_auc"),
                    champion_score=current_champ["metrics"].get("val_roc_auc") if current_champ else None,
                    decision="PROMOTED",
                    reason=reason
                )
                db.add(promo_audit)
            else:
                job.decision = "REJECTED"
                job.decision_reason = reason
                # Re-affirm original champion in registry
                if current_champ_ver:
                    self.registry.promote_model(current_champ_ver, reason="Preserved previous champion after challenger rejection")

            job.status = "COMPLETED"
            job.completed_at = datetime.datetime.utcnow()
            db.commit()
            db.refresh(job)

            return RetrainingJobResponse(
                id=job.id,
                triggered_at=job.triggered_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                trigger_reason=job.trigger_reason,
                status=job.status,
                challenger_version=job.challenger_version,
                champion_version=job.champion_version,
                decision=job.decision,
                decision_reason=job.decision_reason,
                training_metrics=job.training_metrics,
                error_message=job.error_message
            )

        except Exception as e:
            logger.error(f"Retraining workflow failed: {e}", exc_info=True)
            job.status = "FAILED"
            job.error_message = str(e)
            job.completed_at = datetime.datetime.utcnow()
            db.commit()
            raise
        finally:
            _retraining_lock.release()

    def rollback_model(self, db: Session) -> RollbackResponse:
        """Rollback to the previous champion model version."""
        curr_champ = self.registry.get_champion()
        curr_ver = curr_champ["version"] if curr_champ else None

        success, msg = self.registry.rollback_to_previous_champion()
        if not success:
            return RollbackResponse(
                success=False,
                message=msg,
                current_champion=curr_ver,
                previous_champion=None
            )

        new_champ = self.registry.get_champion()
        new_ver = new_champ["version"] if new_champ else None

        # Log rollback audit
        promo_audit = ModelPromotion(
            timestamp=datetime.datetime.utcnow(),
            promoted_version=new_ver,
            previous_version=curr_ver,
            primary_metric="manual_rollback",
            challenger_score=None,
            champion_score=None,
            decision="ROLLED_BACK",
            reason=f"Manual rollback requested: {msg}"
        )
        db.add(promo_audit)
        db.commit()

        return RollbackResponse(
            success=True,
            message=msg,
            current_champion=new_ver,
            previous_champion=curr_ver
        )


retraining_service = RetrainingService()
