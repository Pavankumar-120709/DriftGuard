"""Inference service for customer churn prediction."""

import datetime
import logging
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.models.schema import PredictionLog
from backend.app.schemas.prediction import CustomerFeatures, PredictionResponse
from ml.registry import ModelRegistryManager

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self):
        self.registry = ModelRegistryManager()
        self._cached_champion = None
        self._cached_pipeline = None

    def _get_active_pipeline(self) -> Tuple[str, Any]:
        champion = self.registry.get_champion()
        if not champion:
            raise RuntimeError("No champion model registered. Please train models first.")
        
        # Invalidate cache if version changed
        if not self._cached_champion or self._cached_champion["version"] != champion["version"]:
            self._cached_champion = champion
            self._cached_pipeline = self.registry.load_champion_pipeline()

        if self._cached_pipeline is None:
            raise RuntimeError(f"Failed to load pipeline artifact for version {champion['version']}.")

        return champion["version"], self._cached_pipeline

    def predict_single(self, features: CustomerFeatures, db: Session) -> PredictionResponse:
        model_version, pipeline = self._get_active_pipeline()
        start = time.perf_counter()

        # Convert to single-row DataFrame
        feat_dict = features.model_dump()
        input_df = pd.DataFrame([feat_dict])

        # Predict
        pred_class = int(pipeline.predict(input_df)[0])
        pred_proba = float(pipeline.predict_proba(input_df)[0, 1])
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Log prediction to database for drift and performance tracking
        log_entry = PredictionLog(
            customer_id=features.customerID or f"CUST-{int(time.time()*1000)}",
            model_version=model_version,
            input_features=feat_dict,
            predicted_class=pred_class,
            prediction_probability=round(pred_proba, 4),
            latency_ms=latency_ms,
            timestamp=datetime.datetime.utcnow()
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        return PredictionResponse(
            customer_id=log_entry.customer_id,
            churn_prediction=pred_class,
            churn_label="Yes" if pred_class == 1 else "No",
            churn_probability=round(pred_proba, 4),
            model_version=model_version,
            latency_ms=latency_ms,
            timestamp=log_entry.timestamp.isoformat(),
            prediction_id=log_entry.id
        )

    def predict_batch(self, customers: List[CustomerFeatures], db: Session) -> Tuple[List[PredictionResponse], float, int, int]:
        model_version, pipeline = self._get_active_pipeline()
        if not customers:
            return [], 0.0, 0, 0

        feat_dicts = [c.model_dump() for c in customers]
        input_df = pd.DataFrame(feat_dicts)

        start = time.perf_counter()
        preds = pipeline.predict(input_df)
        probas = pipeline.predict_proba(input_df)[:, 1]
        batch_latency = (time.perf_counter() - start) * 1000
        per_item_latency = round(batch_latency / len(customers), 2)

        now = datetime.datetime.utcnow()
        responses = []
        db_entries = []
        churn_count = 0

        for i, c in enumerate(customers):
            p_class = int(preds[i])
            p_prob = round(float(probas[i]), 4)
            c_id = c.customerID or f"CUST-{int(time.time()*1000)}-{i}"
            if p_class == 1:
                churn_count += 1

            log_entry = PredictionLog(
                customer_id=c_id,
                model_version=model_version,
                input_features=feat_dicts[i],
                predicted_class=p_class,
                prediction_probability=p_prob,
                latency_ms=per_item_latency,
                timestamp=now
            )
            db_entries.append(log_entry)

            responses.append(PredictionResponse(
                customer_id=c_id,
                churn_prediction=p_class,
                churn_label="Yes" if p_class == 1 else "No",
                churn_probability=p_prob,
                model_version=model_version,
                latency_ms=per_item_latency,
                timestamp=now.isoformat(),
                prediction_id=None
            ))

        db.add_all(db_entries)
        db.commit()

        # Update prediction_id in responses
        for i, entry in enumerate(db_entries):
            responses[i].prediction_id = entry.id

        avg_prob = round(float(np.mean(probas)), 4)
        non_churn_count = len(customers) - churn_count
        return responses, avg_prob, churn_count, non_churn_count


inference_service = InferenceService()
