"""FastAPI prediction router for DriftGuard."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.prediction import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from backend.app.services.inference_service import inference_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(features: CustomerFeatures, db: Session = Depends(get_db)):
    """Generate customer churn prediction for a single customer and log for drift monitoring."""
    try:
        return inference_service.predict_single(features, db)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(batch_request: BatchPredictionRequest, db: Session = Depends(get_db)):
    """Generate predictions for a batch of customers."""
    try:
        predictions, avg_prob, churn_count, non_churn_count = inference_service.predict_batch(
            batch_request.customers, db
        )
        model_version = predictions[0].model_version if predictions else "unknown"
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            model_version=model_version,
            average_churn_probability=avg_prob,
            churn_count=churn_count,
            non_churn_count=non_churn_count
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch error: {str(e)}")
