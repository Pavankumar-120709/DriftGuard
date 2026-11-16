"""FastAPI labels and ground truth router for DriftGuard."""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.labels import (
    GroundTruthItem,
    BatchGroundTruthRequest,
    PerformanceMetricsResponse,
)
from backend.app.services.performance_service import performance_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ground Truth & Performance"])


@router.post("/labels", status_code=status.HTTP_201_CREATED)
def ingest_labels(batch: BatchGroundTruthRequest, db: Session = Depends(get_db)):
    """Ingest delayed ground truth labels and pair them with past production predictions."""
    try:
        return performance_service.ingest_ground_truth_labels(batch.labels, db)
    except Exception as e:
        logger.error(f"Error ingesting ground truth labels: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/labels/performance", response_model=PerformanceMetricsResponse)
def get_model_performance(model_version: str = None, db: Session = Depends(get_db)):
    """Calculate true production performance metrics based on actual delayed churn outcomes."""
    try:
        return performance_service.compute_current_performance(db, model_version=model_version)
    except Exception as e:
        logger.error(f"Error computing performance metrics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
