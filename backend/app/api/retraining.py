"""FastAPI retraining router for DriftGuard."""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.schema import RetrainingJob, ModelPromotion
from backend.app.schemas.retraining import (
    RetrainTriggerRequest,
    RetrainingJobResponse,
    RollbackResponse,
)
from backend.app.services.retraining_service import retraining_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Automated Retraining"])


@router.post("/retrain/trigger", response_model=RetrainingJobResponse, status_code=status.HTTP_200_OK)
def trigger_retraining(req: RetrainTriggerRequest = RetrainTriggerRequest(), db: Session = Depends(get_db)):
    """Trigger the automated retraining pipeline with champion/challenger validation gates."""
    try:
        return retraining_service.trigger_retraining(
            db=db,
            trigger_reason=req.trigger_reason,
            min_improvement_threshold=req.min_improvement_threshold,
            include_production_labels=req.include_production_labels,
        )
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(re))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Retraining execution error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/retrain/rollback", response_model=RollbackResponse)
def rollback_model(db: Session = Depends(get_db)):
    """Rollback to the previous champion model version."""
    try:
        return retraining_service.rollback_model(db)
    except Exception as e:
        logger.error(f"Rollback error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/retrain/history", response_model=List[RetrainingJobResponse])
def get_retraining_history(db: Session = Depends(get_db)):
    """Return all historical retraining attempts, decisions, and outcomes."""
    jobs = db.query(RetrainingJob).order_by(RetrainingJob.triggered_at.desc()).limit(50).all()
    results = []
    for j in jobs:
        results.append(RetrainingJobResponse(
            id=j.id,
            triggered_at=j.triggered_at.isoformat(),
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            trigger_reason=j.trigger_reason,
            status=j.status,
            challenger_version=j.challenger_version,
            champion_version=j.champion_version,
            decision=j.decision,
            decision_reason=j.decision_reason,
            training_metrics=j.training_metrics,
            error_message=j.error_message
        ))
    return results


@router.get("/retrain/promotions")
def get_promotions_history(db: Session = Depends(get_db)):
    """Return the audit log of model promotions and rollbacks."""
    promotions = db.query(ModelPromotion).order_by(ModelPromotion.timestamp.desc()).limit(50).all()
    return [
        {
            "id": p.id,
            "timestamp": p.timestamp.isoformat(),
            "promoted_version": p.promoted_version,
            "previous_version": p.previous_version,
            "primary_metric": p.primary_metric,
            "challenger_score": p.challenger_score,
            "champion_score": p.champion_score,
            "decision": p.decision,
            "reason": p.reason,
        }
        for p in promotions
    ]
