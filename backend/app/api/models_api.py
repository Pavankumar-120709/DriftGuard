"""FastAPI model management and info router for DriftGuard."""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.retraining import (
    ModelInfoResponse,
    ChampionChallengerComparisonResponse,
)
from ml.registry import ModelRegistryManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Model Management"])


@router.get("/health")
def health_check():
    """Health check endpoint for Docker / load balancers."""
    registry = ModelRegistryManager()
    champion = registry.get_champion()
    return {
        "status": "healthy",
        "service": "DriftGuard MLOps API",
        "champion_available": champion is not None,
        "champion_version": champion["version"] if champion else None,
    }


@router.get("/model/info", response_model=ModelInfoResponse)
def get_current_model_info():
    """Return active champion model metadata, parameters, and version."""
    registry = ModelRegistryManager()
    champ = registry.get_champion()
    if not champ:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active champion model found.")
    
    return ModelInfoResponse(
        version=champ["version"],
        model_name=champ["model_name"],
        is_champion=True,
        status=champ.get("status", "ACTIVE"),
        mlflow_run_id=champ.get("mlflow_run_id"),
        dataset_fingerprint=champ.get("dataset_fingerprint"),
        params=champ.get("params"),
        metrics=champ.get("metrics"),
        feature_importance=champ.get("feature_importance")
    )


@router.get("/model/metrics")
def get_current_model_metrics():
    """Return evaluation metrics of the active champion model."""
    registry = ModelRegistryManager()
    champ = registry.get_champion()
    if not champ:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active champion model found.")
    return {
        "model_version": champ["version"],
        "model_name": champ["model_name"],
        "metrics": champ["metrics"]
    }


@router.get("/models", response_model=List[ModelInfoResponse])
def list_registered_models():
    """Return all models in the local model registry."""
    registry = ModelRegistryManager()
    models = registry.list_models()
    return [
        ModelInfoResponse(
            version=m["version"],
            model_name=m["model_name"],
            is_champion=m.get("is_champion", False),
            status=m.get("status", "REGISTERED"),
            mlflow_run_id=m.get("mlflow_run_id"),
            dataset_fingerprint=m.get("dataset_fingerprint"),
            params=m.get("params"),
            metrics=m.get("metrics"),
            feature_importance=m.get("feature_importance")
        )
        for m in models
    ]


@router.get("/models/compare", response_model=ChampionChallengerComparisonResponse)
def compare_models(challenger_version: Optional[str] = None):
    """Compare current champion against a challenger model."""
    registry = ModelRegistryManager()
    champ = registry.get_champion()
    all_models = registry.list_models()

    if not champ:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No champion model found to compare.")

    challenger_entry = None
    if challenger_version:
        for m in all_models:
            if m["version"] == challenger_version:
                challenger_entry = m
                break
    else:
        # Find latest non-champion model
        non_champs = [m for m in all_models if not m.get("is_champion", False)]
        if non_champs:
            challenger_entry = non_champs[-1]

    champ_score = champ["metrics"].get("val_roc_auc", 0.0)
    chal_score = challenger_entry["metrics"].get("val_roc_auc", 0.0) if challenger_entry else 0.0
    margin = chal_score - champ_score

    meets_thresh, reason = (False, "No challenger specified or available")
    if challenger_entry:
        meets_thresh, reason = registry.evaluate_challenger_promotion(
            challenger_entry["version"],
            primary_metric="val_roc_auc",
            min_improvement_threshold=0.005
        )

    champ_resp = ModelInfoResponse(
        version=champ["version"],
        model_name=champ["model_name"],
        is_champion=True,
        status="ACTIVE",
        metrics=champ.get("metrics"),
        params=champ.get("params"),
        feature_importance=champ.get("feature_importance")
    )

    chal_resp = None
    if challenger_entry:
        chal_resp = ModelInfoResponse(
            version=challenger_entry["version"],
            model_name=challenger_entry["model_name"],
            is_champion=False,
            status="CHALLENGER",
            metrics=challenger_entry.get("metrics"),
            params=challenger_entry.get("params"),
            feature_importance=challenger_entry.get("feature_importance")
        )

    return ChampionChallengerComparisonResponse(
        champion=champ_resp,
        challenger=chal_resp,
        primary_metric="val_roc_auc",
        champion_score=round(champ_score, 4),
        challenger_score=round(chal_score, 4),
        margin=round(margin, 4),
        meets_threshold=meets_thresh,
        recommendation=reason
    )
