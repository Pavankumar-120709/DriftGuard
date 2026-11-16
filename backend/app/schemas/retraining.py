"""Pydantic schemas for automated retraining and model management."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RetrainTriggerRequest(BaseModel):
    trigger_reason: str = Field("Manual retraining requested from dashboard", description="Reason for triggering retraining")
    min_improvement_threshold: float = Field(0.005, description="Required primary metric improvement over champion")
    include_production_labels: bool = Field(True, description="Whether to include recent labeled production data in training")


class RetrainingJobResponse(BaseModel):
    id: int
    triggered_at: str
    completed_at: Optional[str] = None
    trigger_reason: str
    status: str
    challenger_version: Optional[str] = None
    champion_version: Optional[str] = None
    decision: str
    decision_reason: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class RollbackResponse(BaseModel):
    success: bool
    message: str
    current_champion: Optional[str] = None
    previous_champion: Optional[str] = None


class ModelInfoResponse(BaseModel):
    version: str
    model_name: str
    is_champion: bool
    status: str
    created_at: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    feature_importance: Optional[List[Dict[str, Any]]] = None


class ChampionChallengerComparisonResponse(BaseModel):
    champion: Optional[ModelInfoResponse] = None
    challenger: Optional[ModelInfoResponse] = None
    primary_metric: str = "val_roc_auc"
    champion_score: float = 0.0
    challenger_score: float = 0.0
    margin: float = 0.0
    meets_threshold: bool = False
    recommendation: str = ""
