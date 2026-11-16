"""Pydantic schemas for delayed-label ingestion and model performance monitoring."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GroundTruthItem(BaseModel):
    customer_id: str
    actual_churn: int = Field(..., ge=0, le=1, description="Actual churn outcome: 0 = No, 1 = Yes")
    prediction_log_id: Optional[int] = Field(None, description="Optional ID of specific prediction record")


class BatchGroundTruthRequest(BaseModel):
    labels: List[GroundTruthItem]


class PerformanceMetricsResponse(BaseModel):
    total_evaluated: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    estimated_net_benefit_usd: float
    performance_drop_detected: bool
    evaluation_window: str
