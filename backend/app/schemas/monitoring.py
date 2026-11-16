"""Pydantic schemas for data drift monitoring and status."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DriftRunRequest(BaseModel):
    sample_limit: Optional[int] = Field(500, description="Number of recent production predictions to evaluate")
    p_value_threshold: Optional[float] = Field(0.05, description="P-value significance cutoff")
    drift_share_threshold: Optional[float] = Field(0.30, description="Share of drifted features to flag dataset drift")
    trigger_type: Optional[str] = Field("MANUAL", description="'MANUAL' or 'SCHEDULED'")


class FeatureDriftDetail(BaseModel):
    feature_name: str
    feature_type: str
    drift_score: float
    p_value: Optional[float] = None
    is_drifted: bool
    stat_test: Optional[str] = None
    reference_distribution: Optional[Dict[str, Any]] = None
    current_distribution: Optional[Dict[str, Any]] = None


class MonitoringRunResponse(BaseModel):
    id: int
    timestamp: str
    run_type: str
    reference_rows: int
    current_rows: int
    dataset_drift_detected: bool
    drift_share: float
    drifted_features_count: int
    total_features_count: int
    p_value_threshold: float
    drift_share_threshold: float
    report_html_path: Optional[str] = None
    report_json_path: Optional[str] = None
    features: List[FeatureDriftDetail] = []


class MonitoringStatusResponse(BaseModel):
    status: str
    latest_run: Optional[MonitoringRunResponse] = None
    total_monitoring_runs: int
    active_dataset_drift: bool
    drift_share: float
    total_prediction_records: int
    total_labeled_records: int
