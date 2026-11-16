"""SQLAlchemy ORM models for DriftGuard persistence."""

import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Text,
    Enum,
    Index
)
from sqlalchemy.orm import relationship
from backend.app.database import Base


class PredictionLog(Base):
    """Stores incoming production inference features and predictions for monitoring."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(String(64), index=True, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    model_version = Column(String(64), index=True, nullable=False)
    input_features = Column(JSON, nullable=False)
    predicted_class = Column(Integer, nullable=False)  # 0 or 1
    prediction_probability = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=True)

    # Relationship to ground truth label once it becomes available
    ground_truth = relationship("GroundTruthLabel", back_populates="prediction", uselist=False)

    __table_args__ = (
        Index("idx_pred_model_time", "model_version", "timestamp"),
    )


class GroundTruthLabel(Base):
    """Stores delayed actual customer churn outcomes associated with prior predictions."""
    __tablename__ = "ground_truth_labels"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_log_id = Column(Integer, ForeignKey("prediction_logs.id"), unique=True, index=True, nullable=True)
    customer_id = Column(String(64), index=True, nullable=False)
    actual_churn = Column(Integer, nullable=False)  # 0 or 1
    labeled_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    source = Column(String(64), default="delayed_label_ingestion")

    prediction = relationship("PredictionLog", back_populates="ground_truth")


class MonitoringRun(Base):
    """Stores metadata and aggregate statistics from an Evidently AI drift monitoring run."""
    __tablename__ = "monitoring_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    run_type = Column(String(32), default="MANUAL")  # MANUAL, SCHEDULED, INFERENCE_TRIGGER
    reference_rows = Column(Integer, nullable=False)
    current_rows = Column(Integer, nullable=False)
    dataset_drift_detected = Column(Boolean, default=False)
    drift_share = Column(Float, default=0.0)
    drifted_features_count = Column(Integer, default=0)
    total_features_count = Column(Integer, default=0)
    p_value_threshold = Column(Float, default=0.05)
    drift_share_threshold = Column(Float, default=0.30)
    report_html_path = Column(Text, nullable=True)
    report_json_path = Column(Text, nullable=True)
    summary_metrics = Column(JSON, nullable=True)

    metrics = relationship("DriftMetric", back_populates="monitoring_run", cascade="all, delete-orphan")


class DriftMetric(Base):
    """Detailed feature-level drift statistics for each monitored column."""
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    monitoring_run_id = Column(Integer, ForeignKey("monitoring_runs.id"), index=True, nullable=False)
    feature_name = Column(String(128), index=True, nullable=False)
    feature_type = Column(String(32), nullable=False)  # num or cat
    drift_score = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    is_drifted = Column(Boolean, nullable=False)
    stat_test = Column(String(64), nullable=True)
    reference_distribution = Column(JSON, nullable=True)
    current_distribution = Column(JSON, nullable=True)

    monitoring_run = relationship("MonitoringRun", back_populates="metrics")


class ModelVersionEntity(Base):
    """Persistent catalog of registered models and champion state."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version = Column(String(64), unique=True, index=True, nullable=False)
    model_name = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_champion = Column(Boolean, default=False, index=True)
    status = Column(String(32), default="ACTIVE")  # ACTIVE, ARCHIVED, REJECTED
    metrics = Column(JSON, nullable=False)
    params = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    mlflow_run_id = Column(String(128), nullable=True)
    dataset_fingerprint = Column(String(64), nullable=True)


class RetrainingJob(Base):
    """Tracks every automated or manual retraining run, evaluation decision, and status."""
    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    triggered_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    trigger_reason = Column(String(256), nullable=False)
    status = Column(String(32), default="PENDING", index=True)  # PENDING, RUNNING, COMPLETED, FAILED
    challenger_version = Column(String(64), nullable=True)
    champion_version = Column(String(64), nullable=True)
    decision = Column(String(32), default="NONE")  # PROMOTED, REJECTED, NONE
    decision_reason = Column(Text, nullable=True)
    training_metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)


class ModelPromotion(Base):
    """Audit log of champion promotions and rollbacks."""
    __tablename__ = "model_promotions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    promoted_version = Column(String(64), nullable=False)
    previous_version = Column(String(64), nullable=True)
    primary_metric = Column(String(64), default="val_roc_auc")
    challenger_score = Column(Float, nullable=True)
    champion_score = Column(Float, nullable=True)
    decision = Column(String(32), nullable=False)  # PROMOTED, ROLLED_BACK
    reason = Column(Text, nullable=False)
