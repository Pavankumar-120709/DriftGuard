"""Application settings and environment configuration for DriftGuard."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "DriftGuard MLOps Platform"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'driftguard.db'}")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlruns.db'}")
    MLFLOW_ARTIFACT_ROOT: str = os.getenv("MLFLOW_ARTIFACT_ROOT", str(BASE_DIR / "mlartifacts"))
    MLFLOW_EXPERIMENT_NAME: str = "customer-churn-prediction"

    # Monitoring & Evidently
    REPORTS_DIR: str = str(BASE_DIR / "reports" / "drift")
    REFERENCE_DATA_PATH: str = str(BASE_DIR / "data" / "processed" / "reference_data.csv")
    DRIFT_SHARE_THRESHOLD: float = 0.30
    P_VALUE_THRESHOLD: float = 0.05

    # Retraining Gates
    RETRAINING_DRIFT_THRESHOLD: float = 0.35
    RETRAINING_PERFORMANCE_DROP_THRESHOLD: float = 0.05
    MIN_RETRAINING_SAMPLES: int = 50
    AUTO_RETRAIN_ENABLED: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
