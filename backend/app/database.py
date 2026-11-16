"""Database initialization and session management for DriftGuard."""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.app.config import settings

logger = logging.getLogger(__name__)

is_sqlite = settings.DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for obtaining database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all database tables."""
    from backend.app.models.schema import (
        PredictionLog,
        GroundTruthLabel,
        MonitoringRun,
        DriftMetric,
        ModelVersionEntity,
        RetrainingJob,
        ModelPromotion,
    )
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
