"""Pytest configuration and shared fixtures for DriftGuard."""

import os
import sys
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database import Base, get_db
import backend.app.models.schema  # noqa: F401 - Register all models with Base.metadata
from backend.app.main import app
from ml.dataset import generate_synthetic_telco_data


@pytest.fixture(scope="session")
def sample_raw_data() -> pd.DataFrame:
    """Provide reproducible synthetic Telco Customer Churn dataset."""
    return generate_synthetic_telco_data(num_samples=200, random_state=42)


from sqlalchemy.pool import StaticPool

@pytest.fixture(scope="session")
def test_db_engine():
    """Create in-memory SQLite database for testing with StaticPool so all connections share state."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(test_db_engine):
    """Provide isolated transactional session per test."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden database dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
