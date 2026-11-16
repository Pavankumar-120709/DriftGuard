"""FastAPI main entrypoint for DriftGuard MLOps Platform."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database import init_db
from backend.app.api.predict import router as predict_router
from backend.app.api.monitoring import router as monitoring_router
from backend.app.api.retraining import router as retraining_router
from backend.app.api.models_api import router as models_router
from backend.app.api.labels import router as labels_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("driftguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing DriftGuard MLOps Platform...")
    init_db()
    # Check if a model is trained; if not, train initial model baseline
    from ml.registry import ModelRegistryManager
    registry = ModelRegistryManager()
    if registry.get_champion() is None:
        logger.info("No active champion found in registry. Triggering initial model training pipeline...")
        try:
            from ml.train import train_and_evaluate_all_models
            train_and_evaluate_all_models(
                experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
                mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI
            )
            logger.info("Initial champion model successfully trained and registered.")
        except Exception as e:
            logger.error(f"Error initializing champion model on startup: {e}")
    yield
    logger.info("Shutting down DriftGuard MLOps Platform...")


app = FastAPI(
    title="DriftGuard — End-to-End MLOps Platform",
    description="Automated Customer Churn Prediction with Evidently AI Drift Detection and Controlled Retraining",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend Next.js application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(predict_router)
app.include_router(monitoring_router)
app.include_router(retraining_router)
app.include_router(models_router)
app.include_router(labels_router)


@app.get("/")
def root():
    return {
        "platform": "DriftGuard MLOps Platform",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
        "monitoring_status": "/monitoring/status"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
