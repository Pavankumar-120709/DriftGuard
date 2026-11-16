from backend.app.api.predict import router as predict_router
from backend.app.api.monitoring import router as monitoring_router
from backend.app.api.retraining import router as retraining_router
from backend.app.api.models_api import router as models_router
from backend.app.api.labels import router as labels_router

__all__ = [
    "predict_router",
    "monitoring_router",
    "retraining_router",
    "models_router",
    "labels_router",
]
