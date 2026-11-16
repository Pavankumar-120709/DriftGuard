"""FastAPI monitoring router for DriftGuard."""

import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models.schema import MonitoringRun, PredictionLog, GroundTruthLabel
from backend.app.schemas.monitoring import (
    DriftRunRequest,
    MonitoringRunResponse,
    MonitoringStatusResponse,
)
from backend.app.services.drift_service import drift_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Monitoring & Drift Detection"])


@router.get("/monitoring/status", response_model=MonitoringStatusResponse)
def get_monitoring_status(db: Session = Depends(get_db)):
    """Return high-level drift monitoring health, active status, and sample counts."""
    latest_run = (
        db.query(MonitoringRun)
        .order_by(MonitoringRun.timestamp.desc())
        .first()
    )

    total_runs = db.query(MonitoringRun).count()
    total_preds = db.query(PredictionLog).count()
    total_labels = db.query(GroundTruthLabel).count()

    latest_resp = None
    active_drift = False
    drift_share = 0.0

    if latest_run:
        active_drift = latest_run.dataset_drift_detected
        drift_share = latest_run.drift_share
        features = []
        for m in latest_run.metrics:
            features.append({
                "feature_name": m.feature_name,
                "feature_type": m.feature_type,
                "drift_score": m.drift_score,
                "p_value": m.p_value,
                "is_drifted": m.is_drifted,
                "stat_test": m.stat_test,
                "reference_distribution": m.reference_distribution,
                "current_distribution": m.current_distribution,
            })
        latest_resp = MonitoringRunResponse(
            id=latest_run.id,
            timestamp=latest_run.timestamp.isoformat(),
            run_type=latest_run.run_type,
            reference_rows=latest_run.reference_rows,
            current_rows=latest_run.current_rows,
            dataset_drift_detected=latest_run.dataset_drift_detected,
            drift_share=latest_run.drift_share,
            drifted_features_count=latest_run.drifted_features_count,
            total_features_count=latest_run.total_features_count,
            p_value_threshold=latest_run.p_value_threshold,
            drift_share_threshold=latest_run.drift_share_threshold,
            report_html_path=latest_run.report_html_path,
            report_json_path=latest_run.report_json_path,
            features=features
        )

    return MonitoringStatusResponse(
        status="DRIFT_DETECTED" if active_drift else "STABLE",
        latest_run=latest_resp,
        total_monitoring_runs=total_runs,
        active_dataset_drift=active_drift,
        drift_share=drift_share,
        total_prediction_records=total_preds,
        total_labeled_records=total_labels,
    )


@router.post("/monitoring/run", response_model=MonitoringRunResponse)
def trigger_drift_run(req: DriftRunRequest = DriftRunRequest(), db: Session = Depends(get_db)):
    """Trigger an Evidently AI / statistical drift detection run against production inferences."""
    try:
        return drift_service.run_drift_analysis(
            db=db,
            sample_limit=req.sample_limit,
            p_value_threshold=req.p_value_threshold,
            drift_share_threshold=req.drift_share_threshold,
            trigger_type=req.trigger_type,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error running drift analysis: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/monitoring/report/html/{run_id}")
def get_html_report(run_id: int, db: Session = Depends(get_db)):
    """Serve the generated Evidently HTML report directly in the browser."""
    run = db.query(MonitoringRun).filter(MonitoringRun.id == run_id).first()
    if not run or not run.report_html_path or not Path(run.report_html_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="HTML report file not found")
    return FileResponse(run.report_html_path, media_type="text/html")
