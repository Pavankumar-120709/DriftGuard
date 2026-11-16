"""Model Registry and Champion/Challenger management for DriftGuard."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models_store"
REGISTRY_METADATA_FILE = MODELS_DIR / "registry.json"


class ModelRegistryManager:
    """Manages active champion model, challengers, versions, and rollback capability."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or MODELS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "registry.json"
        self._ensure_registry()

    def _ensure_registry(self):
        if not self.metadata_file.exists():
            initial_data = {
                "champion_version": None,
                "previous_champion_version": None,
                "models": {},
                "promotion_history": []
            }
            with open(self.metadata_file, "w") as f:
                json.dump(initial_data, f, indent=2)

    def _load_metadata(self) -> Dict[str, Any]:
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def _save_metadata(self, data: Dict[str, Any]):
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        version: str,
        model_name: str,
        pipeline_obj: Any,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        feature_importance: List[Dict[str, Any]],
        mlflow_run_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save pipeline artifact and register metadata."""
        version_dir = self.storage_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = version_dir / "pipeline.joblib"
        joblib.dump(pipeline_obj, artifact_path)

        meta = self._load_metadata()
        model_entry = {
            "version": version,
            "model_name": model_name,
            "artifact_path": str(artifact_path),
            "metrics": metrics,
            "params": params,
            "feature_importance": feature_importance,
            "mlflow_run_id": mlflow_run_id,
            "dataset_fingerprint": dataset_fingerprint,
            "is_champion": False,
            "status": "REGISTERED",
        }
        meta["models"][version] = model_entry

        # If this is the very first registered model, automatically set as initial champion
        if meta["champion_version"] is None:
            meta["champion_version"] = version
            meta["models"][version]["is_champion"] = True
            meta["promotion_history"].append({
                "version": version,
                "promoted_at": pd.Timestamp.now().isoformat() if "pd" in globals() else "initial",
                "reason": "Initial model deployment",
                "previous_version": None
            })

        self._save_metadata(meta)
        logger.info(f"Registered model {model_name} version {version}")
        return model_entry

    def get_champion(self) -> Optional[Dict[str, Any]]:
        meta = self._load_metadata()
        champ_ver = meta.get("champion_version")
        if champ_ver and champ_ver in meta["models"]:
            return meta["models"][champ_ver]
        return None

    def load_champion_pipeline(self) -> Optional[Any]:
        champ = self.get_champion()
        if champ and Path(champ["artifact_path"]).exists():
            return joblib.load(champ["artifact_path"])
        return None

    def load_model_pipeline(self, version: str) -> Optional[Any]:
        meta = self._load_metadata()
        if version in meta["models"]:
            path = Path(meta["models"][version]["artifact_path"])
            if path.exists():
                return joblib.load(path)
        return None

    def evaluate_challenger_promotion(
        self,
        challenger_version: str,
        primary_metric: str = "val_roc_auc",
        min_improvement_threshold: float = 0.005,
    ) -> Tuple[bool, str]:
        """Validate if challenger model meets the business quality gate to supersede the champion.
        
        Strict Guard:
        - Prevents accidental promotion if challenger fails minimum baseline or underperforms current champion.
        """
        meta = self._load_metadata()
        champ_ver = meta.get("champion_version")

        if challenger_version not in meta["models"]:
            return False, f"Challenger version {challenger_version} not found in registry."

        challenger = meta["models"][challenger_version]
        challenger_score = challenger["metrics"].get(primary_metric, 0.0)

        # Baseline absolute quality gate: must have ROC-AUC >= 0.70 and F1 >= 0.50
        baseline_roc = challenger["metrics"].get("val_roc_auc", 0.0)
        baseline_f1 = challenger["metrics"].get("val_f1", 0.0)
        if baseline_roc < 0.70:
            return False, f"Challenger rejected: ROC-AUC {baseline_roc:.4f} is below mandatory baseline threshold 0.70."
        if baseline_f1 < 0.45:
            return False, f"Challenger rejected: F1 score {baseline_f1:.4f} is below mandatory baseline threshold 0.45."

        if not champ_ver or champ_ver not in meta["models"]:
            return True, "No active champion exists. Challenger promoted as initial champion."

        champion = meta["models"][champ_ver]
        champion_score = champion["metrics"].get(primary_metric, 0.0)

        margin = challenger_score - champion_score
        if margin >= min_improvement_threshold:
            return True, (
                f"Challenger approved: {primary_metric} improved from {champion_score:.4f} "
                f"to {challenger_score:.4f} (+{margin:.4f} >= threshold {min_improvement_threshold})."
            )
        else:
            return False, (
                f"Challenger rejected: {primary_metric} {challenger_score:.4f} did not exceed "
                f"champion {champion_score:.4f} by required improvement {min_improvement_threshold:.4f} (actual: {margin:+.4f})."
            )

    def promote_model(self, version: str, reason: str = "Retraining validation passed") -> bool:
        meta = self._load_metadata()
        if version not in meta["models"]:
            return False

        current_champ = meta.get("champion_version")
        if current_champ == version:
            return True

        if current_champ and current_champ in meta["models"]:
            meta["models"][current_champ]["is_champion"] = False

        meta["previous_champion_version"] = current_champ
        meta["champion_version"] = version
        meta["models"][version]["is_champion"] = True

        from datetime import datetime, timezone
        meta["promotion_history"].append({
            "version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "previous_version": current_champ
        })

        self._save_metadata(meta)
        logger.info(f"Promoted version {version} to champion (Previous: {current_champ})")
        return True

    def rollback_to_previous_champion(self) -> Tuple[bool, str]:
        """Rollback safely to the previous champion model version."""
        meta = self._load_metadata()
        prev_ver = meta.get("previous_champion_version")
        if not prev_ver or prev_ver not in meta["models"]:
            return False, "No previous champion version recorded for rollback."

        current_champ = meta.get("champion_version")
        success = self.promote_model(prev_ver, reason=f"Rollback from {current_champ}")
        if success:
            return True, f"Successfully rolled back champion to version {prev_ver}."
        return False, "Failed to rollback to previous champion."

    def list_models(self) -> List[Dict[str, Any]]:
        meta = self._load_metadata()
        return list(meta["models"].values())

    def get_promotion_history(self) -> List[Dict[str, Any]]:
        meta = self._load_metadata()
        return meta.get("promotion_history", [])
