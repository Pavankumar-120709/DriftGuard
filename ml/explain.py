"""Explainability module using SHAP for DriftGuard."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_feature_importance(
    model: Any,
    feature_names: List[str],
    X_sample: Optional[np.ndarray] = None,
    max_features: int = 15
) -> List[Dict[str, Any]]:
    """Extract global feature importance from tree-based or linear models with SHAP fallback."""
    importance_dict = {}

    # Check for native tree feature_importances_
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
        for name, imp in zip(feature_names, raw_importances):
            importance_dict[name] = float(imp)

    # Check for linear model coefficients
    elif hasattr(model, "coef_"):
        raw_coefs = np.abs(model.coef_[0])
        for name, coef in zip(feature_names, raw_coefs):
            importance_dict[name] = float(coef)

    # Try SHAP explainer if sample is provided
    elif X_sample is not None:
        try:
            import shap
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample[:100])
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap[:, 1]
            for name, val in zip(feature_names, mean_abs_shap):
                importance_dict[name] = float(val)
        except Exception as e:
            logger.warning(f"SHAP calculation skipped: {e}")

    # Fallback if empty
    if not importance_dict and feature_names:
        for name in feature_names:
            importance_dict[name] = 1.0 / len(feature_names)

    # Normalize and sort
    total = sum(importance_dict.values()) or 1.0
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:max_features]

    return [
        {
            "feature": k,
            "importance": round(float(v / total), 4),
            "raw_score": round(float(v), 4),
        }
        for k, v in sorted_items
    ]
