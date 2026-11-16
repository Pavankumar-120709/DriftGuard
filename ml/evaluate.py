"""Comprehensive model evaluation module for DriftGuard."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_curve,
    brier_score_loss,
)

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    prefix: str = "val"
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # Core scalar metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.5

    # Precision-Recall AUC
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.0

    # Calibration / Brier score
    try:
        brier = float(brier_score_loss(y_true, y_proba))
    except Exception:
        brier = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    # Business impact calculation (Customer churn context):
    # Customer lifetime value saved (TP) vs Cost of retention intervention (TP + FP) vs Lost customers (FN)
    estimated_clv = 1500.0  # e.g. $1500 avg lifetime value
    retention_cost = 120.0  # e.g. $120 retention incentive
    revenue_saved = float(tp * (estimated_clv - retention_cost) - fp * retention_cost)

    metrics = {
        f"{prefix}_accuracy": round(acc, 4),
        f"{prefix}_precision": round(prec, 4),
        f"{prefix}_recall": round(rec, 4),
        f"{prefix}_f1": round(f1, 4),
        f"{prefix}_roc_auc": round(roc_auc, 4),
        f"{prefix}_pr_auc": round(pr_auc, 4),
        f"{prefix}_brier_score": round(brier, 4),
        f"{prefix}_true_negatives": int(tn),
        f"{prefix}_false_positives": int(fp),
        f"{prefix}_false_negatives": int(fn),
        f"{prefix}_true_positives": int(tp),
        f"{prefix}_estimated_net_benefit_usd": round(revenue_saved, 2),
    }

    return metrics


def get_curve_points(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_samples: int = 50
) -> Dict[str, Any]:
    """Extract downsampled ROC and PR curves for frontend charts and artifact logging."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        prec, rec, _ = precision_recall_curve(y_true, y_proba)

        indices_roc = np.linspace(0, len(fpr) - 1, min(num_samples, len(fpr))).astype(int)
        indices_pr = np.linspace(0, len(prec) - 1, min(num_samples, len(prec))).astype(int)

        roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in indices_roc]
        pr_points = [{"recall": round(float(rec[i]), 4), "precision": round(float(prec[i]), 4)} for i in indices_pr]

        return {
            "roc_curve": roc_points,
            "pr_curve": pr_points,
        }
    except Exception as e:
        logger.error(f"Error computing curves: {e}")
        return {"roc_curve": [], "pr_curve": []}
