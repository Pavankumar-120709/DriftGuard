"""Evidently AI Data Drift detection service for DriftGuard."""

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.schema import PredictionLog, MonitoringRun, DriftMetric
from backend.app.schemas.monitoring import MonitoringRunResponse, FeatureDriftDetail

logger = logging.getLogger(__name__)


class DriftMonitoringService:
    def __init__(self):
        self.reports_dir = Path(settings.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.ref_data_path = Path(settings.REFERENCE_DATA_PATH)

    def _get_reference_data(self) -> pd.DataFrame:
        if self.ref_data_path.exists():
            return pd.read_csv(self.ref_data_path)
        # If not yet preprocessed, ingest telco data
        from ml.dataset import ingest_telco_data
        df = ingest_telco_data()
        from ml.validation import clean_and_impute_dataset
        df = clean_and_impute_dataset(df)
        self.ref_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.ref_data_path, index=False)
        return df

    def _get_production_data(self, db: Session, sample_limit: int = 500) -> Tuple[pd.DataFrame, List[int]]:
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.timestamp.desc())
            .limit(sample_limit)
            .all()
        )
        if not logs:
            return pd.DataFrame(), []

        rows = []
        log_ids = []
        for log in logs:
            row = dict(log.input_features)
            row["prediction"] = log.predicted_class
            row["prediction_probability"] = log.prediction_probability
            rows.append(row)
            log_ids.append(log.id)

        df = pd.DataFrame(rows)
        return df, log_ids

    def run_drift_analysis(
        self,
        db: Session,
        sample_limit: int = 500,
        p_value_threshold: float = 0.05,
        drift_share_threshold: float = 0.30,
        trigger_type: str = "MANUAL"
    ) -> MonitoringRunResponse:
        """Execute comprehensive Evidently / statistical drift detection comparing reference vs production."""
        ref_df = self._get_reference_data()
        prod_df, log_ids = self._get_production_data(db, sample_limit=sample_limit)

        if prod_df.empty:
            raise ValueError("No production prediction logs found to analyze. Submit predictions first.")

        # Align columns
        eval_features = [
            c for c in ref_df.columns
            if c not in ["customerID", "Churn"] and c in prod_df.columns
        ]

        run_timestamp = datetime.datetime.utcnow()
        run_id_tag = int(run_timestamp.timestamp())
        html_report_path = self.reports_dir / f"drift_report_{run_id_tag}.html"
        json_report_path = self.reports_dir / f"drift_report_{run_id_tag}.json"

        # Attempt Evidently Report execution
        evidently_success = False
        evidently_json = {}
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            # Create clean copies for Evidently
            ref_sub = ref_df[eval_features].copy()
            prod_sub = prod_df[eval_features].copy()

            # Ensure numeric conversion
            for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                if col in ref_sub.columns:
                    ref_sub[col] = pd.to_numeric(ref_sub[col], errors="coerce").fillna(0)
                if col in prod_sub.columns:
                    prod_sub[col] = pd.to_numeric(prod_sub[col], errors="coerce").fillna(0)

            report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
            report.run(reference_data=ref_sub, current_data=prod_sub)
            report.save_html(str(html_report_path))
            evidently_json = report.as_dict()
            with open(json_report_path, "w") as f:
                json.dump(evidently_json, f, indent=2)
            evidently_success = True
            logger.info("Evidently AI report generated successfully.")
        except Exception as e:
            logger.warning(f"Evidently automated preset exception ({e}), generating statistical drift analysis.")

        # Compute feature-level drift statistics (KS-test for numerical, Chi-square for categorical)
        feature_results: List[FeatureDriftDetail] = []
        drifted_count = 0

        for col in eval_features:
            is_num = False
            ref_series = ref_df[col].dropna()
            prod_series = prod_df[col].dropna()

            if pd.api.types.is_numeric_dtype(ref_series):
                is_num = True
                ref_num = pd.to_numeric(ref_series, errors="coerce").dropna()
                prod_num = pd.to_numeric(prod_series, errors="coerce").dropna()

                if len(prod_num) > 5 and len(ref_num) > 5:
                    ks_stat, p_val = ks_2samp(ref_num, prod_num)
                    is_drifted = bool(p_val < p_value_threshold)
                    drift_score = float(ks_stat)
                    stat_test = "Kolmogorov-Smirnov"
                else:
                    drift_score, p_val, is_drifted, stat_test = 0.0, 1.0, False, "Insufficient Data"

                ref_dist = {
                    "mean": round(float(ref_num.mean()), 2),
                    "std": round(float(ref_num.std()), 2),
                    "median": round(float(ref_num.median()), 2),
                }
                prod_dist = {
                    "mean": round(float(prod_num.mean()), 2),
                    "std": round(float(prod_num.std()), 2),
                    "median": round(float(prod_num.median()), 2),
                }

            else:
                # Categorical
                ref_counts = ref_series.value_counts(normalize=True)
                prod_counts = prod_series.value_counts(normalize=True)

                all_cats = list(set(ref_counts.index).union(set(prod_counts.index)))
                ref_freq = [ref_counts.get(cat, 0.001) for cat in all_cats]
                prod_freq = [prod_counts.get(cat, 0.001) for cat in all_cats]

                # Total variation distance / drift score
                tvd = 0.5 * sum(abs(p - r) for p, r in zip(prod_freq, ref_freq))
                drift_score = round(float(tvd), 4)

                # Chi-Square test approximation
                try:
                    c_table = pd.crosstab(
                        pd.concat([pd.Series(["ref"]*len(ref_series)), pd.Series(["prod"]*len(prod_series))]),
                        pd.concat([ref_series, prod_series])
                    )
                    chi2, p_val, _, _ = chi2_contingency(c_table)
                except Exception:
                    p_val = 1.0 if drift_score < 0.15 else 0.01

                is_drifted = bool(drift_score > 0.15 or (p_val is not None and p_val < p_value_threshold))
                stat_test = "TVD / Chi-Square"

                ref_dist = {k: round(float(v), 3) for k, v in ref_counts.head(5).to_dict().items()}
                prod_dist = {k: round(float(v), 3) for k, v in prod_counts.head(5).to_dict().items()}

            if is_drifted:
                drifted_count += 1

            feature_results.append(FeatureDriftDetail(
                feature_name=col,
                feature_type="numerical" if is_num else "categorical",
                drift_score=round(drift_score, 4),
                p_value=round(float(p_val), 4) if p_val is not None else None,
                is_drifted=is_drifted,
                stat_test=stat_test,
                reference_distribution=ref_dist,
                current_distribution=prod_dist
            ))

        total_features = len(eval_features)
        drift_share = round(float(drifted_count / total_features), 4) if total_features > 0 else 0.0
        dataset_drift_detected = bool(drift_share >= drift_share_threshold)

        # Fallback HTML report generation if evidently HTML didn't create
        if not html_report_path.exists():
            self._generate_static_drift_html(
                html_report_path,
                dataset_drift_detected,
                drift_share,
                drift_share_threshold,
                len(ref_df),
                len(prod_df),
                feature_results
            )

        # Persist to database
        db_run = MonitoringRun(
            timestamp=run_timestamp,
            run_type=trigger_type,
            reference_rows=len(ref_df),
            current_rows=len(prod_df),
            dataset_drift_detected=dataset_drift_detected,
            drift_share=drift_share,
            drifted_features_count=drifted_count,
            total_features_count=total_features,
            p_value_threshold=p_value_threshold,
            drift_share_threshold=drift_share_threshold,
            report_html_path=str(html_report_path),
            report_json_path=str(json_report_path) if json_report_path.exists() else None,
            summary_metrics={
                "dataset_drift_detected": dataset_drift_detected,
                "drift_share": drift_share,
                "drifted_features_count": drifted_count,
            }
        )
        db.add(db_run)
        db.flush()

        # Add feature metrics
        for feat in feature_results:
            db_metric = DriftMetric(
                monitoring_run_id=db_run.id,
                feature_name=feat.feature_name,
                feature_type=feat.feature_type,
                drift_score=feat.drift_score,
                p_value=feat.p_value,
                is_drifted=feat.is_drifted,
                stat_test=feat.stat_test,
                reference_distribution=feat.reference_distribution,
                current_distribution=feat.current_distribution,
            )
            db.add(db_metric)

        db.commit()
        db.refresh(db_run)

        return MonitoringRunResponse(
            id=db_run.id,
            timestamp=db_run.timestamp.isoformat(),
            run_type=db_run.run_type,
            reference_rows=db_run.reference_rows,
            current_rows=db_run.current_rows,
            dataset_drift_detected=db_run.dataset_drift_detected,
            drift_share=db_run.drift_share,
            drifted_features_count=db_run.drifted_features_count,
            total_features_count=db_run.total_features_count,
            p_value_threshold=db_run.p_value_threshold,
            drift_share_threshold=db_run.drift_share_threshold,
            report_html_path=db_run.report_html_path,
            report_json_path=db_run.report_json_path,
            features=feature_results
        )

    def _generate_static_drift_html(
        self,
        output_path: Path,
        drift_detected: bool,
        drift_share: float,
        threshold: float,
        ref_rows: int,
        curr_rows: int,
        features: List[FeatureDriftDetail]
    ):
        """Generate a polished HTML drift report."""
        badge_color = "#ef4444" if drift_detected else "#10b981"
        badge_text = "DRIFT DETECTED" if drift_detected else "DATA STABLE"

        rows_html = ""
        for f in features:
            status_color = "#ef4444" if f.is_drifted else "#10b981"
            status_label = "DRIFTED" if f.is_drifted else "STABLE"
            rows_html += f"""
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 12px; font-weight: 600;">{f.feature_name}</td>
                <td style="padding: 12px; color: #94a3b8;">{f.feature_type}</td>
                <td style="padding: 12px; color: #cbd5e1;">{f.stat_test}</td>
                <td style="padding: 12px; font-family: monospace;">{f.drift_score:.4f}</td>
                <td style="padding: 12px; font-family: monospace;">{f.p_value if f.p_value is not None else 'N/A'}</td>
                <td style="padding: 12px;"><span style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">{status_label}</span></td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>DriftGuard Data Drift Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ background: {badge_color}; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 13px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #0f172a; padding: 12px; color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }}
        .metric-box {{ background: #0f172a; padding: 16px; border-radius: 6px; border: 1px solid #334155; }}
        .metric-val {{ font-size: 24px; font-weight: bold; color: #38bdf8; margin-top: 4px; }}
        .metric-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 24px;">DriftGuard Monitoring Report</h1>
                <p style="color: #94a3b8; margin: 4px 0 0 0;">Dataset-level and feature-level statistical drift detection</p>
            </div>
            <div class="badge">{badge_text}</div>
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Drift Share</div>
                <div class="metric-val">{drift_share * 100:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Drift Threshold</div>
                <div class="metric-val">{threshold * 100:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Reference Rows</div>
                <div class="metric-val">{ref_rows}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Production Rows</div>
                <div class="metric-val">{curr_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2 style="font-size: 18px; margin-top: 0;">Feature Drift Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Test</th>
                    <th>Drift Score</th>
                    <th>P-Value</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


drift_service = DriftMonitoringService()
