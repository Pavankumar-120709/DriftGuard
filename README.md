# DriftGuard — Enterprise End-to-End MLOps Platform with Automated Drift Detection & Controlled Retraining

[![CI/CD Pipeline](https://github.com/driftguard/driftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/driftguard/driftguard/actions)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14.2+-black.svg?logo=next.js&logoColor=white)](https://nextjs.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Evidently AI](https://img.shields.io/badge/Evidently%20AI-Distribution%20Monitoring-orange.svg)](https://evidentlyai.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0+-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary & Objective

**DriftGuard** is an industrial-grade, full-lifecycle MLOps platform engineered to predict, monitor, and mitigate **Customer Churn**. Real-world machine learning systems rarely fail because of poor initial model architecture—they fail silently in production due to unmonitored feature shifts, changing consumer behavior, silent schema alterations, and uncalibrated retraining loops.

DriftGuard implements a **closed-loop feedback architecture** that bridges:
1. **Reproducible Model Training & Experiment Tracking** via MLflow.
2. **Sub-15ms Low-Latency Inference Serving** with structured logging via FastAPI.
3. **Continuous Statistical Data Drift Detection** using Evidently AI and rigorous two-sample statistical hypothesis tests (Kolmogorov-Smirnov & Chi-Square).
4. **Delayed-Label Ground Truth Pairing**, isolating input distribution shifts from true model degradation.
5. **Controlled, Automated Retraining with Champion/Challenger Quality Gates**, preventing sub-standard models from reaching production while enabling atomic single-click rollback.
6. **A Warm Minimalist, Editorial UI** inspired by Claude's design language with full dark/light theme support.

---

## 2. System Architecture & Lifecycle Diagram

DriftGuard enforces strict separation of concerns across Data, Training, Tracking, Serving, Monitoring, and Governance layers.

```mermaid
flowchart TD
    subgraph Data Layer
        A[Canonical Telco Dataset / External Ingestion] --> B[Data Validation & Cleaning]
        B --> C[Domain Feature Engineering]
        C --> D[Leak-Free Stratified Split]
    end

    subgraph MLOps & Experimentation
        D --> E[Multi-Model Benchmark: LR, RF, XGBoost]
        E --> F[Hyperparameter Tuning & Evaluation]
        F --> G[MLflow Tracking Server: Run, Artifacts, Fingerprints]
        G --> H[Model Registry: Version Catalog & Champion Selection]
    end

    subgraph Production Inference
        H --> I[FastAPI High-Performance Engine]
        CLIENT[Client Apps / SDKs] -->|POST /predict| I
        I --> J[(PostgreSQL / SQLite: Prediction Logs)]
    end

    subgraph Drift Monitoring & Telemetry
        J --> K[Evidently AI Engine]
        REF[(Reference Baseline Data)] --> K
        K --> L[Two-Sample KS & Chi-Square Tests]
        L --> M{Dataset Drift Detected?}
        M -->|Drift Share >= 30%| N[Retraining Candidate Flag]
        M -->|Stable| O[Log Monitoring Run]
    end

    subgraph Delayed-Label Performance
        LABELS[Delayed Customer Churn Outcomes] -->|POST /labels| P[Ground Truth Pairing Service]
        P --> Q[Calculate True Production Metrics]
        Q --> R{Performance Drop > 5% vs Validation?}
        R -->|Degradation Alert| N
        R -->|Normal| O
    end

    subgraph Controlled Retraining & Quality Gates
        N --> S[Atomic Concurrency Lock]
        S --> T[Validate Combined Reference + Production Data]
        T --> U[Train New Challenger Pipeline]
        U --> V[MLflow Run Logging & Fingerprinting]
        V --> W{Champion / Challenger Gate}
        W -->|Challenger improves >= +0.5% ROC-AUC & passes baseline| X[Promote Challenger to Champion]
        W -->|Fails Margin or Baseline Floor| Y[Reject Challenger & Retain Champion]
        X --> Z[Dynamic Production Reload & Audit Log]
        Y --> AA[Audit Log Reason & Alert]
        Z -.->|Safety Fallback| AB[Instant Rollback to Previous Champion]
    end
```

---

## 3. Technology Stack

| Layer | Technology | Key Capabilities & Rationale |
| :--- | :--- | :--- |
| **Machine Learning** | **Python 3.11–3.13**, **Scikit-learn**, **XGBoost**, **SHAP** | Pipeline abstraction, gradient boosted decision trees, class weight balancing, and global/local game-theoretic feature importance. |
| **Experiment Tracking** | **MLflow 2.18+** | Centralized metric logging, ROC/PR curves, hyperparameter tracking, model artifacts (`.joblib`), and dataset content SHA-256 fingerprinting. |
| **Serving & API** | **FastAPI 0.115+**, **Pydantic v2**, **Uvicorn** | Asynchronous execution, strict schema contract validation, sub-15ms prediction latency, OpenAPI/Swagger interactive documentation. |
| **Drift Monitoring** | **Evidently AI 0.7+**, **Scipy Stats** | Non-parametric two-sample Kolmogorov-Smirnov test (numerical) and Contingency Chi-Square/TVD (categorical). Standalone interactive HTML & JSON reports. |
| **Persistence** | **PostgreSQL 15** / **SQLAlchemy 2.0** | Relational schema with foreign keys, JSONB fields, audit tables (`PredictionLog`, `GroundTruthLabel`, `MonitoringRun`, `DriftMetric`, `ModelPromotion`, `RetrainingJob`). |
| **Frontend UI** | **Next.js 14**, **TypeScript**, **Tailwind CSS**, **Recharts** | Warm minimalist, editorial Claude-inspired design system with light/dark theme switching, real-time gauges, and comparative analytics. |
| **DevOps & Testing** | **Docker**, **Docker Compose**, **GitHub Actions**, **Pytest** | Multi-container orchestration, health checks, automated linting, and 21-point unit/integration test coverage. |

---

## 4. Machine Learning Pipeline Specification

### 4.1 Benchmark Dataset
DriftGuard defaults to the canonical **IBM Telco Customer Churn** dataset (7,043 customer accounts, 19 predictive features, 1 target label `Churn`). If remote repositories are unreachable, an integrated statistical generator synthesizes an identical distribution with accurate correlations (e.g. Month-to-month contracts and Fiber Optic services correlate with higher churn; tenure correlates negatively).

### 4.2 Data Ingestion & Validation Gate (`ml/validation.py`)
Incoming datasets pass through strict data validation:
1. **Schema Integrity**: Verifies presence of all 19 mandatory demographic, account, and service fields.
2. **Duplicate Detection**: Identifies and drops duplicate customer records based on `customerID`.
3. **Missing Value Imputation**: Dynamically converts white-space strings in `TotalCharges` to float and imputes missing numeric values with the column median; categoricals with column mode.
4. **IQR Outlier Analysis**: Calculates Interquartile Range ($Q_{75} - Q_{25}$) fences ($[Q_{25} - 1.5 \times \text{IQR}, Q_{75} + 1.5 \times \text{IQR}]$) to surface distribution anomalies without silently distorting training data.

### 4.3 Feature Engineering (`ml/pipeline.py`)
A custom scikit-learn transformer (`ChurnFeatureEngineer`) creates domain-specific predictors:
- `tenure_years`: Normalized customer lifetime duration ($\text{tenure} / 12$).
- `monthly_to_total_ratio`: Monthly charges relative to accumulated spend ($\text{MonthlyCharges} / (\text{TotalCharges} + \epsilon)$).
- `total_services_subscribed`: Count of active subscription add-ons (Tech Support, Online Security, Backup, Streaming, etc.).
- `has_tech_or_security`: Binary indicator for whether retention-critical support services are active.
- `is_high_risk_combo`: High-risk interaction flag ($\text{Contract == 'Month-to-month'} \land \text{Internet == 'Fiber optic'}$).

### 4.4 Leak-Free Splitting & Preprocessing
To strictly prevent data leakage:
- Stratified splitting divides data into **Train (70%)**, **Validation (15%)**, and **Holdout Test (15%)**.
- Preprocessing transformers (`StandardScaler` for numeric columns, `OneHotEncoder(handle_unknown='ignore')` for categoricals) are **fit only on the training split**, then used to transform validation and test sets.
- Class weights are dynamically computed ($w_c = \frac{N}{2 \times N_c}$) to handle the natural ~73:27 churn class imbalance.

### 4.5 Multi-Model Comparison & Baseline Results
Three distinct algorithm families are trained and compared under identical conditions:
1. **Logistic Regression (L2 regularized)**: High-interpretability linear baseline.
2. **Random Forest Classifier**: Non-linear ensemble with bootstrap aggregation (150 estimators, max depth 8).
3. **XGBoost Classifier**: Extreme gradient boosting with tree subsampling and column subsampling (150 estimators, learning rate 0.05).

#### Empirical Validation Metrics:
| Architecture | Validation ROC-AUC | Validation F1 | Test ROC-AUC | Test F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Active Champion)** | **0.8378** | **0.6266** | **0.8512** | **0.6295** | ~1.09s |
| **Random Forest** | 0.8364 | 0.6206 | 0.8492 | 0.6371 | ~1.85s |
| **Logistic Regression** | 0.8364 | 0.6171 | 0.8593 | 0.6391 | ~0.35s |

*Champion Selection Rationale*: XGBoost was selected as champion because customer churn intervention is cost-sensitive. ROC-AUC balances ranking quality across threshold cutoffs, and F1 maximizes actionable precision and recall.

---

## 5. Serving & REST API Reference

FastAPI serves predictions with Pydantic v2 validation, sub-15ms response latency, and automatic database logging.

### 5.1 Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Evaluates a single customer profile, logs inference features, returns probability and model version. |
| `POST` | `/predict/batch` | High-throughput batch inference for corporate billing pipelines. |
| `GET` | `/health` | Health check for Docker, Kubernetes, and load balancers. Returns active champion version. |
| `GET` | `/model/info` | Returns active champion metadata, hyperparameters, dataset fingerprint, and SHAP drivers. |
| `GET` | `/model/metrics` | Returns champion offline validation and test metrics. |
| `GET` | `/models` | Catalogs all registered models in the local and MLflow model registry. |
| `GET` | `/models/compare` | Evaluates active champion against any candidate challenger version. |
| `GET` | `/monitoring/status` | Current drift telemetry health, total predictions logged, and active drift flags. |
| `POST` | `/monitoring/run` | Triggers on-demand Evidently AI distribution drift analysis. |
| `GET` | `/monitoring/report/html/{id}` | Serves the generated standalone Evidently HTML diagnostic report. |
| `POST` | `/labels` | Ingests delayed ground-truth customer churn outcomes. |
| `GET` | `/labels/performance` | Returns true production classification metrics calculated from paired outcomes. |
| `POST` | `/retrain/trigger` | Executes controlled retraining under the atomic execution lock. |
| `POST` | `/retrain/rollback` | Reverts the active champion to the previous version with instant failover. |
| `GET` | `/retrain/history` | Audit log of all retraining attempts, parameters, and promotion/rejection decisions. |

### 5.2 Example Prediction Request & Response

#### Request: `POST /predict`
```json
{
  "customerID": "CUST-98214",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 94.85,
  "TotalCharges": 284.55
}
```

#### Response (`200 OK` in 12ms):
```json
{
  "customer_id": "CUST-98214",
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.8142,
  "model_version": "xgboost-v1794858045",
  "latency_ms": 11.45,
  "timestamp": "2026-11-17T01:10:45.123456",
  "prediction_id": 142
}
```

---

## 6. Drift Detection & Performance Telemetry

DriftGuard enforces a foundational MLOps principle: **Data Drift is fundamentally distinct from Concept Drift.**

$$\underbrace{P(X)}_{\text{Data Drift}} \quad \neq \quad \underbrace{P(Y \mid X)}_{\text{Concept Drift / Model Degradation}}$$

```
+-----------------------------------------------------------------------------------+
|                                  MONITORING MATRIX                                |
+------------------------------------+----------------------------------------------+
| DATA DRIFT DETECTED: YES           | DATA DRIFT DETECTED: NO                      |
| PERFORMANCE DROP: NO               | PERFORMANCE DROP: YES                        |
| -> Features shifted, but model     | -> Silent concept drift (e.g. competitor     |
|    decision boundaries hold.       |    launched aggressive price cut). Retrain!  |
+------------------------------------+----------------------------------------------+
| DATA DRIFT DETECTED: YES           | DATA DRIFT DETECTED: NO                      |
| PERFORMANCE DROP: YES              | PERFORMANCE DROP: NO                         |
| -> Severe covariate shift causing  | -> Ideal operational state. Pipeline stable. |
|    model failure. Urgent Retrain!  |                                              |
+------------------------------------+----------------------------------------------+
```

### 6.1 Statistical Tests Applied
1. **Numerical Features** (`tenure`, `MonthlyCharges`, `TotalCharges`):
   - Two-Sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`).
   - Null hypothesis $H_0$: Production distribution $F_{\text{prod}}(x)$ equals baseline $F_{\text{ref}}(x)$.
   - Rejects $H_0$ and flags drift when $p\text{-value} < 0.05$.
2. **Categorical Features** (`Contract`, `InternetService`, `PaymentMethod`, etc.):
   - Total Variation Distance (TVD) and Contingency Chi-Square test (`chi2_contingency`).
   - Flags drift when frequency divergence exceeds threshold or $p\text{-value} < 0.05$.
3. **Dataset-Level Drift**:
   - Evaluates the share of drifted features:
   $$\text{Drift Share} = \frac{\sum_{i=1}^M \mathbb{I}(\text{Feature}_i \text{ drifted})}{M}$$
   - When $\text{Drift Share} \ge 30\%$, the dataset is flagged as drifted.

### 6.2 Delayed-Label Concept Monitoring
In churn prediction, true labels arrive with delay (e.g. 30–90 days after prediction). Clients submit verified labels via `POST /labels`. DriftGuard joins labels with prior inferences in `PredictionLog` to compute true operational metrics:
- **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**, and **Brier Calibration Score**.
- Automatically triggers a **Degradation Alert** when production ROC-AUC drops by $> 5\%$ from the validation baseline.

---

## 7. Controlled Automated Retraining & Quality Gates

Retraining is protected by strict engineering invariants to prevent production regressions.

```
DRIFT DETECTED or PERFORMANCE DROP or MANUAL TRIGGER
                     |
                     v
          [ATOMIC CONCURRENCY LOCK]  <-- Prevents concurrent training conflicts
                     |
                     v
           [DATA VALIDATION GATE]   <-- Verifies schema, removes duplicates
                     |
                     v
       [MULTI-MODEL RE-BENCHMARK]   <-- Trains LR, RF, XGBoost on combined data
                     |
                     v
      [MLFLOW LOGGING & ARTIFACTS]  <-- Logs runs, curves, and fingerprints
                     |
                     v
      [CHAMPION / CHALLENGER GATE]  <-- Evaluates candidate against champion
                     |
       +-------------+-------------+
       |                           |
       v                           v
[PASSES QUALITY GATES]    [FAILS MARGIN OR BASELINE]
  - ROC-AUC >= 0.70         - Delta < +0.005 ROC-AUC
  - F1 >= 0.45              - OR fails baseline floor
  - Delta >= +0.005 ROC-AUC        |
       |                           v
       v                  [REJECT CHALLENGER]
[PROMOTE TO CHAMPION]     - Preserve active champion
  - Update Registry       - Audit rejection decision
  - Hot Reload API
       |
       v (Emergency Safety)
[INSTANT ROLLBACK READY]  <-- Single-click revert to previous champion
```

### 7.1 Quality Gate Rules
1. **Absolute Baseline Floor**: A challenger model must achieve $\text{ROC-AUC} \ge 0.70$ and $\text{F1} \ge 0.45$. Any model failing this floor is rejected immediately.
2. **Margin Improvement Gate**: The challenger must exceed the active champion by at least a configurable margin:
   $$\text{ROC-AUC}_{\text{challenger}} - \text{ROC-AUC}_{\text{champion}} \ge \Delta_{\text{min}} \quad (\text{default: } +0.005)$$
3. **Atomic Concurrency Protection**: Retraining jobs acquire an atomic thread lock (`threading.Lock`). Duplicate or overlapping triggers return `409 Conflict`.
4. **Audit Trail**: Every decision (`PROMOTED` or `REJECTED`), along with the underlying reason, metric deltas, and timestamp, is recorded in the `model_promotions` table.

---

## 8. Frontend UI & Visual Identity

The user interface is designed with a **Claude-inspired Warm Minimalist aesthetic**:
- **Design Language**: Warm minimalism, editorial typography, generous whitespace, and subtle 1px borders instead of heavy drop shadows.
- **Palette**: Warm parchment background (`#F5F1E9` light / `#191918` dark), warm surface (`#FFFCF6` / `#242320`), terracotta accents (`#D97757` / `#E38B6D`), and sage/amber/crimson status indicators.
- **Theme Switching**: Functional Light/Dark theme switcher with local storage persistence and OS preference synchronization.

### 8.1 The 6 Views
1. **Overview (`/`)**: High-level observability dashboard, active champion summary, drift alert banner, production volume, and interactive MLOps lifecycle flow.
2. **Model Registry (`/models`)**: Model catalog, champion/challenger comparison card with score margins, and audit history.
3. **Drift Monitoring (`/drift`)**: Feature drift table with search filtering, KS/Chi-Square statistics, distribution comparison, and direct link to the Evidently HTML report.
4. **Prediction Playground (`/playground`)**: Interactive inference form with pre-loaded customer archetypes (High Risk Churner, Loyal Customer, Fiber Drift Profile), real-time probability gauge, and delayed-label submission.
5. **Retraining Pipeline (`/retraining`)**: Retraining trigger controls, improvement threshold configuration, safety invariant explainer cards, audit trail, and instant rollback button.
6. **Analytics & Financials (`/analytics`)**: Recharts graphs comparing Validation vs. Production metrics, SHAP global feature importances, and confusion matrix with financial net benefit calculation.

---

## 9. Database Schema & Data Models

DriftGuard utilizes **PostgreSQL** in production (with automatic **SQLite** fallback for local development).

```mermaid
erDiagram
    PredictionLog ||--o| GroundTruthLabel : "pairs with"
    MonitoringRun ||--o{ DriftMetric : "contains"
    
    PredictionLog {
        int id PK
        string customer_id
        timestamp timestamp
        string model_version
        json input_features
        int predicted_class
        float prediction_probability
        float latency_ms
    }
    
    GroundTruthLabel {
        int id PK
        int prediction_log_id FK
        string customer_id
        int actual_churn
        timestamp labeled_at
        string source
    }
    
    MonitoringRun {
        int id PK
        timestamp timestamp
        string run_type
        int reference_rows
        int current_rows
        boolean dataset_drift_detected
        float drift_share
        int drifted_features_count
        int total_features_count
        string report_html_path
        string report_json_path
    }
    
    DriftMetric {
        int id PK
        int monitoring_run_id FK
        string feature_name
        string feature_type
        float drift_score
        float p_value
        boolean is_drifted
        string stat_test
        json reference_distribution
        json current_distribution
    }
    
    RetrainingJob {
        int id PK
        timestamp triggered_at
        timestamp completed_at
        string trigger_reason
        string status
        string challenger_version
        string champion_version
        string decision
        text decision_reason
        json training_metrics
    }
    
    ModelPromotion {
        int id PK
        timestamp timestamp
        string promoted_version
        string previous_version
        string primary_metric
        float challenger_score
        float champion_score
        string decision
        text reason
    }
```

---

## 10. Financial Impact & Business Value Model

In subscription businesses, the ROI of machine learning depends on balancing the value of saved customers against the cost of retention incentives.

$$\text{Net Protected Value} = (\text{TP} \times (\text{CLV} - C_{\text{retention}})) - (\text{FP} \times C_{\text{retention}})$$

- **Customer Lifetime Value ($\text{CLV}$)**: Estimated at **$1,500.00** per retained customer.
- **Cost of Retention Campaign ($C_{\text{retention}}$)**: Estimated at **$120.00** per intervention.
- **True Positives ($\text{TP}$)**: Customers correctly identified as high churn risk and successfully saved.
- **False Positives ($\text{FP}$)**: Loyal customers needlessly targeted with discounted offers.
- **False Negatives ($\text{FN}$)**: Lost customers representing $1,500 in lost revenue.

*DriftGuard's active champion model delivers an estimated **+$279,720.00** in net financial benefit on the test cohort alone.*

---

## 11. Quick Start Guide

### Option A: Multi-Container Docker Stack (Recommended)

Run the full platform—PostgreSQL 15, MLflow Tracking Server, FastAPI Backend, and Next.js Frontend:

```bash
# 1. Clone repository
git clone https://github.com/driftguard/driftguard.git
cd driftguard

# 2. Launch full stack
docker compose up --build
```

#### Access Points:
- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking Server**: [http://localhost:5000](http://localhost:5000)
- **Backend Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### Option B: Local Development Setup

#### 1. Backend & ML Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\activate
# Or Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run initial training pipeline (trains baseline champion and initializes registry)
python -m ml.train

# Start FastAPI server
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### 2. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Start Next.js development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 12. Automated Test Suite

DriftGuard includes a comprehensive 21-point test suite covering data validation, preprocessing leak prevention, model evaluation, API schemas, Evidently drift analysis, delayed label ingestion, and the retraining lifecycle.

```bash
# Execute pytest suite
pytest tests/ -v
```

### Test Suite Execution Matrix:
```
tests/test_api.py::test_health_endpoint PASSED                           [  4%]
tests/test_api.py::test_model_info_endpoint PASSED                       [  9%]
tests/test_api.py::test_model_metrics_endpoint PASSED                    [ 14%]
tests/test_api.py::test_predict_endpoint_valid PASSED                    [ 19%]
tests/test_api.py::test_predict_endpoint_invalid_inputs PASSED           [ 23%]
tests/test_api.py::test_batch_prediction_endpoint PASSED                 [ 28%]
tests/test_data_pipeline.py::test_synthetic_data_generation PASSED       [ 33%]
tests/test_data_pipeline.py::test_data_validation_valid PASSED           [ 38%]
tests/test_data_pipeline.py::test_data_validation_missing_columns PASSED [ 42%]
tests/test_data_pipeline.py::test_clean_and_impute_dataset PASSED        [ 47%]
tests/test_data_pipeline.py::test_feature_engineer_transformer PASSED    [ 52%]
tests/test_data_pipeline.py::test_leak_free_split PASSED                 [ 57%]
tests/test_data_pipeline.py::test_class_weights_computation PASSED       [ 61%]
tests/test_drift_monitoring.py::test_drift_analysis_with_production_logs PASSED [ 66%]
tests/test_model_training.py::test_evaluate_classifier PASSED            [ 71%]
tests/test_model_training.py::test_curve_points_computation PASSED       [ 76%]
tests/test_model_training.py::test_compute_dataset_fingerprint PASSED    [ 80%]
tests/test_model_training.py::test_train_and_evaluate_models_run PASSED  [ 85%]
tests/test_retraining_lifecycle.py::test_delayed_label_ingestion_and_performance PASSED [ 90%]
tests/test_retraining_lifecycle.py::test_retraining_quality_gate_rejection PASSED [ 95%]
tests/test_retraining_lifecycle.py::test_retraining_and_rollback_lifecycle PASSED [100%]

============================== 21 passed in 16.23s ==============================
```

---

## 13. Continuous Integration & Deployment (CI/CD)

The GitHub Actions workflow (`.github/workflows/ci.yml`) enforces quality standards on every pull request and push to `main`:
1. **Flake8 Linting**: Checks for syntax errors and undefined symbols.
2. **Backend Unit & Integration Testing**: Runs the complete pytest suite against a live PostgreSQL 15 test service.
3. **Frontend Static Analysis & Build**: Executes `npm run build` to verify TypeScript typing and Next.js page generation.
4. **Docker Image Build Validation**: Validates container builds for both backend and frontend.

---

## 14. Environment Variables Configuration

Copy `.env.example` to `.env` to customize settings:

```ini
# Application
APP_NAME=DriftGuard
APP_ENV=development
API_PORT=8000
API_HOST=0.0.0.0

# Database
# Production PostgreSQL:
# DATABASE_URL=postgresql://driftguard:driftguard_secure_password@postgres:5432/driftguard_db
# Local SQLite fallback:
DATABASE_URL=sqlite:///./driftguard.db

# MLflow
MLFLOW_TRACKING_URI=sqlite:///./mlruns.db
MLFLOW_EXPERIMENT_NAME=customer-churn-prediction

# Evidently Monitoring Thresholds
DRIFT_SHARE_THRESHOLD=0.30
P_VALUE_THRESHOLD=0.05
REFERENCE_DATA_PATH=./data/processed/reference_data.csv

# Retraining Gates
RETRAINING_DRIFT_THRESHOLD=0.35
RETRAINING_PERFORMANCE_DROP_THRESHOLD=0.05
MIN_RETRAINING_SAMPLES=50
AUTO_RETRAIN_ENABLED=true
```

---

## 15. Repository Structure

```
DriftGuard/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions CI/CD pipeline
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI router endpoints
│   │   │   ├── predict.py       # /predict and /predict/batch endpoints
│   │   │   ├── monitoring.py    # /monitoring/status, /monitoring/run, HTML reports
│   │   │   ├── retraining.py    # /retrain/trigger, /retrain/rollback, history
│   │   │   ├── models_api.py    # /health, /model/info, /models, /models/compare
│   │   │   └── labels.py        # /labels, /labels/performance
│   │   ├── models/              # SQLAlchemy database ORM schema
│   │   │   └── schema.py        # PredictionLog, MonitoringRun, RetrainingJob, etc.
│   │   ├── schemas/             # Pydantic v2 request/response contracts
│   │   ├── services/            # Core business & MLOps service logic
│   │   │   ├── inference_service.py   # Latency logging & model prediction
│   │   │   ├── drift_service.py       # Evidently AI & KS/Chi-Square testing
│   │   │   ├── performance_service.py # Delayed-label ground truth pairing
│   │   │   └── retraining_service.py  # Concurrency lock & quality gates
│   │   ├── config.py            # Pydantic BaseSettings environment config
│   │   ├── database.py          # SQLAlchemy session & table init
│   │   └── main.py              # FastAPI application entrypoint & lifecycle
├── data/
│   ├── raw/                     # Raw Telco Customer Churn benchmark
│   └── processed/               # Reference baseline dataset for Evidently
├── frontend/                    # Next.js 14 + TypeScript + Tailwind CSS UI
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root layout with theme context
│   │   │   ├── page.tsx         # Overview observability dashboard
│   │   │   ├── models/page.tsx  # Model Registry & Champion/Challenger
│   │   │   ├── drift/page.tsx   # Evidently AI feature drift diagnostics
│   │   │   ├── playground/page.tsx # Live inference & delayed-label pairing
│   │   │   ├── retraining/page.tsx # Controlled retraining & rollback
│   │   │   └── analytics/page.tsx  # Recharts curves & confusion matrix
│   │   ├── components/
│   │   │   ├── Navbar.tsx       # Minimalist navigation & status pill
│   │   │   └── ThemeToggle.tsx  # Light/Dark mode switcher
│   │   └── lib/
│   │       └── api.ts           # Typed API client
│   ├── package.json
│   └── tailwind.config.js       # Claude warm minimal palette design tokens
├── ml/
│   ├── dataset.py               # Ingestion, synthetic generator & drift scenarios
│   ├── validation.py            # Schema checks, imputation & outlier analysis
│   ├── pipeline.py              # Feature engineering & ColumnTransformer
│   ├── train.py                 # Multi-model training, MLflow logging & tuning
│   ├── evaluate.py              # Classification metrics & financial ROI calculation
│   ├── explain.py               # SHAP feature importance extraction
│   └── registry.py              # Local/MLflow Model Registry & quality gates
├── models_store/                # Local serialized model artifacts & registry.json
├── reports/                     # Generated Evidently HTML & JSON drift reports
├── tests/                       # 21-point pytest unit & integration test suite
│   ├── conftest.py              # Shared fixtures, in-memory DB & TestClient
│   ├── test_data_pipeline.py    # Ingestion, validation, preprocessing tests
│   ├── test_model_training.py   # Model evaluation, metrics & MLflow tests
│   ├── test_api.py              # FastAPI endpoints & schema validation tests
│   ├── test_drift_monitoring.py # Evidently AI & KS drift analysis tests
│   └── test_retraining_lifecycle.py # Quality gates, promotion & rollback tests
├── .env.example                 # Environment variables specification
├── .gitignore                   # Version control ignore rules
├── ARCHITECTURE.md              # Detailed architectural design & flowcharts
├── Dockerfile.backend           # Production Python 3.11 container
├── Dockerfile.frontend          # Multi-stage Node 20 Next.js container
├── docker-compose.yml           # PostgreSQL + MLflow + Backend + Frontend stack
├── PROJECT_STATUS.md            # Verification status matrix & empirical numbers
├── README.md                    # Industry-standard documentation (this file)
└── requirements.txt             # Pinned production Python dependencies
```

---

## 16. Contributing & Standards

1. **Branch Protection**: All pull requests must pass GitHub Actions CI/CD (linting, pytest, frontend build).
2. **Deterministic Preprocessing**: Any new feature transformer must be incorporated inside `ml/pipeline.py` to prevent training-serving skew.
3. **No Fabricated Experiments**: Model results, drift metrics, and evaluation scores must always be produced by verifiable code and authentic datasets.

---

## 17. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Built for transparent, observable, and reproducible MLOps.
