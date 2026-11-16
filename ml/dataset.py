"""Data ingestion, validation, and dataset generation for DriftGuard."""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "reference_data.csv"


def generate_synthetic_telco_data(num_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic Telco Customer Churn dataset matching canonical IBM schema and distributions.
    
    Used as an offline fallback or reproducible benchmark.
    """
    np.random.seed(random_state)
    
    # Demographics
    gender = np.random.choice(["Male", "Female"], size=num_samples)
    senior = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", np.random.choice(["Yes", "No"], size=num_samples, p=[0.55, 0.45]), "No")
    
    # Account info
    tenure = np.random.geometric(p=0.03, size=num_samples)
    tenure = np.clip(tenure, 1, 72)
    
    # Contract
    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract = []
    for t in tenure:
        if t < 12:
            contract.append(np.random.choice(contract_choices, p=[0.85, 0.10, 0.05]))
        elif t < 36:
            contract.append(np.random.choice(contract_choices, p=[0.45, 0.35, 0.20]))
        else:
            contract.append(np.random.choice(contract_choices, p=[0.20, 0.35, 0.45]))
    
    paperless = np.random.choice(["Yes", "No"], size=num_samples, p=[0.59, 0.41])
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=num_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # Services
    phone = np.random.choice(["Yes", "No"], size=num_samples, p=[0.90, 0.10])
    multiple_lines = []
    for p in phone:
        if p == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    internet = np.random.choice(["DSL", "Fiber optic", "No"], size=num_samples, p=[0.34, 0.44, 0.22])
    
    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    service_cols = {}
    for s in services:
        col = []
        for inet in internet:
            if inet == "No":
                col.append("No internet service")
            else:
                col.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
        service_cols[s] = col
        
    # Charges
    monthly = []
    for i in range(num_samples):
        base = 20.0
        if phone[i] == "Yes":
            base += 10.0
            if multiple_lines[i] == "Yes":
                base += 5.0
        if internet[i] == "DSL":
            base += 25.0
        elif internet[i] == "Fiber optic":
            base += 45.0
            
        for s in services:
            if service_cols[s][i] == "Yes":
                base += 6.5
                
        noise = np.random.normal(0, 2.5)
        monthly.append(round(max(18.25, base + noise), 2))
        
    monthly = np.array(monthly)
    total = np.round(monthly * tenure + np.random.normal(0, 15.0, size=num_samples), 2)
    total = np.clip(total, monthly, None)
    
    # Calculate churn probability based on well-established domain factors
    churn_score = (
        (np.array(contract) == "Month-to-month") * 1.6 +
        (internet == "Fiber optic") * 1.2 +
        (payment == "Electronic check") * 0.9 +
        (senior == 1) * 0.4 +
        (monthly > 70) * 0.7 -
        (tenure / 20.0) -
        (np.array(service_cols["TechSupport"]) == "Yes") * 0.8 -
        (np.array(service_cols["OnlineSecurity"]) == "Yes") * 0.7 -
        (np.array(contract) == "Two year") * 1.5 -
        2.2
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_score))
    churn = np.where(np.random.uniform(0, 1, size=num_samples) < churn_prob, "Yes", "No")
    
    # Customer IDs
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(num_samples)]
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": service_cols["OnlineSecurity"],
        "OnlineBackup": service_cols["OnlineBackup"],
        "DeviceProtection": service_cols["DeviceProtection"],
        "TechSupport": service_cols["TechSupport"],
        "StreamingTV": service_cols["StreamingTV"],
        "StreamingMovies": service_cols["StreamingMovies"],
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": [f"{t:.2f}" for t in total],
        "Churn": churn
    })
    
    # Simulate authentic real-world missing/empty values in TotalCharges (approx ~0.2%, e.g. ~11 in 7043)
    num_blanks = max(1, int(num_samples * 0.002)) if num_samples >= 50 else 0
    if num_blanks > 0:
        blank_indices = np.random.choice(num_samples, size=num_blanks, replace=False)
        for idx in blank_indices:
            df.loc[idx, "TotalCharges"] = " "
        
    return df


def ingest_telco_data(force_download: bool = False) -> pd.DataFrame:
    """Ingest Telco Churn dataset, either by downloading or using reproducible generator."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    
    if RAW_DATA_PATH.exists() and not force_download:
        logger.info(f"Loading existing raw dataset from {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)
        return df
        
    logger.info("Attempting to download Telco Customer Churn dataset...")
    try:
        response = requests.get(RAW_TELCO_URL, timeout=10)
        if response.status_code == 200 and len(response.content) > 10000:
            with open(RAW_DATA_PATH, "wb") as f:
                f.write(response.content)
            df = pd.read_csv(RAW_DATA_PATH)
            logger.info(f"Successfully downloaded {len(df)} records.")
            return df
    except Exception as e:
        logger.warning(f"Download failed ({e}), falling back to synthetic distribution generator.")
        
    df = generate_synthetic_telco_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Generated and saved {len(df)} records to {RAW_DATA_PATH}")
    return df


def generate_drifted_data(
    base_df: pd.DataFrame,
    scenario: str = "gradual_fiber_optic_shift",
    num_samples: int = 500,
    random_state: int = 101
) -> pd.DataFrame:
    """Generate production inference samples with configurable drift scenarios.
    
    Scenarios:
    - 'normal': no intentional drift, drawn from reference distribution
    - 'gradual_fiber_optic_shift': InternetService Fiber Optic increases from 44% to 85%, MonthlyCharges shifts +$25
    - 'demographic_shift': SeniorCitizen surges to 70%, tenure collapses to mean 4 months
    - 'economic_crisis': PaymentMethod electronic check drops, Month-to-month contracts surge to 95%
    """
    np.random.seed(random_state)
    sample_df = base_df.sample(n=num_samples, replace=True, random_state=random_state).copy().reset_index(drop=True)
    
    # Assign fresh customer IDs
    sample_df["customerID"] = [f"PROD-{np.random.randint(10000, 99999)}" for _ in range(num_samples)]
    
    if scenario == "normal":
        # Slight sampling noise only
        return sample_df
        
    elif scenario == "gradual_fiber_optic_shift":
        # Drastic increase in Fiber Optic and higher monthly charges
        mask = np.random.uniform(0, 1, size=num_samples) < 0.75
        sample_df.loc[mask, "InternetService"] = "Fiber optic"
        sample_df["MonthlyCharges"] = sample_df["MonthlyCharges"] + np.random.normal(22.0, 5.0, size=num_samples)
        sample_df["MonthlyCharges"] = sample_df["MonthlyCharges"].clip(20.0, 140.0)
        
    elif scenario == "demographic_shift":
        # SeniorCitizen surges, tenure plunges
        sample_df["SeniorCitizen"] = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
        sample_df["tenure"] = np.random.exponential(scale=3.5, size=num_samples).astype(int) + 1
        sample_df["tenure"] = sample_df["tenure"].clip(1, 15)
        
    elif scenario == "economic_crisis":
        # Contracts shift almost entirely to month-to-month
        sample_df["Contract"] = np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            size=num_samples,
            p=[0.92, 0.05, 0.03]
        )
        sample_df["PaymentMethod"] = "Electronic check"
        
    return sample_df
