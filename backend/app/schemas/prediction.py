"""Pydantic schemas for prediction requests and responses."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    customerID: Optional[str] = Field(None, description="Unique customer identifier")
    gender: str = Field(..., description="Customer gender: 'Male' or 'Female'")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen: 0 or 1")
    Partner: str = Field(..., description="Whether customer has partner: 'Yes' or 'No'")
    Dependents: str = Field(..., description="Whether customer has dependents: 'Yes' or 'No'")
    tenure: int = Field(..., ge=0, le=120, description="Tenure in months")
    PhoneService: str = Field(..., description="Phone service subscription: 'Yes' or 'No'")
    MultipleLines: str = Field(..., description="'No phone service', 'No', or 'Yes'")
    InternetService: str = Field(..., description="Internet service: 'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    OnlineBackup: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    TechSupport: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(..., description="'Yes', 'No', or 'No internet service'")
    Contract: str = Field(..., description="Contract term: 'Month-to-month', 'One year', or 'Two year'")
    PaperlessBilling: str = Field(..., description="Paperless billing: 'Yes' or 'No'")
    PaymentMethod: str = Field(..., description="Payment method name")
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Monthly charges in USD")
    TotalCharges: Union[float, str] = Field(..., description="Total accumulated charges or numeric string")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in ["Male", "Female"]:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v):
        if isinstance(v, str):
            v_strip = v.strip()
            if not v_strip:
                return 0.0
            try:
                return float(v_strip)
            except ValueError:
                raise ValueError("TotalCharges must be a valid float or numeric string")
        return float(v)


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int = Field(..., description="0 = No Churn, 1 = Churn")
    churn_label: str = Field(..., description="'No' or 'Yes'")
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    model_version: str
    latency_ms: float
    timestamp: str
    prediction_id: Optional[int] = None


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    model_version: str
    average_churn_probability: float
    churn_count: int
    non_churn_count: int
