"""FastAPI application for supply chain risk prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import SupplyChainPredictor

app = FastAPI(
    title="Supply Chain Risk Prediction API",
    description="Predict supplier risk levels using ML-powered analysis",
    version="1.0.0",
)

predictor: SupplyChainPredictor | None = None


class SupplierInput(BaseModel):
    supplier_country: str = Field(..., example="China")
    product_category: str = Field(..., example="Electronics")
    shipping_mode: str = Field(..., example="Sea")
    payment_terms: str = Field(..., example="Net 60")
    on_time_delivery_rate: float = Field(..., ge=0, le=1, example=0.85)
    defect_rate: float = Field(..., ge=0, le=1, example=0.03)
    lead_time_days: int = Field(..., ge=1, example=30)
    order_quantity: int = Field(..., ge=1, example=500)
    unit_price: float = Field(..., ge=0, example=25.50)
    supplier_rating: float = Field(..., ge=1, le=5, example=4.0)
    financial_health_score: float = Field(..., ge=0, le=100, example=75.0)
    years_in_business: int = Field(..., ge=0, example=10)
    num_past_orders: int = Field(..., ge=0, example=200)
    avg_quality_score: float = Field(..., ge=0, le=100, example=80.0)
    distance_km: float = Field(..., ge=0, example=5000.0)
    num_returns: int = Field(..., ge=0, example=5)
    compliance_score: float = Field(..., ge=0, le=100, example=85.0)


class PredictionResponse(BaseModel):
    risk_level: str
    risk_code: int
    probabilities: dict[str, float]


@app.on_event("startup")
def load_model():
    global predictor
    predictor = SupplyChainPredictor()


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(supplier: SupplierInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict(supplier.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
