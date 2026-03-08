"""Inference pipeline for supply chain risk prediction."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, RISK_LABELS
from src.feature_engineering import create_derived_features

logger = logging.getLogger(__name__)


class SupplyChainPredictor:
    """Production-ready predictor wrapping the trained model and preprocessing."""

    def __init__(self, model_dir: Path | str = MODELS_DIR):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "best_model.pkl")
        self.scaler = joblib.load(DATA_PROCESSED / "scaler.pkl")
        self.label_encoders = joblib.load(DATA_PROCESSED / "label_encoders.pkl")

        with open(model_dir / "model_metadata.json") as f:
            meta = json.load(f)
        self.feature_names = meta["feature_names"]
        self.model_name = meta["best_model"]
        logger.info("Loaded model: %s with %d features", self.model_name, len(self.feature_names))

    def preprocess(self, data: dict | pd.DataFrame) -> pd.DataFrame:
        """Preprocess a single sample or DataFrame for prediction."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        if "supplier_id" in df.columns:
            df = df.drop(columns=["supplier_id"])

        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        df = create_derived_features(df)

        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0

        df = df[self.feature_names]
        df = pd.DataFrame(self.scaler.transform(df), columns=self.feature_names)
        return df

    def predict(self, data: dict | pd.DataFrame) -> dict:
        """Return predicted risk level and probabilities."""
        df = self.preprocess(data)
        pred = self.model.predict(df)
        proba = self.model.predict_proba(df)

        results = []
        for i in range(len(df)):
            results.append({
                "risk_level": RISK_LABELS[int(pred[i])],
                "risk_code": int(pred[i]),
                "probabilities": {
                    RISK_LABELS[j]: round(float(proba[i][j]), 4) for j in range(len(RISK_LABELS))
                },
            })
        return results if len(results) > 1 else results[0]


def run_inference_demo():
    """Quick demonstration of the inference pipeline."""
    predictor = SupplyChainPredictor()

    sample = {
        "supplier_country": "China",
        "product_category": "Electronics",
        "shipping_mode": "Sea",
        "payment_terms": "Net 60",
        "on_time_delivery_rate": 0.72,
        "defect_rate": 0.08,
        "lead_time_days": 45,
        "order_quantity": 500,
        "unit_price": 25.50,
        "supplier_rating": 3.2,
        "financial_health_score": 55.0,
        "years_in_business": 5,
        "num_past_orders": 120,
        "avg_quality_score": 65.0,
        "distance_km": 8500.0,
        "num_returns": 12,
        "compliance_score": 60.0,
    }

    result = predictor.predict(sample)
    print("\n" + "=" * 50)
    print("INFERENCE DEMO")
    print("=" * 50)
    print(f"Predicted Risk Level: {result['risk_level']}")
    print(f"Probabilities: {result['probabilities']}")
    return result
