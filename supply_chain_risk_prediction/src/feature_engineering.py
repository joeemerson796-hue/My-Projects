"""Feature engineering for supply chain risk prediction."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED

logger = logging.getLogger(__name__)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and ratio features from existing columns."""
    df = df.copy()

    if "order_quantity" in df.columns and "unit_price" in df.columns:
        df["total_order_value"] = df["order_quantity"] * df["unit_price"]

    if "num_returns" in df.columns and "num_past_orders" in df.columns:
        df["return_rate"] = df["num_returns"] / (df["num_past_orders"] + 1)

    if "on_time_delivery_rate" in df.columns and "defect_rate" in df.columns:
        df["reliability_index"] = df["on_time_delivery_rate"] * (1 - df["defect_rate"])

    if "financial_health_score" in df.columns and "compliance_score" in df.columns:
        df["stability_score"] = (df["financial_health_score"] + df["compliance_score"]) / 2

    if "distance_km" in df.columns and "lead_time_days" in df.columns:
        df["delivery_efficiency"] = df["distance_km"] / (df["lead_time_days"] + 1)

    if "avg_quality_score" in df.columns and "supplier_rating" in df.columns:
        df["quality_rating_product"] = df["avg_quality_score"] * df["supplier_rating"]

    logger.info("Created derived features – new shape: %s", df.shape)
    return df


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize numerical features using training-set statistics."""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
    else:
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train), columns=X_train.columns, index=X_train.index
        )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    import joblib
    joblib.dump(scaler, DATA_PROCESSED / "scaler.pkl")
    logger.info("Features scaled and scaler saved")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def run_feature_engineering(artifacts: dict) -> dict:
    """Apply feature engineering to all splits."""
    X_train = create_derived_features(artifacts["X_train"])
    X_val = create_derived_features(artifacts["X_val"])
    X_test = create_derived_features(artifacts["X_test"])

    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    artifacts.update({
        "X_train": X_train_sc,
        "X_val": X_val_sc,
        "X_test": X_test_sc,
        "scaler": scaler,
        "feature_names": list(X_train_sc.columns),
    })
    return artifacts
