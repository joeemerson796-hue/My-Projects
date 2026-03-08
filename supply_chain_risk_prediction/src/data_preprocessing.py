"""Data loading, generation, and preprocessing for supply chain risk prediction."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED,
    DATA_RAW,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


def generate_supply_chain_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate a realistic synthetic supply chain dataset.

    Features are correlated with the target variable to produce learnable
    patterns that mirror real-world supplier risk dynamics.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    supplier_countries = ["China", "India", "USA", "Germany", "Mexico", "Vietnam", "Brazil", "Japan"]
    product_categories = ["Electronics", "Raw Materials", "Chemicals", "Textiles", "Machinery", "Packaging"]
    shipping_modes = ["Sea", "Air", "Road", "Rail"]
    payment_terms_options = ["Net 30", "Net 60", "Net 90", "Advance"]

    latent_risk = rng.normal(0, 1, n_samples)

    financial_health = np.clip(70 - 15 * latent_risk + rng.normal(0, 10, n_samples), 0, 100)
    on_time_delivery = np.clip(0.85 - 0.12 * latent_risk + rng.normal(0, 0.08, n_samples), 0.3, 1.0)
    defect_rate = np.clip(0.03 + 0.04 * latent_risk + rng.normal(0, 0.02, n_samples), 0, 0.5)
    supplier_rating = np.clip(3.8 - 0.6 * latent_risk + rng.normal(0, 0.4, n_samples), 1.0, 5.0)
    compliance_score = np.clip(80 - 12 * latent_risk + rng.normal(0, 8, n_samples), 0, 100)
    avg_quality = np.clip(78 - 10 * latent_risk + rng.normal(0, 7, n_samples), 0, 100)
    years_in_business = np.clip(rng.poisson(12, n_samples) - 2 * latent_risk, 1, 50).astype(int)

    lead_time = np.clip(25 + 10 * latent_risk + rng.normal(0, 8, n_samples), 1, 120).astype(int)
    order_qty = rng.integers(10, 10000, n_samples)
    unit_price = np.round(rng.uniform(1, 500, n_samples), 2)
    distance = np.round(rng.uniform(100, 15000, n_samples), 1)
    num_past_orders = rng.integers(1, 1000, n_samples)
    num_returns = np.clip(rng.poisson(3, n_samples) + (2 * latent_risk).astype(int), 0, 100)

    risk_score = (
        -0.25 * (financial_health / 100)
        - 0.20 * on_time_delivery
        + 0.20 * defect_rate * 10
        - 0.15 * (supplier_rating / 5)
        - 0.10 * (compliance_score / 100)
        + 0.10 * (lead_time / 120)
    )
    risk_score += rng.normal(0, 0.05, n_samples)

    thresholds = np.percentile(risk_score, [55, 82])
    risk_level = np.where(risk_score < thresholds[0], 0,
                          np.where(risk_score < thresholds[1], 1, 2))

    data = pd.DataFrame({
        "supplier_id": [f"SUP-{i:05d}" for i in range(n_samples)],
        "supplier_country": rng.choice(supplier_countries, n_samples),
        "product_category": rng.choice(product_categories, n_samples),
        "shipping_mode": rng.choice(shipping_modes, n_samples),
        "payment_terms": rng.choice(payment_terms_options, n_samples),
        "on_time_delivery_rate": np.round(on_time_delivery, 4),
        "defect_rate": np.round(defect_rate, 4),
        "lead_time_days": lead_time,
        "order_quantity": order_qty,
        "unit_price": unit_price,
        "supplier_rating": np.round(supplier_rating, 2),
        "financial_health_score": np.round(financial_health, 2),
        "years_in_business": years_in_business,
        "num_past_orders": num_past_orders,
        "avg_quality_score": np.round(avg_quality, 2),
        "distance_km": distance,
        "num_returns": num_returns,
        "compliance_score": np.round(compliance_score, 2),
        TARGET_COL: risk_level,
    })

    noise_mask = rng.random(n_samples) < 0.03
    data.loc[noise_mask, "financial_health_score"] = np.nan
    noise_mask2 = rng.random(n_samples) < 0.02
    data.loc[noise_mask2, "supplier_rating"] = np.nan

    return data


def load_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the dataset from CSV or generate it if not present."""
    raw_path = DATA_RAW / "supply_chain_data.csv"
    if filepath and Path(filepath).exists():
        raw_path = Path(filepath)

    if raw_path.exists():
        logger.info("Loading data from %s", raw_path)
        return pd.read_csv(raw_path)

    logger.info("Generating synthetic supply chain dataset")
    df = generate_supply_chain_data()
    df.to_csv(raw_path, index=False)
    logger.info("Saved generated data to %s (%d rows)", raw_path, len(df))
    return df


from pathlib import Path


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove duplicates."""
    logger.info("Cleaning data – shape before: %s", df.shape)

    if "supplier_id" in df.columns:
        df = df.drop(columns=["supplier_id"])

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()
    logger.info("Cleaning data – shape after: %s", df.shape)
    return df


def encode_features(
    df: pd.DataFrame, label_encoders: dict | None = None, fit: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """Label-encode categorical features."""
    if label_encoders is None:
        label_encoders = {}

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))

    return df, label_encoders


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into train / validation / test sets with stratification."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    relative_val = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val,
        random_state=RANDOM_STATE, stratify=y_train_val,
    )

    logger.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def handle_imbalance(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance training classes."""
    logger.info("Class distribution before SMOTE:\n%s", y_train.value_counts().to_dict())
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info("Class distribution after SMOTE:\n%s", pd.Series(y_resampled).value_counts().to_dict())
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def run_preprocessing_pipeline() -> dict:
    """Execute the full preprocessing pipeline and return artifacts."""
    df = load_data()
    df = clean_data(df)
    df, label_encoders = encode_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train)

    import joblib
    joblib.dump(label_encoders, DATA_PROCESSED / "label_encoders.pkl")

    return {
        "X_train": X_train_bal,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_bal,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoders": label_encoders,
        "feature_names": list(X_train.columns),
    }
