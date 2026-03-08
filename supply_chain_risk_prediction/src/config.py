"""Configuration for the Supply Chain Risk Prediction pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15

TARGET_COL = "risk_level"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

CATEGORICAL_FEATURES = [
    "supplier_country",
    "product_category",
    "shipping_mode",
    "payment_terms",
]

NUMERICAL_FEATURES = [
    "on_time_delivery_rate",
    "defect_rate",
    "lead_time_days",
    "order_quantity",
    "unit_price",
    "supplier_rating",
    "financial_health_score",
    "years_in_business",
    "num_past_orders",
    "avg_quality_score",
    "distance_km",
    "num_returns",
    "compliance_score",
]

N_OPTUNA_TRIALS = 15
CV_FOLDS = 3
