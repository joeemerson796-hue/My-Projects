"""Configuration for the Document Classification pipeline."""

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

SELECTED_CATEGORIES = [
    "comp.graphics",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "comp.os.ms-windows.misc",
]

CATEGORY_LABELS = {i: cat.split(".")[-1] for i, cat in enumerate(SELECTED_CATEGORIES)}

MAX_TFIDF_FEATURES = 10000
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5
BERT_TRAIN_SAMPLES = 1200
BERT_TEST_SAMPLES = 300
