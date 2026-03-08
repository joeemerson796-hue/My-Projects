# AI Document Classification System

An end-to-end machine learning pipeline for classifying technical documents into domain-specific categories using Natural Language Processing (NLP). The system compares classical TF-IDF-based models against a transformer-based DistilBERT approach and exposes the best-performing model through a production-ready FastAPI REST endpoint.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [ML Pipeline Overview](#ml-pipeline-overview)
- [NLP Preprocessing](#nlp-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models and Algorithms](#models-and-algorithms)
- [Model Evaluation Results](#model-evaluation-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [API Usage](#api-usage)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Problem Statement

Organizations that manage large volumes of technical documentation — data sheets, research papers, engineering specifications, support tickets — need automated systems to route, tag, and organize these documents by domain. Manual classification is slow, inconsistent, and does not scale.

This project builds a supervised multi-class text classifier that assigns incoming technical documents to one of eight domain categories spanning computer hardware, software, and scientific disciplines. The classifier is designed to:

1. **Accurately categorize** unseen documents with minimal latency.
2. **Compare classical and deep-learning approaches** to identify the best cost-performance tradeoff.
3. **Serve predictions in real time** through a lightweight REST API suitable for integration into document management workflows.

---

## Dataset

### 20 Newsgroups

The project uses the **20 Newsgroups** dataset, a well-established benchmark in text classification research originally collected by Ken Lang. The dataset is available directly through `scikit-learn` via `sklearn.datasets.fetch_20newsgroups`.

**Why 20 Newsgroups?**

| Reason | Detail |
|---|---|
| **Real-world text** | Posts are written by real users with natural variation in length, vocabulary, and quality. |
| **Multi-class structure** | Twenty fine-grained categories provide a challenging classification problem that mirrors real enterprise taxonomies. |
| **Reproducibility** | The dataset is publicly available, versioned, and widely cited, making results directly comparable to published baselines. |
| **Noise present** | Documents contain email headers, signatures, quoted text, and typos — forcing the pipeline to handle messy real-world input. |

### Selected Categories

Eight technically focused newsgroups were selected to simulate a domain-specific document classification scenario:

| # | Category | Domain |
|---|---|---|
| 0 | `comp.graphics` | Computer Graphics |
| 1 | `comp.sys.ibm.pc.hardware` | IBM PC Hardware |
| 2 | `comp.sys.mac.hardware` | Mac Hardware |
| 3 | `sci.crypt` | Cryptography |
| 4 | `sci.electronics` | Electronics |
| 5 | `sci.med` | Medicine |
| 6 | `sci.space` | Space Science |
| 7 | `comp.os.ms-windows.misc` | MS Windows |

### Dataset Statistics

| Split | Documents |
|---|---|
| Training | 4,579 |
| Test | 3,038 |
| **Total** | **7,617** |

Email headers, footers, and quoted reply blocks are removed at load time (`remove=("headers", "footers", "quotes")`) to prevent the models from learning author identity or email metadata rather than topical content.

---

## ML Pipeline Overview

The pipeline is orchestrated by `run_pipeline.py` and executes five sequential stages:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Data Loading  │────▶│  2. NLP Preproc.  │────▶│  3. TF-IDF       │
│  & Cleaning       │     │  & Lemmatization  │     │  Vectorization   │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                         ┌─────────────────────────────────┼──────────────────┐
                         ▼                                 ▼                  ▼
                  ┌──────────────┐              ┌──────────────┐    ┌──────────────────┐
                  │ Logistic Reg │              │  Linear SVM  │    │  DistilBERT      │
                  │ (GridSearch) │              │ (GridSearch)  │    │  Fine-Tuning     │
                  └──────┬───────┘              └──────┬───────┘    └────────┬─────────┘
                         │                             │                     │
                         └─────────────┬───────────────┘                     │
                                       ▼                                     ▼
                              ┌──────────────────┐              ┌────────────────────┐
                              │   Evaluation &   │              │   Evaluation &     │
                              │   Comparison     │◀─────────────│   Comparison       │
                              └──────┬───────────┘              └────────────────────┘
                                     ▼
                              ┌──────────────────┐
                              │  FastAPI Serving  │
                              └──────────────────┘
```

### Stage Details

1. **Data Preprocessing** (`src/data_preprocessing.py`) — Loads the 20 Newsgroups dataset, applies text cleaning, tokenization, stopword removal, and lemmatization.
2. **Feature Engineering** (`src/feature_engineering.py`) — Builds TF-IDF feature vectors from the preprocessed text.
3. **Model Training** (`src/train_model.py`) — Trains Logistic Regression, Linear SVM (both with grid search), and fine-tunes DistilBERT.
4. **Evaluation** (`src/evaluate_model.py`) — Computes metrics, generates confusion matrices, and produces a model comparison chart.
5. **Inference** (`src/inference.py`) — Provides a `DocumentClassifier` class used by both the FastAPI app and the CLI demo.

---

## NLP Preprocessing

Every document passes through a four-step NLP pipeline before feature extraction:

| Step | Technique | Implementation |
|---|---|---|
| **Text Cleaning** | Lowercasing, HTML tag removal, URL/email stripping, non-alphabetic character removal, whitespace normalization | `clean_text()` in `data_preprocessing.py` |
| **Tokenization** | Word-level tokenization using NLTK's Punkt tokenizer | `nltk.tokenize.word_tokenize` |
| **Stopword Removal** | English stopwords from NLTK corpus; tokens shorter than 3 characters also removed | `nltk.corpus.stopwords` |
| **Lemmatization** | WordNet-based lemmatization to reduce words to their dictionary form | `nltk.stem.WordNetLemmatizer` |

**Example transformation:**

```
Raw:    "The GPU renders 3D graphics using ray tracing on Windows 10!"
Clean:  "the gpu renders  d graphics using ray tracing on windows"
Final:  "gpu render graphics using ray tracing window"
```

---

## Feature Engineering

### TF-IDF Vectorization

Term Frequency–Inverse Document Frequency (TF-IDF) converts preprocessed text into fixed-length numerical vectors suitable for classical ML models.

| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 10,000 | Caps vocabulary to the 10K most informative terms, balancing signal and dimensionality. |
| `ngram_range` | (1, 2) | Includes unigrams and bigrams to capture two-word phrases (e.g., "ray tracing", "hard drive"). |
| `min_df` | 3 | Drops terms appearing in fewer than 3 documents, removing noise and typos. |
| `max_df` | 0.95 | Drops terms appearing in more than 95% of documents, effectively removing corpus-wide stopwords. |
| `sublinear_tf` | True | Applies `1 + log(tf)` scaling, dampening the effect of very high term frequencies. |

### BERT Tokenization

For the DistilBERT model, raw cleaned text (before lemmatization) is tokenized using the `distilbert-base-uncased` WordPiece tokenizer with a maximum sequence length of 256 tokens, padding and truncation enabled.

---

## Models and Algorithms

### 1. TF-IDF + Logistic Regression

Logistic Regression is a strong linear baseline for text classification. It models the log-odds of each class as a linear combination of TF-IDF features and supports multi-class classification natively via the softmax (multinomial) formulation.

- **Solver:** L-BFGS (efficient for multi-class with L2 regularization)
- **Hyperparameter search:** `C ∈ {0.1, 1, 10}`, `penalty = l2` via 3-fold cross-validation
- **Best hyperparameters:** `C = 10`, `penalty = l2`

**Why Logistic Regression?**
- Produces calibrated probability estimates (`predict_proba`), useful for confidence thresholds in production.
- Fast to train and predict, making it ideal for real-time serving.
- Highly interpretable — feature weights directly indicate which terms drive each class.

### 2. TF-IDF + Linear SVM (LinearSVC)

Support Vector Machines with a linear kernel are the historically dominant approach for high-dimensional text classification. LinearSVC finds the maximum-margin hyperplane separating each pair of classes.

- **Hyperparameter search:** `C ∈ {0.1, 1, 10}` via 3-fold cross-validation
- **Best hyperparameters:** `C = 0.1`
- **Max iterations:** 5,000

**Why Linear SVM?**
- Excels in high-dimensional, sparse feature spaces (exactly what TF-IDF produces).
- The margin-maximizing objective provides strong generalization, especially with limited data.
- Empirically, LinearSVC often matches or slightly outperforms Logistic Regression on text tasks.

### 3. DistilBERT Fine-Tuning

DistilBERT is a distilled version of BERT that retains 97% of BERT's language understanding capability with 40% fewer parameters and 60% faster inference. Fine-tuning adapts the pretrained transformer weights to the specific classification task.

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Max sequence length | 256 tokens |
| Training samples | 1,200 (CPU-friendly subset) |
| Test samples | 300 |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Optimizer | AdamW (Hugging Face default) |

**Why DistilBERT?**
- Contextual embeddings capture semantic meaning, synonyms, and word order — information that bag-of-words TF-IDF discards.
- Transfer learning from massive pretraining corpora can boost performance, especially on small or domain-specific datasets.
- DistilBERT specifically balances performance with computational cost.

> **Important Note on BERT Results:** The DistilBERT model in this project was trained on a reduced subset of only 1,200 training samples (out of 4,579) using CPU-only training for 3 epochs. This was a deliberate design choice to keep the pipeline runnable on any machine without GPU requirements. With GPU training on the full dataset, more epochs, and hyperparameter tuning, DistilBERT would likely **outperform** the TF-IDF models, as demonstrated consistently in NLP benchmarks. The current BERT result should be interpreted as a lower-bound proof of concept, not as evidence of transformer inferiority for this task.

---

## Model Evaluation Results

### Summary Table

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | Precision (Weighted) | Recall (Weighted) |
|---|---|---|---|---|---|
| **TF-IDF + SVM** | **74.52%** | **0.7437** | **0.7439** | **0.7437** | **0.7452** |
| TF-IDF + LogReg | 73.57% | 0.7357 | 0.7359 | 0.7364 | 0.7357 |
| DistilBERT* | — | 0.6186 | — | — | — |

*\*DistilBERT trained on 1,200-sample CPU subset (see note above).*

### Key Observations

- **TF-IDF + SVM is the best-performing model** with a weighted F1 of 0.7437, outperforming Logistic Regression by ~0.8 percentage points.
- **Logistic Regression is a strong second** and has the advantage of producing calibrated probability estimates.
- **DistilBERT underperforms** relative to classical models in this constrained training setup, achieving an F1 of 0.6186. This is expected given the reduced training data (1,200 vs. 4,579 samples) and limited epochs on CPU.
- The **small gap between SVM and LogReg** (~0.8 F1 points) suggests both models are operating near the ceiling for linear models on this feature space.
- Removing headers, footers, and quotes makes the task harder than standard 20 Newsgroups benchmarks (which often report 85%+ accuracy with metadata leakage).

### Best Model Hyperparameters (via Grid Search)

| Model | Parameter | Best Value |
|---|---|---|
| Logistic Regression | C | 10 |
| Logistic Regression | penalty | l2 |
| Linear SVM | C | 0.1 |

The SVM's lower optimal C (stronger regularization) indicates it benefits from constraining the decision boundary, consistent with its margin-maximizing objective in high-dimensional space.

---

## Project Structure

```
document_classification/
├── app/
│   └── main.py                    # FastAPI REST API application
├── data/
│   ├── raw/                       # Original train/test CSVs from 20 Newsgroups
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/                 # TF-IDF vectorizer and processed artifacts
├── models/
│   ├── tfidf_logreg.pkl           # Serialized Logistic Regression model
│   ├── tfidf_svm.pkl              # Serialized LinearSVC model
│   ├── bert_model/                # Fine-tuned DistilBERT weights and tokenizer
│   ├── bert_checkpoints/          # Intermediate BERT training checkpoints
│   └── model_metadata.json        # Best model info and category mappings
├── reports/
│   ├── figures/                   # Confusion matrices, model comparison charts
│   ├── evaluation_results.json    # Full evaluation metrics for all models
│   └── training_results.json      # Training summary with best hyperparameters
├── src/
│   ├── config.py                  # Centralized configuration and paths
│   ├── data_preprocessing.py      # Data loading and NLP preprocessing
│   ├── feature_engineering.py     # TF-IDF vectorization
│   ├── train_model.py             # Model training (LogReg, SVM, BERT)
│   ├── evaluate_model.py          # Evaluation, metrics, and visualization
│   └── inference.py               # DocumentClassifier for production inference
├── run_pipeline.py                # End-to-end pipeline orchestrator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── DOCUMENTATION.md               # Detailed engineering documentation
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 4 GB+ RAM recommended (DistilBERT fine-tuning is memory-intensive)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd document_classification

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | TF-IDF, Logistic Regression, LinearSVC, metrics, grid search |
| `nltk` | Tokenization, stopword removal, lemmatization |
| `torch` | PyTorch backend for DistilBERT |
| `transformers` | Hugging Face DistilBERT model and tokenizer |
| `matplotlib`, `seaborn` | Visualization (confusion matrices, comparison charts) |
| `fastapi`, `uvicorn` | REST API serving |
| `joblib` | Model serialization |

---

## Running the Pipeline

### Full Pipeline (Training + Evaluation + Inference Demo)

```bash
cd document_classification
python run_pipeline.py
```

This will:
1. Download and preprocess the 20 Newsgroups dataset
2. Build TF-IDF feature vectors
3. Train Logistic Regression, Linear SVM, and DistilBERT
4. Evaluate all models and generate reports in `reports/`
5. Run a quick inference demo on sample documents

Expected runtime: ~10–20 minutes on CPU (DistilBERT training is the bottleneck).

### Start the API Server

```bash
cd document_classification
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

The API will be available at `http://localhost:8002`. Interactive documentation is served at `http://localhost:8002/docs` (Swagger UI).

> **Note:** The API requires trained model artifacts to exist in the `models/` directory. Run the full pipeline first if models have not been trained.

---

## API Usage

### Health Check

```bash
curl http://localhost:8002/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Classify a Document

```bash
curl -X POST http://localhost:8002/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The new GPU architecture features improved ray tracing cores and CUDA performance for rendering complex 3D graphics in real time.",
    "backend": "tfidf"
  }'
```

**Response:**
```json
{
  "category": "comp.graphics",
  "label": 0,
  "probabilities": {
    "comp.graphics": 0.7823,
    "comp.sys.ibm.pc.hardware": 0.0412,
    "comp.sys.mac.hardware": 0.0198,
    "sci.crypt": 0.0134,
    "sci.electronics": 0.0567,
    "sci.med": 0.0089,
    "sci.space": 0.0321,
    "comp.os.ms-windows.misc": 0.0456
  }
}
```

### More Examples

**Cryptography document:**
```bash
curl -X POST http://localhost:8002/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Researchers developed a novel encryption protocol based on elliptic curve cryptography that provides enhanced security for IoT devices and protects against quantum computing attacks."
  }'
```

**Medical document:**
```bash
curl -X POST http://localhost:8002/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient was diagnosed with acute myocardial infarction and treated with percutaneous coronary intervention within the first hour of symptom onset."
  }'
```

**Space science document:**
```bash
curl -X POST http://localhost:8002/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "NASA launched the James Webb Space Telescope to observe the earliest galaxies formed after the Big Bang using infrared spectroscopy from the L2 Lagrange point."
  }'
```

### Request Schema

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | Yes | — | Document text to classify (minimum 10 characters) |
| `backend` | string | No | `"tfidf"` | Model backend: `"tfidf"` or `"bert"` |

### Response Schema

| Field | Type | Description |
|---|---|---|
| `category` | string | Predicted newsgroup category |
| `label` | integer | Numeric label (0–7) |
| `probabilities` | object | Per-class probability distribution (available for LogReg/BERT backends) |

---

## Limitations

1. **Domain specificity:** The model is trained exclusively on 20 Newsgroups data from the early 1990s. Vocabulary, writing style, and topics may not generalize to modern technical documents without retraining or domain adaptation.

2. **Constrained BERT training:** DistilBERT was fine-tuned on only 1,200 samples using CPU, yielding an F1 well below its potential. This limits the comparison's validity as a reflection of transformer capability.

3. **Fixed category set:** The system supports only 8 predefined categories. Documents outside these domains will be force-classified into the nearest category with no "unknown" or "out-of-distribution" detection.

4. **No confidence thresholding:** The API returns a prediction for every input regardless of model confidence. In production, low-confidence predictions should be routed to human review.

5. **English only:** The NLP pipeline (stopwords, lemmatizer, tokenizer) and pretrained BERT model are English-specific.

6. **Linear models lack semantic understanding:** TF-IDF models treat text as a bag of words and cannot capture word order, synonymy, or contextual meaning. A document about "Apple the company" and "apple the fruit" would receive similar feature vectors.

7. **No incremental learning:** Adding new categories or retraining on new data requires re-running the full pipeline from scratch.

---

## Future Improvements

| Improvement | Impact |
|---|---|
| **GPU-accelerated BERT training** on the full dataset with learning rate scheduling and early stopping | Expected to push F1 above 0.85 based on published benchmarks |
| **Ensemble methods** combining SVM and BERT predictions (e.g., stacking, weighted voting) | Leverages complementary strengths of both approaches |
| **Out-of-distribution detection** using prediction entropy or dedicated OOD classifiers | Prevents silent misclassification of off-topic documents |
| **Confidence thresholding** with human-in-the-loop escalation for low-confidence predictions | Improves precision in production deployments |
| **Active learning** to iteratively label the most informative unlabeled documents | Reduces labeling cost for domain adaptation |
| **Modern transformer architectures** (RoBERTa, DeBERTa, or domain-specific SciBERT) | Better representations for scientific/technical text |
| **Containerized deployment** with Docker and model versioning via MLflow | Reproducible, scalable production serving |
| **Multilingual support** using multilingual transformers (XLM-RoBERTa) | Extends coverage beyond English documents |
| **Document chunking** for long documents exceeding the 256-token BERT limit | Prevents information loss from truncation |
| **Explainability layer** using LIME or SHAP for per-prediction feature attribution | Builds user trust and aids debugging |

---

## License

This project is intended for educational and portfolio demonstration purposes. The 20 Newsgroups dataset is publicly available and widely used in academic research.
