# Supply Chain Risk Prediction

A production-grade machine learning pipeline that predicts supplier risk levels (**Low**, **Medium**, **High**) from supply chain operational data. Built as an end-to-end system — from synthetic data generation through model training, evaluation, explainability analysis, and a live FastAPI prediction endpoint.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Algorithms & Rationale](#algorithms--rationale)
- [Model Evaluation Results](#model-evaluation-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the Full Pipeline](#run-the-full-pipeline)
  - [Serve the Prediction API](#serve-the-prediction-api)
  - [API Examples](#api-examples)
- [Explainability](#explainability)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Problem Statement

Global supply chains are becoming increasingly complex and vulnerable to disruptions — from logistics delays and quality failures to financial instability of suppliers. Companies like **Z2Data**, **Resilinc**, and **Interos** build risk intelligence platforms that help procurement and operations teams proactively identify at-risk suppliers before disruptions cascade through production lines.

This project tackles the **supplier risk classification** problem: given a set of operational, financial, and quality metrics for a supplier, predict whether the supplier represents a **Low**, **Medium**, or **High** risk to the buying organization.

**Why this matters:**

| Impact Area | Consequence of Unmanaged Risk |
|---|---|
| Production continuity | Unexpected stockouts, line stoppages |
| Quality assurance | High defect rates, costly recalls |
| Financial exposure | Supplier bankruptcy, unpaid liabilities |
| Compliance | Regulatory violations, reputational damage |
| Lead time reliability | Missed delivery windows, customer churn |

The pipeline automates risk scoring that would otherwise require manual supplier audits, enabling data-driven procurement decisions at scale.

---

## Dataset

The project uses a **synthetic dataset of 10,000 supplier records** generated to mirror real-world supply chain risk dynamics. A latent risk factor drives correlated feature distributions, producing learnable patterns while preserving realistic noise (including ~3% missing values injected into `financial_health_score` and ~2% into `supplier_rating`).

### Raw Features

| Feature | Type | Range / Values | Description |
|---|---|---|---|
| `supplier_country` | Categorical | China, India, USA, Germany, Mexico, Vietnam, Brazil, Japan | Supplier's country of operation |
| `product_category` | Categorical | Electronics, Raw Materials, Chemicals, Textiles, Machinery, Packaging | Product type supplied |
| `shipping_mode` | Categorical | Sea, Air, Road, Rail | Primary shipping method |
| `payment_terms` | Categorical | Net 30, Net 60, Net 90, Advance | Contractual payment terms |
| `on_time_delivery_rate` | Float | [0.30, 1.00] | Fraction of orders delivered on time |
| `defect_rate` | Float | [0.00, 0.50] | Fraction of units found defective |
| `lead_time_days` | Integer | [1, 120] | Average lead time in days |
| `order_quantity` | Integer | [10, 10000] | Typical order size |
| `unit_price` | Float | [1.00, 500.00] | Price per unit (USD) |
| `supplier_rating` | Float | [1.0, 5.0] | Internal supplier quality rating |
| `financial_health_score` | Float | [0, 100] | Composite financial stability score |
| `years_in_business` | Integer | [1, 50] | Supplier operating history |
| `num_past_orders` | Integer | [1, 1000] | Historical order count |
| `avg_quality_score` | Float | [0, 100] | Mean incoming quality inspection score |
| `distance_km` | Float | [100, 15000] | Geographic distance to supplier |
| `num_returns` | Integer | [0, 100] | Count of returned shipments |
| `compliance_score` | Float | [0, 100] | Regulatory and audit compliance score |

### Engineered Features

| Feature | Formula | Rationale |
|---|---|---|
| `total_order_value` | `order_quantity × unit_price` | Captures financial exposure per order |
| `return_rate` | `num_returns / (num_past_orders + 1)` | Normalizes returns by order volume |
| `reliability_index` | `on_time_delivery_rate × (1 − defect_rate)` | Joint delivery + quality reliability |
| `stability_score` | `(financial_health_score + compliance_score) / 2` | Combined organizational stability |
| `delivery_efficiency` | `distance_km / (lead_time_days + 1)` | Logistics efficiency proxy |
| `quality_rating_product` | `avg_quality_score × supplier_rating` | Quality signal amplification |

### Target Variable

| Label | Code | Distribution | Description |
|---|---|---|---|
| Low | 0 | ~55% | Supplier is reliable and low-risk |
| Medium | 1 | ~27% | Moderate risk — warrants monitoring |
| High | 2 | ~18% | High risk — requires immediate action |

Class imbalance is addressed with **SMOTE** (Synthetic Minority Over-sampling Technique) applied only to the training set.

---

## ML Pipeline

The pipeline is orchestrated by `run_pipeline.py` and executes five sequential stages:

```
┌─────────────────────┐
│ 1. Data Preprocessing│
│   • Load / generate  │
│   • Clean & impute   │
│   • Label encode     │
│   • Stratified split │
│   • SMOTE balancing  │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ 2. Feature Engineering│
│   • Derived features  │
│   • StandardScaler    │
└────────┬──────────────┘
         ▼
┌──────────────────────────────┐
│ 3. Model Training & Tuning   │
│   • 4 algorithms             │
│   • Optuna (15 trials, 3-CV) │
│   • Best model selection     │
└────────┬─────────────────────┘
         ▼
┌──────────────────────────────┐
│ 4. Evaluation & Explainability│
│   • Test-set metrics          │
│   • Confusion matrices        │
│   • Feature importance plots  │
│   • SHAP analysis             │
└────────┬──────────────────────┘
         ▼
┌─────────────────────┐
│ 5. Inference Demo    │
│   • Sample prediction│
│   • Probability output│
└──────────────────────┘
```

### Data Split Strategy

| Split | Proportion | Purpose |
|---|---|---|
| Training | 68% (after 80/20 test split, then 85/15 val split) | Model fitting (SMOTE applied here) |
| Validation | 12% | Hyperparameter selection |
| Test | 20% | Final unbiased evaluation |

All splits use **stratified sampling** to maintain class proportions.

---

## Algorithms & Rationale

Four classifiers were compared, chosen to span the interpretability–complexity spectrum:

| Algorithm | Why Selected | Key Strengths |
|---|---|---|
| **Logistic Regression** | Strong linear baseline; highly interpretable; fast training; regularized via ElasticNet (`saga` solver) | Probability calibration, coefficient-based feature attribution, low overfitting risk |
| **Random Forest** | Robust ensemble; handles non-linear interactions; built-in feature importance | Variance reduction through bagging, minimal hyperparameter sensitivity |
| **XGBoost** | State-of-the-art gradient boosting; strong on tabular data; built-in regularization | Sequential error correction, native handling of missing values |
| **LightGBM** | Efficient gradient boosting; leaf-wise growth for better accuracy on smaller datasets | Speed, memory efficiency, histogram-based splitting |

### Hyperparameter Tuning

All models were tuned using **Optuna** with a Bayesian optimization strategy:

- **Trials per model:** 15
- **Cross-validation folds:** 3
- **Scoring metric:** Weighted F1-score
- **Search spaces:**

| Model | Tuned Parameters |
|---|---|
| Logistic Regression | `C` (1e-3 to 100, log), `l1_ratio` (0.0 to 1.0) |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| LightGBM | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |

---

## Model Evaluation Results

All metrics are computed on the **held-out test set (20%)** which was never seen during training or hyperparameter tuning.

### Performance Comparison

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | Precision (Weighted) | Recall (Weighted) | ROC-AUC (OVR) |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.8470** | **0.8499** | **0.8295** | **0.8554** | **0.8470** | **0.9607** |
| Random Forest | 0.8420 | 0.8448 | 0.8252 | 0.8494 | 0.8420 | 0.9568 |
| LightGBM | 0.8420 | 0.8443 | 0.8198 | 0.8480 | 0.8420 | 0.9540 |
| XGBoost | 0.8390 | 0.8412 | 0.8172 | 0.8446 | 0.8390 | 0.9545 |

### Best Model: Logistic Regression

| Metric | Value |
|---|---|
| Test Accuracy | 84.70% |
| Test F1 (Weighted) | 0.8499 |
| Test ROC-AUC (OVR) | 0.9607 |
| Best CV F1 | 0.8658 |
| Validation F1 | 0.8723 |
| Optimized `C` | 0.0097 |
| Optimized `l1_ratio` | 0.68 |

**Why Logistic Regression won:** Despite being the simplest model, it achieved the highest test F1 and ROC-AUC. The engineered features and proper scaling created a feature space where linear decision boundaries were sufficient. The strong regularization (`C ≈ 0.01`) prevented overfitting, and the high `l1_ratio` performed effective feature selection. The ensemble methods showed signs of slight overfitting (higher CV scores but lower test scores).

---

## Project Structure

```
supply_chain_risk_prediction/
├── app/
│   └── main.py                    # FastAPI prediction endpoint
├── data/
│   ├── raw/
│   │   └── supply_chain_data.csv  # Generated dataset (10,000 rows)
│   └── processed/
│       ├── label_encoders.pkl     # Fitted LabelEncoders
│       └── scaler.pkl             # Fitted StandardScaler
├── models/
│   ├── best_model.pkl             # Serialized best model
│   ├── logistic_regression.pkl    # Individual model artifacts
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   └── model_metadata.json        # Feature names, best model info, metrics
├── reports/
│   ├── evaluation_results.json    # Test metrics for all models
│   ├── training_results.json      # Hyperparameter tuning results
│   └── figures/
│       ├── confusion_matrix_*.png # Per-model confusion matrices
│       ├── feature_importance_*.png
│       ├── shap_summary_*.png     # SHAP explainability plots
│       └── model_comparison.png
├── src/
│   ├── config.py                  # Paths, constants, feature lists
│   ├── data_preprocessing.py      # Data generation, cleaning, splitting, SMOTE
│   ├── feature_engineering.py     # Derived features, scaling
│   ├── train_model.py             # Optuna tuning, model training
│   ├── evaluate_model.py          # Metrics, plots, SHAP analysis
│   └── inference.py               # SupplyChainPredictor class, demo
├── run_pipeline.py                # End-to-end pipeline orchestrator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── DOCUMENTATION.md               # Detailed technical documentation
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

```bash
# Clone and navigate to the project
cd supply_chain_risk_prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing, Logistic Regression, Random Forest, metrics |
| xgboost | XGBoost classifier |
| lightgbm | LightGBM classifier |
| imbalanced-learn | SMOTE oversampling |
| optuna | Bayesian hyperparameter optimization |
| shap | Model explainability |
| matplotlib, seaborn | Visualization |
| fastapi, uvicorn | REST API serving |
| joblib | Model serialization |

---

## Usage

### Run the Full Pipeline

```bash
python run_pipeline.py
```

This executes all five stages sequentially:

1. Generates (or loads) the dataset → `data/raw/supply_chain_data.csv`
2. Cleans, encodes, splits, and applies SMOTE
3. Engineers derived features and scales all splits
4. Trains 4 models with Optuna hyperparameter tuning (15 trials each)
5. Evaluates on test set, generates plots and SHAP analysis
6. Runs an inference demo with a sample supplier

**Expected runtime:** ~3–8 minutes depending on hardware.

**Output artifacts:**
- Trained models in `models/`
- Evaluation reports in `reports/`
- Visualization plots in `reports/figures/`

### Serve the Prediction API

```bash
# From the supply_chain_risk_prediction/ directory
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The API documentation is available at:
- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

### API Examples

#### Health Check

```bash
curl http://localhost:8001/health
```

```json
{"status": "healthy", "model_loaded": true}
```

#### Predict Supplier Risk

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "compliance_score": 60.0
  }'
```

```json
{
  "risk_level": "High",
  "risk_code": 2,
  "probabilities": {
    "Low": 0.0521,
    "Medium": 0.2734,
    "High": 0.6745
  }
}
```

#### Predict a Low-Risk Supplier

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "supplier_country": "Germany",
    "product_category": "Machinery",
    "shipping_mode": "Rail",
    "payment_terms": "Net 30",
    "on_time_delivery_rate": 0.97,
    "defect_rate": 0.005,
    "lead_time_days": 10,
    "order_quantity": 200,
    "unit_price": 150.0,
    "supplier_rating": 4.8,
    "financial_health_score": 92.0,
    "years_in_business": 25,
    "num_past_orders": 800,
    "avg_quality_score": 95.0,
    "distance_km": 500.0,
    "num_returns": 1,
    "compliance_score": 96.0
  }'
```

---

## Explainability

### SHAP Analysis

SHAP (SHapley Additive exPlanations) values are computed for the best model to provide global and local feature attribution. The SHAP summary plot reveals which features most influence high-risk predictions.

Key findings from the analysis:
- **`defect_rate`** and **`on_time_delivery_rate`** are consistently the strongest predictors
- **`financial_health_score`** and **`compliance_score`** contribute significantly through the `stability_score` engineered feature
- **`reliability_index`** (engineered) captures the joint effect of delivery and quality, ranking among the top features
- Categorical features (`supplier_country`, `product_category`) have lower but non-negligible impact

### Feature Importance

Both coefficient magnitudes (Logistic Regression) and tree-based feature importances are plotted for each model, saved in `reports/figures/`.

### Confusion Matrices

Per-model confusion matrices are generated to visualize classification performance across all three risk classes, highlighting where the model confuses adjacent risk levels.

---

## Limitations

1. **Synthetic data:** The dataset is algorithmically generated. While it captures realistic correlations, it cannot represent the full complexity of real-world supply chain dynamics (e.g., geopolitical events, seasonal patterns, supplier-specific idiosyncrasies).

2. **Static features:** The model treats each supplier as a single snapshot. Real supply chain risk is temporal — a supplier's risk profile changes over time with new orders, market shifts, and compliance updates.

3. **Class imbalance handling:** SMOTE generates synthetic minority samples in feature space, which can create unrealistic data points near class boundaries, potentially introducing noise.

4. **Limited categorical resolution:** Supplier country and product category are coarse-grained. Real systems would incorporate sub-categories, specific supplier identifiers, and richer geographic data.

5. **No external data integration:** The model does not incorporate external signals such as news sentiment, commodity prices, weather data, or regulatory changes that often drive supply chain risk.

6. **Binary encoding of categoricals:** Label encoding imposes an artificial ordinal relationship on categorical features. One-hot encoding or target encoding may yield better results for tree-based models.

7. **Single-point prediction:** The API returns point estimates without confidence intervals or prediction uncertainty quantification.

---

## Future Improvements

- **Temporal modeling:** Incorporate time-series features (rolling averages, trend indicators) and explore sequence models (LSTM, Transformer) for dynamic risk scoring.
- **Real data integration:** Connect to live ERP/procurement systems (SAP Ariba, Oracle) for real supplier data ingestion.
- **External risk signals:** Integrate news feeds, commodity price APIs, and weather data for holistic risk assessment.
- **Advanced encoding:** Replace label encoding with target encoding or embedding layers for categorical features.
- **Model monitoring:** Add drift detection (PSI, KS-test) and automated retraining triggers in production.
- **Uncertainty quantification:** Implement conformal prediction or Bayesian methods to provide prediction intervals alongside point estimates.
- **Multi-task learning:** Jointly predict risk level, expected delay, and potential financial impact.
- **Graph-based modeling:** Represent the supply chain as a network and use GNNs to capture supplier interdependencies.
- **A/B testing framework:** Enable controlled rollout of new models against the current production model.
- **Enhanced API:** Add batch prediction, async processing, model versioning, and authentication.

---

## License

This project is developed for educational and portfolio demonstration purposes.
