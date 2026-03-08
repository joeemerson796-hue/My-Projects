# Supply Chain Risk Prediction — Technical Documentation

This document provides a comprehensive deep-dive into the engineering decisions, ML workflow, and design rationale behind the Supply Chain Risk Prediction pipeline. It is intended for technical reviewers, interviewers, and collaborators who want to understand **why** each decision was made, not just **what** was built.

---

## Table of Contents

- [1. ML Engineering Workflow](#1-ml-engineering-workflow)
  - [1.1 Data Generation & Preprocessing](#11-data-generation--preprocessing)
  - [1.2 Feature Engineering](#12-feature-engineering)
  - [1.3 Model Training & Hyperparameter Tuning](#13-model-training--hyperparameter-tuning)
  - [1.4 Model Evaluation & Selection](#14-model-evaluation--selection)
  - [1.5 Explainability](#15-explainability)
  - [1.6 Inference & Deployment](#16-inference--deployment)
- [2. Design Decisions & Rationale](#2-design-decisions--rationale)
  - [2.1 Why Synthetic Data?](#21-why-synthetic-data)
  - [2.2 Why These Four Algorithms?](#22-why-these-four-algorithms)
  - [2.3 Why SMOTE Over Other Resampling Techniques?](#23-why-smote-over-other-resampling-techniques)
  - [2.4 Why Optuna for Hyperparameter Tuning?](#24-why-optuna-for-hyperparameter-tuning)
  - [2.5 Why Weighted F1 as the Primary Metric?](#25-why-weighted-f1-as-the-primary-metric)
  - [2.6 Why Logistic Regression Won](#26-why-logistic-regression-won)
  - [2.7 Why StandardScaler Over Other Scaling Methods?](#27-why-standardscaler-over-other-scaling-methods)
  - [2.8 Why Label Encoding for Categoricals?](#28-why-label-encoding-for-categoricals)
  - [2.9 Why a Three-Way Split Instead of Just Train/Test?](#29-why-a-three-way-split-instead-of-just-traintest)
  - [2.10 Why FastAPI for Serving?](#210-why-fastapi-for-serving)
- [3. Model Training Deep Dive](#3-model-training-deep-dive)
  - [3.1 Logistic Regression](#31-logistic-regression)
  - [3.2 Random Forest](#32-random-forest)
  - [3.3 XGBoost](#33-xgboost)
  - [3.4 LightGBM](#34-lightgbm)
  - [3.5 Model Tradeoffs Summary](#35-model-tradeoffs-summary)
- [4. Interview Questions & Answers](#4-interview-questions--answers)
- [5. Appendix](#5-appendix)

---

## 1. ML Engineering Workflow

### 1.1 Data Generation & Preprocessing

#### Data Generation Strategy

The dataset is generated using a **latent variable approach**. A single hidden risk factor (`latent_risk ~ N(0,1)`) drives the correlations across all features. Each observable feature is constructed as a function of this latent variable plus independent noise:

```
financial_health = 70 - 15 × latent_risk + N(0, 10)
on_time_delivery = 0.85 - 0.12 × latent_risk + N(0, 0.08)
defect_rate = 0.03 + 0.04 × latent_risk + N(0, 0.02)
```

This approach ensures that:
- Features are **realistically correlated** (high-risk suppliers tend to have low financial health AND high defect rates simultaneously)
- The problem is **learnable** but not trivially separable
- Noise injection makes the problem appropriately challenging

The final risk label is derived from a weighted combination of normalized feature values with percentile-based thresholds (55th and 82nd percentile), producing an intentionally **imbalanced** distribution (55% Low / 27% Medium / 18% High).

#### Missing Value Strategy

Missing values are injected at realistic rates:
- `financial_health_score`: ~3% missing (simulates incomplete financial disclosures)
- `supplier_rating`: ~2% missing (simulates new suppliers without ratings)

**Imputation:** Median imputation for numerical features, mode imputation for categoricals. Median is preferred over mean because it is robust to outliers — a supplier with an anomalously high financial health score should not inflate the imputed value for other suppliers.

#### Encoding

Categorical features are **label-encoded** using scikit-learn's `LabelEncoder`. The fitted encoders are serialized to `data/processed/label_encoders.pkl` to ensure consistent encoding at inference time.

#### Splitting

A **stratified three-way split** is performed:

```
Full Dataset (10,000)
  ├── Test (20%) → 2,000 samples — final evaluation only
  └── Train+Val (80%) → 8,000 samples
        ├── Validation (15% of 80% ≈ 12%) → ~1,200 samples — hyperparam selection
        └── Training (68%) → ~6,800 samples — model fitting
```

Stratification ensures each split preserves the original class distribution. The validation set is used during Optuna tuning to select the best hyperparameters; the test set is reserved exclusively for final, unbiased evaluation.

#### Class Imbalance: SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is applied **only to the training set** after splitting. This is critical — applying SMOTE before splitting would leak synthetic information into the validation and test sets, producing optimistically biased metrics.

SMOTE works by:
1. For each minority-class sample, finding its k nearest neighbors (default k=5)
2. Creating new synthetic samples along the line segments connecting the sample to its neighbors
3. Repeating until class balance is achieved

After SMOTE, all three classes have equal representation in the training set.

---

### 1.2 Feature Engineering

Six derived features are created to capture domain-relevant interactions:

| Feature | Formula | Domain Rationale |
|---|---|---|
| `total_order_value` | `order_quantity × unit_price` | Financial exposure — higher value orders amplify risk impact |
| `return_rate` | `num_returns / (num_past_orders + 1)` | Normalized return frequency; the +1 avoids division by zero for new suppliers |
| `reliability_index` | `on_time_delivery_rate × (1 − defect_rate)` | Joint reliability metric — a supplier must be both timely AND defect-free to score high |
| `stability_score` | `(financial_health + compliance_score) / 2` | Organizational stability; financial distress and compliance failures often co-occur |
| `delivery_efficiency` | `distance_km / (lead_time_days + 1)` | Logistics performance relative to distance; efficient suppliers deliver faster per km |
| `quality_rating_product` | `avg_quality_score × supplier_rating` | Amplifies the quality signal; two independent quality measures multiplied together |

**Why these specific features?** Each captures a non-linear interaction that a linear model cannot learn from raw features alone. For example, `reliability_index` creates a multiplicative interaction between delivery and quality — a supplier with 95% on-time delivery but 20% defect rate should score differently than one with 80% on-time and 2% defects, even if the linear sum is similar.

#### Feature Scaling

**StandardScaler** (zero mean, unit variance) is applied to all features. The scaler is fit **only on the training set** and applied (transform-only) to validation and test sets to prevent data leakage.

---

### 1.3 Model Training & Hyperparameter Tuning

Each of the four candidate models is tuned independently using **Optuna's Tree-structured Parzen Estimator (TPE)** algorithm:

1. Optuna proposes a set of hyperparameters from the defined search space
2. The model is trained and evaluated using **3-fold cross-validation** on the SMOTE-balanced training set
3. The cross-validation weighted F1-score is reported back to Optuna
4. Steps 1–3 repeat for 15 trials
5. The best hyperparameters are used to train a final model on the full training set
6. The final model is evaluated on the validation set for model comparison

The best model is selected based on **validation F1-score** (not CV score, to account for the SMOTE distribution shift).

#### Tuning Results Summary

| Model | Best CV F1 | Validation F1 | Key Parameters |
|---|---|---|---|
| Logistic Regression | 0.8658 | 0.8723 | C=0.0097, l1_ratio=0.68 |
| Random Forest | 0.8880 | 0.8622 | n_estimators=300, max_depth=13 |
| XGBoost | 0.9107 | 0.8626 | n_estimators=250, max_depth=9, lr=0.163 |
| LightGBM | 0.9105 | 0.8563 | n_estimators=300, max_depth=8, lr=0.140 |

Notice the **CV-to-validation gap** for tree-based models: XGBoost achieves 0.9107 on CV but only 0.8626 on validation. This suggests mild overfitting to the SMOTE-augmented training distribution. Logistic Regression shows the smallest gap (0.8658 → 0.8723), indicating superior generalization.

---

### 1.4 Model Evaluation & Selection

Final evaluation uses the **held-out test set** with multiple metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness (can be misleading with imbalanced classes) |
| F1-Score (Weighted) | Harmonic mean of precision and recall, weighted by class support — primary metric |
| F1-Score (Macro) | Unweighted average across classes — emphasizes minority class performance |
| Precision (Weighted) | Cost of false positives (flagging safe suppliers as risky) |
| Recall (Weighted) | Cost of false negatives (missing risky suppliers) |
| ROC-AUC (OVR) | Discrimination ability across all probability thresholds |

#### Final Test Results

| Model | Accuracy | F1 (W) | F1 (M) | Precision (W) | Recall (W) | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.847** | **0.850** | **0.830** | **0.855** | **0.847** | **0.961** |
| Random Forest | 0.842 | 0.845 | 0.825 | 0.849 | 0.842 | 0.957 |
| LightGBM | 0.842 | 0.844 | 0.820 | 0.848 | 0.842 | 0.954 |
| XGBoost | 0.839 | 0.841 | 0.817 | 0.845 | 0.839 | 0.954 |

**Winner: Logistic Regression** — highest on every single metric. Selected as the production model and serialized as `models/best_model.pkl`.

---

### 1.5 Explainability

#### SHAP Analysis

SHAP values are computed for the best model using `LinearExplainer` (or `TreeExplainer` for tree-based models). A summary plot is generated showing:
- **Feature ranking** by mean absolute SHAP value
- **Directional impact** — whether high/low feature values push predictions toward higher or lower risk

SHAP is chosen over simpler approaches (e.g., permutation importance) because it provides:
- Theoretically grounded (Shapley values from game theory) additive feature attributions
- Both global (aggregate) and local (per-prediction) explanations
- Consistent treatment of correlated features

#### Confusion Matrices

Per-model confusion matrices visualize misclassification patterns. Common patterns include:
- Medium-risk suppliers misclassified as Low (the most common error)
- High-risk suppliers rarely misclassified as Low (the most dangerous error, and correctly minimized)

#### Feature Importance Plots

For Logistic Regression, mean absolute coefficient values across classes are plotted. For tree-based models, Gini/gain-based importances are used.

---

### 1.6 Inference & Deployment

The `SupplyChainPredictor` class encapsulates the full inference pipeline:

```
Raw Input (dict or DataFrame)
  → Drop supplier_id
  → Label-encode categoricals (using fitted encoders)
  → Create derived features
  → Reorder columns to match training schema
  → Scale features (using fitted scaler)
  → Model prediction
  → Return risk_level, risk_code, probabilities
```

The FastAPI application (`app/main.py`) wraps this predictor with:
- **Pydantic validation** — input types, ranges, and required fields are validated before reaching the model
- **Health endpoint** (`GET /health`) — for load balancer probes
- **Prediction endpoint** (`POST /predict`) — accepts JSON, returns structured risk assessment
- **Auto-generated docs** — Swagger UI at `/docs`, ReDoc at `/redoc`

---

## 2. Design Decisions & Rationale

### 2.1 Why Synthetic Data?

**Decision:** Generate synthetic data rather than using a real dataset.

**Rationale:**
1. **Availability:** Real supply chain risk datasets with labeled outcomes are proprietary and rarely publicly available. Companies like Z2Data and Resilinc guard this data as a competitive moat.
2. **Controllability:** Synthetic generation allows precise control over class distributions, feature correlations, noise levels, and missing data patterns — enabling a well-defined ML problem.
3. **Reproducibility:** The `RANDOM_STATE = 42` seed ensures identical datasets across runs, making experiments fully reproducible.
4. **Realism:** The latent variable approach produces features that are correlated in domain-appropriate ways (e.g., financially unhealthy suppliers tend to also have poor delivery rates).

**Tradeoff:** Synthetic data cannot capture the full complexity of real supply chain dynamics. The model's performance metrics may not generalize to production data without retraining.

---

### 2.2 Why These Four Algorithms?

**Decision:** Compare Logistic Regression, Random Forest, XGBoost, and LightGBM.

**Rationale:** This set covers the **full spectrum of model complexity** for tabular classification:

| Category | Model | Rationale |
|---|---|---|
| Linear baseline | Logistic Regression | Establishes the performance floor; if it wins, complex models aren't justified |
| Bagging ensemble | Random Forest | Captures non-linear patterns; reduces variance through averaging |
| Gradient boosting | XGBoost | Sequential error correction; state-of-the-art tabular performance |
| Efficient boosting | LightGBM | Similar to XGBoost but faster; leaf-wise growth can find better splits |

**Why not deep learning?** For tabular data with ~10K samples and ~23 features, gradient-boosted trees and linear models consistently outperform neural networks (as demonstrated in the "Tabular Data: Deep Learning is Not All You Need" benchmark). Deep learning requires orders of magnitude more data to show advantages on structured tabular problems.

**Why not SVM?** SVMs scale poorly to multi-class problems (requiring one-vs-one or one-vs-rest), have longer training times with kernel methods, and provide less interpretable decision boundaries. Logistic Regression occupies the same "linear model" niche with better probability calibration.

---

### 2.3 Why SMOTE Over Other Resampling Techniques?

**Decision:** Use SMOTE for class imbalance handling.

**Alternatives considered:**

| Technique | Why Not Chosen |
|---|---|
| Random oversampling | Duplicates existing minority samples exactly, leading to overfitting |
| Random undersampling | Discards majority-class information — wasteful with only 10K samples |
| Class weights | Effective but doesn't increase the effective training set size |
| ADASYN | More complex than SMOTE with marginal improvement in practice |
| SMOTE-Tomek / SMOTE-ENN | Hybrid methods add complexity; cleaning step can remove useful boundary samples |

**Why SMOTE:** It generates synthetic minority samples by interpolating between existing samples and their nearest neighbors. This increases the effective training set size without exact duplication, providing the model with a more diverse set of minority-class examples.

**Critical implementation detail:** SMOTE is applied **after** the train/validation/test split and **only to the training set**. Applying it before splitting would introduce synthetic copies of validation/test samples into training, causing data leakage and inflated metrics.

---

### 2.4 Why Optuna for Hyperparameter Tuning?

**Decision:** Use Optuna with 15 trials per model and 3-fold cross-validation.

**Alternatives considered:**

| Technique | Why Not Chosen |
|---|---|
| Grid search | Exponentially expensive with many hyperparameters; wastes trials on uninformative regions |
| Random search | Better than grid but doesn't learn from previous trials |
| Bayesian optimization (sklearn) | Limited search space definitions; less flexible than Optuna |
| Hyperopt | Similar to Optuna but with a less ergonomic API |

**Why Optuna:**
1. **TPE (Tree-structured Parzen Estimator):** Models the search space as a probability distribution, focusing trials on promising regions. After 5–6 trials, it converges much faster than random search.
2. **Flexible search spaces:** Supports log-uniform, categorical, conditional, and integer parameters natively.
3. **Pruning support:** Can early-stop unpromising trials (not used here, but available for scaling up).
4. **Lightweight:** No external service required; runs in-process.

**Why 15 trials?** A pragmatic balance between thoroughness and runtime. For 2–8 hyperparameters, 15 Bayesian trials typically achieve 90%+ of the improvement that 100 random trials would provide. With 4 models × 15 trials × 3-fold CV, total training runs ≈ 180 — feasible in under 10 minutes on commodity hardware.

**Why 3-fold CV?** Three folds keeps each fold large enough (~2,200 training samples) to be representative, while keeping total computation at 3× rather than 5× or 10×.

---

### 2.5 Why Weighted F1 as the Primary Metric?

**Decision:** Optimize for weighted F1-score throughout the pipeline.

**Rationale:**

- **F1-score** balances precision and recall — important when both false positives (unnecessary supplier scrutiny) and false negatives (missed risky suppliers) have costs.
- **Weighted** averaging accounts for class imbalance by giving each class's F1-score a weight proportional to its support (number of true samples). This prevents the metric from being dominated by minority classes.
- **Macro F1** is also tracked but not used for optimization — it treats all classes equally regardless of support, which can overemphasize small classes.

**Why not accuracy?** With 55% of suppliers being Low-risk, a naive "always predict Low" classifier would achieve 55% accuracy. Weighted F1 penalizes this more harshly because precision and recall for Medium and High classes would be zero.

**Why not ROC-AUC?** ROC-AUC measures discrimination across all thresholds and is useful for ranking, but F1-score evaluates the actual predictions the model makes at a specific threshold — closer to the production use case.

---

### 2.6 Why Logistic Regression Won

This is perhaps the most interesting result. Despite XGBoost and LightGBM achieving higher **CV scores** (0.91 vs 0.87), Logistic Regression achieved the highest **test scores** (F1: 0.850 vs 0.841–0.845).

**Root causes:**

1. **Feature engineering created linear separability.** The derived features (`reliability_index`, `stability_score`, etc.) explicitly encode the non-linear interactions that tree-based models would need to discover. With these features, the problem becomes approximately linearly separable.

2. **Strong regularization.** The optimized `C = 0.0097` (effectively very strong L1/L2 regularization with `l1_ratio = 0.68`) performs automatic feature selection, focusing on the most informative features and ignoring noise.

3. **SMOTE-induced overfitting in tree models.** SMOTE creates synthetic samples in feature space that are structurally different from the original data distribution. Tree-based models, being more flexible, can memorize these artifacts. Logistic Regression's simpler decision boundary is more robust to distribution shift between SMOTE-augmented training data and natural test data.

4. **Better probability calibration.** Logistic Regression produces inherently well-calibrated probabilities, contributing to stronger ROC-AUC (0.961 vs 0.954–0.957).

**Key takeaway:** Model complexity should match problem complexity. More sophisticated models aren't always better — especially when thoughtful feature engineering reduces the need for the model to learn complex interactions.

---

### 2.7 Why StandardScaler Over Other Scaling Methods?

**Decision:** Use StandardScaler (z-score normalization).

| Scaler | When to Use | Why Not Here |
|---|---|---|
| StandardScaler ✅ | Features are approximately Gaussian | Most numerical features follow roughly normal distributions |
| MinMaxScaler | Bounded features, neural networks | Not all features are bounded; sensitive to outliers |
| RobustScaler | Heavy outliers present | Outliers are moderate (clipped during generation) |
| No scaling | Tree-based models only | Logistic Regression requires scaling for convergence |

StandardScaler is essential for Logistic Regression (gradient-based optimization is sensitive to feature scales) and does not hurt tree-based models (which are scale-invariant). Using a single scaler across all models simplifies the pipeline.

---

### 2.8 Why Label Encoding for Categoricals?

**Decision:** Label encoding rather than one-hot encoding.

**Rationale:**
- With 4 categorical features having 4–8 categories each, one-hot encoding would add ~22 binary columns, significantly expanding the feature space
- Label encoding keeps the feature space compact (4 features vs 22)
- For tree-based models, label encoding works well because trees can learn arbitrary splits on encoded values
- For Logistic Regression, the imposed ordinal relationship is imperfect but mitigated by regularization

**Tradeoff:** Label encoding introduces a false ordinal relationship (e.g., "China" < "Germany" < "USA"). This can cause Logistic Regression to learn spurious linear trends. A more robust approach would be target encoding or embedding-based encoding.

---

### 2.9 Why a Three-Way Split Instead of Just Train/Test?

**Decision:** Train (68%) / Validation (12%) / Test (20%).

The validation set serves a specific purpose that cross-validation alone cannot fill:

1. **Cross-validation** evaluates hyperparameters on the SMOTE-augmented training distribution
2. The **validation set** evaluates the final model on the natural (non-SMOTE) distribution
3. The **test set** provides the final unbiased metric reported to stakeholders

Without a validation set, model selection would be based solely on CV scores from the SMOTE-augmented distribution, which (as we saw) can be misleading — XGBoost's CV F1 of 0.91 did not translate to test F1 of 0.91.

---

### 2.10 Why FastAPI for Serving?

**Decision:** FastAPI over Flask, Django, or gRPC.

| Framework | Why Not |
|---|---|
| Flask | No native async, no auto-generated docs, manual validation |
| Django | Heavyweight for a single-endpoint ML service |
| gRPC | Better for microservice-to-microservice; less accessible for external clients |
| **FastAPI** ✅ | Async-native, Pydantic validation, auto OpenAPI docs, excellent performance |

FastAPI's Pydantic integration means input validation (type checking, range constraints like `ge=0, le=1` for rates) is declarative and automatic — no manual validation code needed.

---

## 3. Model Training Deep Dive

### 3.1 Logistic Regression

**Configuration:** `saga` solver with ElasticNet penalty.

The `saga` solver is chosen because it:
- Supports ElasticNet regularization (combined L1 + L2)
- Handles multi-class classification natively (multinomial)
- Is efficient for medium-sized datasets

**Key hyperparameters:**
- `C = 0.0097`: Inverse regularization strength. Very low C means very strong regularization — the model is heavily constrained, preventing overfitting.
- `l1_ratio = 0.68`: 68% L1, 32% L2. The L1 component drives many coefficients to exactly zero (feature selection), while L2 prevents remaining coefficients from exploding.
- `max_iter = 2000`: Generous iteration limit to ensure convergence with strong regularization.

**Interpretation:** The low C value means the model is essentially performing aggressive feature selection and learning a simple, highly regularized linear boundary. This explains its robustness — it ignores noisy features and focuses on the strongest signals.

---

### 3.2 Random Forest

**Configuration:** Bagging ensemble of 300 decision trees.

**Key hyperparameters:**
- `n_estimators = 300`: Number of trees. More trees reduce variance without increasing bias; 300 is where diminishing returns typically begin.
- `max_depth = 13`: Maximum tree depth. Deeper trees capture more complex interactions but risk overfitting. 13 is a moderate depth.
- `min_samples_split = 5`: Minimum samples to split an internal node. Prevents overly specific splits.
- `min_samples_leaf = 2`: Minimum samples in leaf nodes. Prevents single-sample leaves.
- `max_features = "log2"`: Features considered at each split. Using log2 (≈4.5 of 23 features) forces diversity among trees.

**Why it didn't win:** Random Forest showed a smaller CV-to-test gap than XGBoost/LightGBM (0.888 → 0.845) but still overfits slightly to the SMOTE distribution. Its bagging strategy provides variance reduction but cannot match Logistic Regression's simplicity when the problem is approximately linear.

---

### 3.3 XGBoost

**Configuration:** Gradient-boosted trees with multi-class softmax objective.

**Key hyperparameters:**
- `n_estimators = 250`: Boosting rounds. Each tree corrects the residual errors of the ensemble so far.
- `max_depth = 9`: Per-tree depth limit. Deeper than Random Forest because each tree is a weak learner.
- `learning_rate = 0.163`: Step size shrinkage. Higher than typical (0.01–0.1) but appropriate with only 250 trees.
- `subsample = 0.726`: Row sampling per tree. Adds stochastic regularization.
- `colsample_bytree = 0.609`: Column sampling per tree. Forces diverse feature usage.
- `reg_alpha = 0.037`: L1 regularization on leaf weights.
- `reg_lambda = 0.072`: L2 regularization on leaf weights.

**Why it didn't win:** XGBoost achieved the highest CV F1 (0.9107) but the lowest test F1 (0.841). This is the classic sign of overfitting to the training distribution — likely to the synthetic samples generated by SMOTE, which tree-based models can memorize through deep, specific splits.

---

### 3.4 LightGBM

**Configuration:** Leaf-wise gradient boosting with multiclass objective.

**Key hyperparameters:**
- `n_estimators = 300`: Boosting rounds.
- `max_depth = 8`: Depth limit.
- `learning_rate = 0.140`: Step size.
- `num_leaves = 52`: Maximum leaves per tree. LightGBM uses leaf-wise growth (vs. XGBoost's level-wise), so `num_leaves` is the primary complexity control.
- `subsample = 0.997`: Nearly full row sampling (almost no stochastic regularization).
- `colsample_bytree = 0.765`: Moderate column sampling.
- `reg_alpha = 0.024`, `reg_lambda = 2.272`: L1 and L2 regularization. The higher L2 (`reg_lambda`) suggests Optuna found that LightGBM benefits from stronger weight penalization.

**Why it didn't win:** Similar overfitting pattern to XGBoost. The high `subsample` (0.997) means almost no stochastic regularization, and `num_leaves = 52` allows moderately complex trees that can memorize SMOTE artifacts.

---

### 3.5 Model Tradeoffs Summary

| Dimension | Log. Regression | Random Forest | XGBoost | LightGBM |
|---|---|---|---|---|
| Test F1 | **0.850** | 0.845 | 0.841 | 0.844 |
| ROC-AUC | **0.961** | 0.957 | 0.954 | 0.954 |
| Interpretability | **High** (coefficients) | Medium (importance) | Medium (importance) | Medium (importance) |
| Training speed | **Fast** (~seconds) | Moderate (~30s) | Moderate (~45s) | Moderate (~30s) |
| Inference latency | **Lowest** | Medium | Medium | Medium |
| Overfitting risk | **Lowest** | Moderate | Highest | High |
| Handles non-linearity | No (needs FE) | **Yes** | **Yes** | **Yes** |
| Model size (serialized) | **Smallest** (~KB) | Large (~MB) | Medium (~MB) | Medium (~MB) |
| Production readiness | **Best** for this task | Good | Good | Good |

---

## 4. Interview Questions & Answers

### Q1: Why did you choose a synthetic dataset instead of a real one?

**Answer:** Real supply chain risk datasets with labeled outcomes are proprietary (held by companies like Z2Data, Resilinc, Interos) and not publicly available. I generated synthetic data using a latent variable approach that creates realistic feature correlations — for example, financially unhealthy suppliers also tend to have poor delivery rates and high defect rates, which mirrors real-world dynamics. The synthetic approach also gives me full control over class distributions, noise levels, and reproducibility (via a fixed random seed). In a production setting, this pipeline is designed to be retrained on real organizational data.

### Q2: How did you handle class imbalance, and why did you choose that approach?

**Answer:** I used SMOTE (Synthetic Minority Over-sampling Technique), applied only to the training set after the train/validation/test split. SMOTE generates synthetic minority samples by interpolating between existing samples and their k-nearest neighbors, rather than duplicating them (which would cause overfitting). Applying SMOTE after splitting is critical — if applied before, synthetic copies of validation/test samples could leak into training, inflating metrics. I chose SMOTE over random undersampling (which would discard 37% of data), class weights (which don't increase the effective training set size), and ADASYN (marginal improvement with more complexity).

### Q3: Why did Logistic Regression outperform ensemble methods?

**Answer:** This is the most interesting finding. Three factors contributed: (1) Feature engineering created approximate linear separability — derived features like `reliability_index` and `stability_score` explicitly encode the non-linear interactions that tree models would need to discover. (2) Strong ElasticNet regularization (C=0.0097, l1_ratio=0.68) performed aggressive feature selection, focusing on the strongest signals. (3) Tree-based models overfit to SMOTE artifacts — their CV scores were 0.91 but test scores were 0.84, while Logistic Regression showed minimal CV-to-test degradation. The lesson: model complexity should match problem complexity; feature engineering can reduce the need for complex models.

### Q4: Explain your hyperparameter tuning strategy. Why Optuna? Why 15 trials?

**Answer:** I used Optuna with its TPE (Tree-structured Parzen Estimator) sampler, which is a Bayesian optimization method. Unlike grid search (exponentially expensive) or random search (doesn't learn from previous trials), TPE builds a probabilistic model of the hyperparameter space and focuses trials on promising regions. 15 trials per model is a pragmatic choice: for 2–8 hyperparameters, Bayesian search typically achieves 90%+ of the improvement of 100+ random trials. With 3-fold CV, this means ~180 total training runs across all models, completing in minutes rather than hours.

### Q5: How would you deploy this model in production? What would you change?

**Answer:** The current FastAPI endpoint is a good starting point, but production deployment would require: (1) Containerization with Docker, health checks, and graceful shutdown. (2) Model versioning with MLflow or a model registry. (3) Input monitoring — drift detection (PSI, KS-test) on feature distributions to trigger retraining. (4) Prediction logging for auditing and feedback loops. (5) A/B testing framework to safely roll out new model versions. (6) Batch prediction support for periodic scoring of the full supplier base. (7) Authentication and rate limiting. (8) Horizontal scaling behind a load balancer for high-throughput scenarios.

### Q6: What metrics did you use, and why weighted F1 as the primary metric?

**Answer:** I tracked accuracy, weighted F1, macro F1, weighted precision, weighted recall, and ROC-AUC. Weighted F1 was chosen as the primary optimization metric because it balances precision and recall (both false positives and false negatives have business costs) while accounting for class imbalance through support-weighted averaging. Accuracy alone would be misleading — a naive "always predict Low" classifier achieves 55% accuracy. ROC-AUC is great for measuring discrimination ability but evaluates across all thresholds rather than at the operating point. Macro F1 was tracked to ensure minority classes weren't being ignored.

### Q7: How does your feature engineering pipeline prevent data leakage?

**Answer:** Multiple safeguards are in place: (1) The data is split into train/validation/test **before** any transformations. (2) SMOTE is applied **only to the training set**. (3) StandardScaler is fit only on training data and applied (transform-only) to validation and test sets. (4) Label encoders are fit on the full dataset before splitting (acceptable since they only map categories to integers, introducing no statistical information). (5) Feature engineering creates derived features using only same-row information (no aggregations across rows). (6) The inference pipeline applies the exact same sequence of transformations using serialized preprocessing artifacts.

### Q8: Why did you use StandardScaler instead of other normalization methods?

**Answer:** StandardScaler (z-score normalization) was chosen because: (1) Logistic Regression with gradient-based optimization (`saga` solver) requires features on similar scales for stable convergence. (2) Most numerical features in the dataset are approximately Gaussian-distributed (by construction). (3) It preserves outlier information better than MinMaxScaler. (4) Tree-based models are invariant to scaling, so using StandardScaler for all models simplifies the pipeline without hurting ensemble performance. RobustScaler was considered but unnecessary since outliers are clipped during data generation.

### Q9: What would you do differently with more data (e.g., 1M+ samples)?

**Answer:** Several things would change: (1) SMOTE might not be necessary — with sufficient data, natural class frequencies provide enough minority-class examples. (2) I'd explore deep learning — tabular transformers (FT-Transformer, TabNet) start outperforming tree-based methods at large scale. (3) I'd use more sophisticated categorical encoding (target encoding, learned embeddings). (4) Hyperparameter tuning could use more trials (50–100) with early pruning. (5) I'd add temporal features if the data includes timestamps, enabling time-series modeling of supplier risk trajectories. (6) I'd implement stratified mini-batch training for memory efficiency.

### Q10: How do you ensure the model is interpretable for business stakeholders?

**Answer:** Multiple layers of interpretability: (1) SHAP values provide both global feature ranking and local per-prediction explanations (e.g., "This supplier is High-risk primarily because of a defect rate 3x above average and declining financial health"). (2) Logistic Regression's coefficients directly indicate feature importance and direction. (3) Confusion matrices show where the model makes errors, helping stakeholders understand its blind spots. (4) The API returns probability distributions across all three risk levels, not just a single label — allowing stakeholders to see borderline cases. (5) Feature importance plots provide a quick visual summary for non-technical audiences.

### Q11: What are the risks of using SMOTE, and how would you mitigate them?

**Answer:** SMOTE risks include: (1) Generating synthetic samples in unrealistic regions of feature space, especially near class boundaries. (2) Amplifying noise if original minority samples are mislabeled. (3) Not accounting for within-class diversity (a cluster of minority samples in one region gets overrepresented while a sparse region stays sparse). Mitigation strategies: (a) SMOTE-Tomek or SMOTE-ENN to clean boundaries after oversampling. (b) Monitoring the CV-to-test gap to detect SMOTE-induced overfitting (as we observed with tree-based models). (c) In production, class-weighted loss functions might be preferable to avoid distribution shift entirely.

### Q12: How would you monitor this model in production?

**Answer:** I'd implement: (1) **Feature drift detection** — compute Population Stability Index (PSI) and Kolmogorov-Smirnov tests on incoming feature distributions vs. training distribution. (2) **Prediction drift** — monitor the distribution of predicted risk levels; sudden shifts indicate data or model problems. (3) **Performance monitoring** — track downstream outcomes (e.g., did "Low-risk" suppliers actually perform well?) to measure real-world accuracy. (4) **Latency monitoring** — track inference time P50/P95/P99. (5) **Automated retraining triggers** — when drift exceeds thresholds, automatically retrain on recent data. (6) **Shadow mode** — new model versions serve predictions in parallel with the production model for comparison before cutover.

### Q13: Why didn't you use neural networks or deep learning?

**Answer:** For tabular data with ~10K samples and ~23 features, classical ML consistently outperforms deep learning. This has been empirically demonstrated in benchmarks like "Tabular Data: Deep Learning is Not All You Need" (Shwartz-Ziv & Armon, 2022) and "Why do tree-based models still outperform deep learning on tabular data?" (Grinsztajn et al., 2022). Neural networks require orders of magnitude more data to show advantages on structured tabular problems. Additionally, tree-based models and logistic regression offer better interpretability, faster training, and lower serving latency — all important for production supply chain applications.

### Q14: Walk me through what happens when a prediction request hits your API.

**Answer:** Step by step: (1) FastAPI receives the POST request and Pydantic validates all 17 input fields — checking types, ranges (e.g., `on_time_delivery_rate` must be between 0 and 1), and required fields. Invalid inputs return a 400 error immediately. (2) The `SupplyChainPredictor.predict()` method is called. (3) The input dict is converted to a DataFrame. (4) Categorical features are label-encoded using the same encoders from training. (5) Six derived features are computed (total_order_value, return_rate, etc.). (6) Features are reordered to match the training schema (23 features). (7) StandardScaler transforms features using training-set statistics. (8) The Logistic Regression model produces class probabilities via `predict_proba()`. (9) The argmax class becomes the predicted risk level. (10) The API returns the risk label, risk code, and probability distribution as JSON.

### Q15: How would you handle concept drift in a production supply chain risk system?

**Answer:** Concept drift means the relationship between features and risk changes over time — for example, a pandemic might make delivery times less predictive of risk because everyone is delayed. My approach: (1) **Detection:** Regularly compare incoming feature distributions against the training distribution using PSI. For label drift, track prediction distribution changes. (2) **Windowed retraining:** Retrain on a rolling window (e.g., last 12 months) rather than all historical data, so the model adapts to recent patterns. (3) **Online learning:** For gradual drift, incrementally update model weights rather than full retraining. (4) **Ensemble with recency weighting:** Maintain multiple models trained on different time windows and weight their predictions by recency. (5) **Human-in-the-loop:** Flag predictions where model confidence is low for expert review, creating a feedback loop that captures ground truth.

---

## 5. Appendix

### A. Reproducibility

All experiments are fully reproducible with `RANDOM_STATE = 42`:

```bash
python run_pipeline.py
```

The pipeline will generate identical data, splits, SMOTE samples, and tuning results across runs (assuming the same Python and package versions).

### B. Key Configuration (src/config.py)

| Parameter | Value | Description |
|---|---|---|
| `RANDOM_STATE` | 42 | Global seed for reproducibility |
| `TEST_SIZE` | 0.2 | Fraction of data reserved for test set |
| `VAL_SIZE` | 0.15 | Fraction of train+val reserved for validation |
| `N_OPTUNA_TRIALS` | 15 | Hyperparameter search iterations per model |
| `CV_FOLDS` | 3 | Cross-validation folds for tuning |

### C. Dependency Versions

See `requirements.txt` for minimum supported versions. Key dependencies:

| Package | Min Version | Purpose |
|---|---|---|
| scikit-learn | 1.3.0 | Preprocessing, LR, RF, metrics |
| xgboost | 2.0.0 | XGBoost classifier |
| lightgbm | 4.0.0 | LightGBM classifier |
| imbalanced-learn | 0.11.0 | SMOTE |
| optuna | 3.4.0 | Hyperparameter tuning |
| shap | 0.43.0 | Explainability |
| fastapi | 0.104.0 | REST API |

### D. Generated Reports

After running the pipeline, the following reports are available:

| File | Contents |
|---|---|
| `reports/training_results.json` | Best hyperparameters and CV/validation F1 per model |
| `reports/evaluation_results.json` | Full test-set metrics per model |
| `models/model_metadata.json` | Best model name, feature list, test metrics |
| `reports/figures/confusion_matrix_*.png` | Per-model confusion matrices |
| `reports/figures/feature_importance_*.png` | Per-model feature importance plots |
| `reports/figures/shap_summary_*.png` | SHAP analysis for the best model |
| `reports/figures/model_comparison.png` | Bar chart comparing all models |
