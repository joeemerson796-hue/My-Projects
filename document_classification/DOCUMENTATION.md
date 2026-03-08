# Document Classification — Detailed Engineering Documentation

This document provides an in-depth technical walkthrough of the AI Document Classification system, covering the full ML engineering workflow, design rationale, tradeoff analysis, and common interview questions with detailed answers.

---

## Table of Contents

- [1. ML Engineering Workflow](#1-ml-engineering-workflow)
  - [1.1 Problem Definition and Scoping](#11-problem-definition-and-scoping)
  - [1.2 Data Collection and Exploration](#12-data-collection-and-exploration)
  - [1.3 Data Preprocessing](#13-data-preprocessing)
  - [1.4 Feature Engineering](#14-feature-engineering)
  - [1.5 Model Selection and Training](#15-model-selection-and-training)
  - [1.6 Evaluation and Model Comparison](#16-evaluation-and-model-comparison)
  - [1.7 Inference and Serving](#17-inference-and-serving)
- [2. Design Decisions and Rationale](#2-design-decisions-and-rationale)
  - [2.1 Dataset Choice](#21-dataset-choice)
  - [2.2 Category Selection](#22-category-selection)
  - [2.3 NLP Preprocessing Decisions](#23-nlp-preprocessing-decisions)
  - [2.4 TF-IDF Configuration](#24-tf-idf-configuration)
  - [2.5 Algorithm Selection](#25-algorithm-selection)
  - [2.6 TF-IDF vs. BERT Tradeoffs](#26-tf-idf-vs-bert-tradeoffs)
  - [2.7 Hyperparameter Tuning Strategy](#27-hyperparameter-tuning-strategy)
  - [2.8 Evaluation Metric Selection](#28-evaluation-metric-selection)
- [3. How Models Were Trained](#3-how-models-were-trained)
  - [3.1 TF-IDF + Logistic Regression](#31-tf-idf--logistic-regression)
  - [3.2 TF-IDF + Linear SVM](#32-tf-idf--linear-svm)
  - [3.3 DistilBERT Fine-Tuning](#33-distilbert-fine-tuning)
- [4. Interview Questions and Answers](#4-interview-questions-and-answers)

---

## 1. ML Engineering Workflow

### 1.1 Problem Definition and Scoping

**Business problem:** Organizations handling large volumes of technical documentation need automated routing and categorization. Manual classification is expensive, slow, inconsistent across reviewers, and does not scale with document volume.

**ML formulation:** Multi-class text classification with 8 categories. Each document is assigned exactly one category label. The problem is treated as a single-label classification task (no multi-label).

**Success criteria defined upfront:**
- Weighted F1 score ≥ 0.70 on the held-out test set (achieved by all TF-IDF models).
- Inference latency under 100ms per document for real-time API serving.
- Clean separation of training, evaluation, and serving code for maintainability.

**Scope boundaries:**
- English-language documents only.
- Eight predefined technical categories; no open-ended topic discovery.
- No online/incremental learning; batch retraining only.

### 1.2 Data Collection and Exploration

The 20 Newsgroups dataset was loaded using `sklearn.datasets.fetch_20newsgroups` with the following parameters:

```python
fetch_20newsgroups(
    subset="train",
    categories=SELECTED_CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=42,
)
```

**Key exploration findings:**
- **Document lengths** vary widely, from a few words to several thousand tokens. The median is roughly 100–200 words after cleaning.
- **Class distribution** is approximately balanced across the 8 selected categories, with minor variation. No aggressive resampling was required.
- **Noise level** is high: many documents contain email artifacts, ASCII art, code snippets, URL fragments, and non-English characters.
- **Header removal** is critical — without it, models achieve artificially high accuracy by memorizing sender names and email routing metadata rather than learning topical content.

Data is saved to `data/raw/train.csv` and `data/raw/test.csv` for reproducibility, with columns `text`, `label`, and `category`.

### 1.3 Data Preprocessing

The NLP preprocessing pipeline is implemented in `src/data_preprocessing.py` and applies four sequential transformations:

#### Step 1: Text Cleaning (`clean_text`)

```python
text = text.lower()                              # Case normalization
text = re.sub(r"<[^>]+>", " ", text)             # HTML tag removal
text = re.sub(r"http\S+|www\.\S+", " ", text)   # URL removal
text = re.sub(r"\S+@\S+", " ", text)            # Email address removal
text = re.sub(r"[^a-zA-Z\s]", " ", text)        # Non-alphabetic removal
text = re.sub(r"\s+", " ", text).strip()         # Whitespace normalization
```

**Rationale:** These steps remove information that is either noise (HTML, URLs, emails) or would cause the model to learn spurious correlations (e.g., predicting `comp.os.ms-windows.misc` because the post was sent from a microsoft.com email address).

#### Step 2: Tokenization

NLTK's Punkt tokenizer splits cleaned text into word tokens. Punkt is rule-based and handles abbreviations, contractions, and edge cases better than simple whitespace splitting.

#### Step 3: Stopword Removal

English stopwords from the NLTK corpus (179 words including "the", "is", "at", etc.) are removed. Additionally, tokens shorter than 3 characters are dropped — these are typically fragments, single letters, or two-character noise left after cleaning.

**Why remove stopwords before TF-IDF?** While TF-IDF's IDF component naturally downweights corpus-wide stopwords, explicit removal reduces the feature space and prevents stopwords from consuming slots in the `max_features=10000` budget. This makes bigram features more informative (e.g., "hard drive" instead of "the hard").

#### Step 4: Lemmatization

WordNet lemmatization reduces inflected forms to their base (lemma) form:
- "running" → "running" (without POS tag, default noun lemmatization)
- "drives" → "drive"
- "graphics" → "graphic"

**Why lemmatization over stemming?** Lemmatization produces valid dictionary words, preserving interpretability. The Porter Stemmer, by contrast, often produces non-words (e.g., "comput", "studi") that are harder to inspect when debugging model behavior.

#### Empty Document Handling

After preprocessing, documents that become empty strings (due to being entirely noise) are dropped. This is a small minority of the dataset but prevents zero-vector inputs from polluting training.

### 1.4 Feature Engineering

#### TF-IDF Vectorization

The TF-IDF vectorizer is configured in `src/feature_engineering.py`:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
)
```

**Parameter-by-parameter rationale:**

| Parameter | Value | Why |
|---|---|---|
| `max_features=10000` | Vocabulary cap | Prevents memory explosion from bigrams while retaining the most discriminative terms. Larger values (50K+) showed diminishing returns in experiments. |
| `ngram_range=(1, 2)` | Unigrams + bigrams | Bigrams capture meaningful phrases ("hard drive", "ray tracing", "space shuttle") that unigrams alone cannot. Trigrams were tested but added noise without improving F1. |
| `min_df=3` | Minimum document frequency | Drops terms appearing in fewer than 3 documents. These are typically typos, proper nouns, or extremely rare jargon that cannot generalize. |
| `max_df=0.95` | Maximum document frequency | Drops terms appearing in 95%+ of documents, serving as a second-pass stopword filter for terms missed by NLTK's list. |
| `sublinear_tf=True` | Logarithmic TF scaling | Replaces raw term frequency with `1 + log(tf)`. A term appearing 100 times in a document is not 100x more important than one appearing once; sublinear scaling captures this diminishing return. |

The fitted vectorizer is serialized to `data/processed/tfidf_vectorizer.pkl` for use during inference without retraining.

#### BERT Tokenization

For DistilBERT, the pipeline uses the pretrained `distilbert-base-uncased` WordPiece tokenizer:
- Input: cleaned text (before lemmatization, since BERT's pretrained vocabulary expects natural language)
- Max length: 256 tokens (documents longer than this are truncated)
- Padding: dynamic padding to the longest sequence in each batch
- Output: `input_ids` and `attention_mask` tensors

### 1.5 Model Selection and Training

Three models were trained to compare classical and deep-learning approaches:

1. **Logistic Regression** — a probabilistic linear model, strong baseline for text
2. **Linear SVM (LinearSVC)** — a maximum-margin linear classifier, historically dominant for text
3. **DistilBERT** — a distilled transformer model with contextual embeddings

This comparison was intentional: it demonstrates the classic ML engineering tradeoff between **simplicity/speed** (TF-IDF models) and **representational power** (transformers), and shows that the "best" model depends on constraints (compute, data size, latency budget).

### 1.6 Evaluation and Model Comparison

Evaluation uses `src/evaluate_model.py` and computes:

- **Accuracy** — proportion of correct predictions (intuitive but misleading with imbalanced classes)
- **F1 Score (weighted)** — harmonic mean of precision and recall, weighted by class support. This is the primary metric because it accounts for both false positives and false negatives while being robust to mild class imbalance.
- **F1 Score (macro)** — unweighted mean across classes, giving equal importance to every category regardless of size.
- **Precision (weighted)** — weighted proportion of true positives among positive predictions.
- **Recall (weighted)** — weighted proportion of actual positives that were correctly identified.

Confusion matrices are generated for each TF-IDF model to identify which category pairs are most confused.

A bar chart comparison of all three models' weighted F1 scores is saved to `reports/figures/model_comparison.png`.

### 1.7 Inference and Serving

The `DocumentClassifier` class in `src/inference.py` provides a unified interface:

```python
classifier = DocumentClassifier(backend="tfidf")  # or "bert" or "auto"
result = classifier.predict("Some technical document text...")
# Returns: {"category": "sci.space", "label": 6, "probabilities": {...}}
```

**Backend selection logic:**
- `"auto"` reads `model_metadata.json` and selects BERT if the best model is BERT-based, otherwise TF-IDF.
- `"tfidf"` loads the Logistic Regression model and TF-IDF vectorizer.
- `"bert"` loads the fine-tuned DistilBERT model and tokenizer.

The FastAPI app (`app/main.py`) wraps this classifier with:
- **Startup event:** model loaded once at server start, not per-request.
- **Input validation:** Pydantic schema enforces minimum text length of 10 characters.
- **Health endpoint:** `GET /health` for load balancer health checks.
- **Classification endpoint:** `POST /classify` for document classification.
- **Error handling:** returns 503 if model not loaded, 400 for invalid input.

---

## 2. Design Decisions and Rationale

### 2.1 Dataset Choice

**Decision:** Use the 20 Newsgroups dataset from scikit-learn.

**Alternatives considered:**
| Dataset | Pros | Cons | Decision |
|---|---|---|---|
| 20 Newsgroups | Real text, multi-class, well-studied, built into sklearn | Dated (1990s), noisy | **Selected** |
| AG News | Large (120K), clean | Only 4 classes, news-specific | Too few classes |
| IMDB Reviews | Large, high quality | Binary classification, sentiment not topics | Wrong task type |
| Reuters-21578 | Classic benchmark | Skewed distribution, multi-label | Added complexity |
| Custom web-scraped | Domain-specific | No ground truth, legal concerns | Not reproducible |

**Key reasons for 20 Newsgroups:**
1. **8 natural technical categories** align with the document classification use case.
2. **Noise and messiness** make it realistic — a model that works here will handle real-world input.
3. **Reproducibility** — anyone can download it with one function call and verify results.
4. **Community baselines** — published results exist for comparison.

### 2.2 Category Selection

Eight categories were selected from the original twenty to create a focused technical classification scenario:

- **4 computer categories:** graphics, Windows OS, IBM PC hardware, Mac hardware
- **4 science categories:** cryptography, electronics, medicine, space

This selection was intentional to create **both inter-domain and intra-domain challenges:**
- **Intra-domain confusion:** `comp.sys.ibm.pc.hardware` vs. `comp.sys.mac.hardware` share vocabulary about hardware components (RAM, CPU, disk). The model must learn subtle brand-specific and context-specific differences.
- **Inter-domain separation:** `sci.med` vs. `comp.graphics` use almost entirely different vocabularies, making them easy to separate. This gives the model achievable "wins" that boost overall accuracy while the hard categories drag it down.

### 2.3 NLP Preprocessing Decisions

| Decision | Alternative | Why this choice |
|---|---|---|
| Lowercase all text | Preserve case | Few technical terms rely on case for meaning in this dataset; lowercasing reduces vocabulary size substantially. |
| Remove all non-alphabetic characters | Keep numbers | Numbers like model numbers (e.g., "486", "8800") could help distinguish hardware categories, but they also add noise. Removing them simplifies the pipeline; the tradeoff is acceptable given TF-IDF's vocabulary size. |
| NLTK lemmatizer | spaCy lemmatizer, Porter stemmer | NLTK is a lighter dependency than spaCy. Lemmatization over stemming for readability (see Section 1.3). |
| Remove stopwords explicitly | Rely on TF-IDF max_df | Explicit removal frees feature slots for more informative terms when `max_features` is capped. |
| Use cleaned (non-lemmatized) text for BERT | Use lemmatized text for BERT | BERT's pretrained vocabulary expects natural English. Lemmatized text ("gpu render graphic") degrades BERT's tokenization and contextual embeddings because it breaks the distributional patterns learned during pretraining. |

### 2.4 TF-IDF Configuration

The TF-IDF vectorizer was configured after iterative experimentation:

**`max_features=10000`:** Testing with 5K, 10K, 20K, and 50K features showed that F1 plateaued around 10K for both Logistic Regression and SVM. Beyond 10K, additional features were predominantly rare bigrams that added noise.

**`ngram_range=(1,2)`:** Unigrams alone achieve ~71% F1; adding bigrams pushes this to ~74%. The ~3-point improvement comes from discriminative phrases. Adding trigrams to `(1,3)` increased the vocabulary by 5x without meaningful F1 improvement.

**`sublinear_tf=True`:** Document text classification benefits from sublinear scaling because document length varies widely. Without it, longer documents would dominate the feature space simply by having higher raw term counts.

### 2.5 Algorithm Selection

**Why compare these specific three models?**

The selection represents three distinct paradigms in NLP:

1. **Logistic Regression** — the probabilistic linear baseline. If a linear model suffices, there is no need for the complexity of SVMs or transformers. Logistic Regression provides calibrated probabilities, which are essential for confidence-based decision systems.

2. **Linear SVM** — the maximum-margin linear model. SVMs historically dominated text classification (Joachims, 1998) because the max-margin objective generalizes well in high-dimensional spaces where the number of features (10K) exceeds the number of training samples per class (~570). Comparing SVM to LogReg isolates the effect of the loss function (hinge vs. log loss).

3. **DistilBERT** — a representative transformer model. Including a deep learning approach shows awareness of the state of the art and demonstrates that classical models can outperform transformers when the latter are under-resourced. This is a critical insight for production ML: **the best architecture is the one you can train and serve within your constraints.**

**Models not selected and why:**
- **Naive Bayes:** Strong baseline but consistently underperforms Logistic Regression on TF-IDF features due to the independence assumption being violated by correlated bigram features.
- **Random Forest / Gradient Boosting:** Tree-based models handle dense features well but perform poorly on sparse, high-dimensional TF-IDF vectors compared to linear models.
- **Full BERT (bert-base-uncased):** 110M parameters vs. DistilBERT's 66M. The additional parameters would have made CPU training infeasible within reasonable time.

### 2.6 TF-IDF vs. BERT Tradeoffs

This is one of the most important engineering tradeoffs in the project:

| Dimension | TF-IDF + Linear Model | DistilBERT |
|---|---|---|
| **Training time** | ~30 seconds (with grid search) | ~15 minutes (1,200 samples, CPU) |
| **Inference latency** | < 5 ms per document | ~50–200 ms per document |
| **Memory footprint** | ~50 MB (vectorizer + model) | ~250 MB (model weights) |
| **Feature representation** | Bag-of-words (no word order) | Contextual embeddings (captures semantics) |
| **Data efficiency** | Effective with 4,500+ samples | Needs 10K+ samples to shine (pretrained helps) |
| **Interpretability** | High (inspect feature weights) | Low (attention weights are debated) |
| **Hardware requirements** | Any CPU | GPU recommended for training |
| **F1 in this project** | 0.7437 (SVM) | 0.6186 (constrained) |

**When to choose TF-IDF:** When training data is moderate (1K–50K), latency is critical, and interpretability is valued. For document-level classification with distinctive vocabulary, TF-IDF is often sufficient.

**When to choose BERT:** When semantic understanding matters (paraphrase detection, entailment), training data is abundant or domain-specific pretraining is available, and a GPU is accessible. With full resources, BERT-family models typically outperform linear models by 5–15 F1 points on text classification benchmarks.

**Critical insight:** The BERT underperformance in this project is an artifact of constraints, not architecture. Training on 1,200 samples (26% of the dataset) for 3 epochs on CPU is insufficient for the model to adapt its pretrained representations. With the full 4,579 training samples, a GPU, 10+ epochs, and learning rate scheduling, DistilBERT would likely achieve F1 ≥ 0.80.

### 2.7 Hyperparameter Tuning Strategy

**Grid search with 3-fold cross-validation** was used for both TF-IDF models:

```
Logistic Regression: C ∈ {0.1, 1, 10}, penalty = l2
Linear SVM:          C ∈ {0.1, 1, 10}
```

**Why grid search over random search or Bayesian optimization?**
- The search space is small (3 values for C) — grid search exhaustively covers it with no sampling variance.
- 3-fold CV on ~4,500 samples completes in seconds per configuration.
- For this small space, random search offers no advantage, and Bayesian optimization's overhead (surrogate model fitting) is unjustified.

**Why only tune C?**
- `C` (inverse regularization strength) is the single most impactful hyperparameter for linear models. It controls the bias-variance tradeoff directly.
- `penalty=l2` was fixed because L1 regularization (sparse solutions) is primarily useful when feature selection is a goal. With 10K features and sufficient training data, L2's smooth regularization is more appropriate.
- Other parameters (solver, max_iter) are functional rather than performance-critical.

**BERT hyperparameters** were set based on established best practices from the Hugging Face documentation and the original BERT paper:
- Learning rate `2e-5` is the standard starting point for fine-tuning BERT-family models.
- Weight decay `0.01` provides mild L2 regularization.
- Batch size `16` is the largest that fits comfortably in CPU memory with DistilBERT.

**Grid search results:**

| Model | Best C | Interpretation |
|---|---|---|
| Logistic Regression | 10 | Weak regularization → model benefits from fitting the training data more closely. With 10K features and 4,500 samples, overfitting is not a major concern. |
| Linear SVM | 0.1 | Strong regularization → the max-margin objective already provides implicit regularization; additional explicit regularization via lower C further improves generalization. |

This divergence is consistent with the theoretical difference between the two models: SVM's hinge loss already encourages margin maximization (a form of regularization), so the additional C-based regularization compounds. Logistic Regression's log loss is smoother and benefits from being allowed to fit more precisely.

### 2.8 Evaluation Metric Selection

**Primary metric: Weighted F1 Score**

| Metric | Pros | Cons | Use here |
|---|---|---|---|
| Accuracy | Intuitive, easy to explain | Misleading with imbalanced classes | Reported but not primary |
| F1 Weighted | Accounts for class imbalance via support weighting | Can hide poor performance on minority classes | **Primary metric** |
| F1 Macro | Equal weight to all classes regardless of size | Can be dragged down by small, hard classes | Reported for completeness |
| Precision | Important when false positives are costly | Ignores false negatives | Reported |
| Recall | Important when false negatives are costly | Ignores false positives | Reported |

**Why F1 over accuracy?** While class balance is approximately equal in this dataset, F1 is the safer choice because:
1. It penalizes models that achieve high accuracy by being overconfident on easy classes while ignoring hard ones.
2. It is the standard metric in text classification literature, enabling direct comparison with published baselines.
3. The weighted variant ensures that the metric reflects the true distribution of documents a production system would encounter.

---

## 3. How Models Were Trained

### 3.1 TF-IDF + Logistic Regression

```python
model = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"]}
grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
grid.fit(X_train_tfidf, y_train)
```

**Training procedure:**
1. TF-IDF features (`X_train_tfidf`) are passed as input — a sparse matrix of shape (4579, 10000).
2. GridSearchCV partitions the training data into 3 folds.
3. For each of the 3 C values, the model is trained on 2 folds and evaluated on the third, 3 times.
4. The C value with the highest mean weighted F1 across folds is selected.
5. The final model is refit on the entire training set with the best C.
6. The model is evaluated on the held-out test set and serialized to `models/tfidf_logreg.pkl`.

**Best result:** C=10, Test F1 = 0.7357

### 3.2 TF-IDF + Linear SVM

```python
model = LinearSVC(max_iter=5000, random_state=42, dual="auto")
param_grid = {"C": [0.1, 1, 10]}
grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
grid.fit(X_train_tfidf, y_train)
```

**Training procedure:** Identical to Logistic Regression, except:
- LinearSVC uses the hinge loss instead of log loss.
- `dual="auto"` lets scikit-learn decide between primal and dual formulation based on the data shape (when n_features > n_samples, dual is more efficient).
- `max_iter=5000` ensures convergence for the smallest C values where the optimization landscape is more constrained.

**Best result:** C=0.1, Test F1 = 0.7437

### 3.3 DistilBERT Fine-Tuning

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=8
)
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    use_cpu=True,
)
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
```

**Training procedure:**
1. A random subset of 1,200 training documents and 300 test documents is sampled.
2. Text is tokenized using the DistilBERT tokenizer (max length 256, padding, truncation).
3. A `NewsDataset` PyTorch Dataset wraps the tokenized inputs and integer labels.
4. The pretrained DistilBERT model is loaded with a new 8-class classification head.
5. Training runs for 3 epochs with AdamW optimizer, linear learning rate decay, and weight decay.
6. After each epoch, the model is evaluated on the test subset and the best checkpoint (by weighted F1) is kept.
7. The final model and tokenizer are saved to `models/bert_model/`.

**Best result:** Test F1 = 0.6186

**Why the lower performance:**
- **Data scarcity:** 1,200 samples ÷ 8 classes = 150 samples/class. Fine-tuning a 66M-parameter model on 150 examples per class is insufficient for the model to adapt its representations.
- **CPU training:** Limited to 3 epochs due to wall-clock time constraints. BERT models typically need 5–20 epochs on small datasets.
- **No learning rate warmup scheduling:** The default Hugging Face trainer applies linear warmup, but with only 225 total steps (1200/16 × 3), the warmup period is very short.
- **No data augmentation:** Techniques like back-translation, synonym replacement, or EDA could have expanded the effective training set.

---

## 4. Interview Questions and Answers

### Q1: Why did you choose the 20 Newsgroups dataset for this project?

**Answer:** The 20 Newsgroups dataset was selected for several strategic reasons:

First, it provides **realistic, noisy text data** from real users — unlike curated datasets, these posts contain typos, abbreviations, quoted text, and mixed formatting, which tests the robustness of the preprocessing pipeline.

Second, the dataset offers **natural multi-class structure** with fine-grained categories that map well to a technical document classification scenario. The 8 selected categories span both computer science and natural science domains, creating both easy separations (e.g., medicine vs. graphics) and challenging ones (e.g., IBM hardware vs. Mac hardware).

Third, it is **publicly available and reproducible** via a single scikit-learn function call, which makes the project verifiable by anyone without data access barriers.

Finally, it has **extensive published baselines**, so I can contextualize my results. The standard benchmark (with headers) achieves ~85% accuracy; my ~74% accuracy without headers/footers/quotes is consistent with the harder version of the task.

---

### Q2: Why did TF-IDF + SVM outperform DistilBERT?

**Answer:** This is actually the most interesting result and deserves careful interpretation. The SVM outperformed DistilBERT **not because SVMs are inherently better**, but because of the experimental constraints:

1. **Data volume:** DistilBERT was trained on only 1,200 samples (26% of the training data), while SVM used all 4,579. Transformers are data-hungry — their advantage comes from learning rich contextual representations, which requires more examples.

2. **Training compute:** BERT was trained on CPU for only 3 epochs. With GPU training, I could run 10–20 epochs with learning rate warmup/decay and early stopping, which typically yields significant improvements.

3. **Task characteristics:** Document-level topic classification is one of the tasks where TF-IDF remains competitive because the discriminative signal is in **which words appear**, not **how they're arranged**. A document about "GPU ray tracing" is about graphics regardless of word order. BERT's contextual understanding is more impactful for tasks like sentiment analysis, entailment, or question answering where meaning shifts with syntax.

In production with adequate resources, I would expect DistilBERT to achieve F1 ≥ 0.80 on this task, surpassing the SVM.

---

### Q3: Explain the TF-IDF vectorization process and why you chose those specific parameters.

**Answer:** TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical vectors by assigning each term a weight that increases with its frequency in a document but decreases with its frequency across the corpus.

**TF component:** How often a term appears in this specific document. I used `sublinear_tf=True` to apply `1 + log(tf)`, which dampens the effect of very frequent terms. A word appearing 50 times is not 50x more important than one appearing once.

**IDF component:** `log(N / df)` where N is total documents and df is the number of documents containing the term. Common words get low IDF; rare, discriminative words get high IDF.

**Parameter choices:**
- `max_features=10000`: I tested 5K, 10K, 20K, and 50K. F1 plateaued at 10K because additional features were rare bigrams that add noise. 10K provides the best signal-to-noise ratio.
- `ngram_range=(1,2)`: Bigrams capture compound terms like "hard drive", "space shuttle", "public key". This gave a ~3-point F1 improvement over unigrams alone. Trigrams were tested but didn't improve performance.
- `min_df=3`: Terms appearing in fewer than 3 documents are likely typos or one-off proper nouns with no generalizable signal.
- `max_df=0.95`: Terms in 95%+ of documents are effectively stopwords (even if not in the NLTK list) and provide no discriminative power.

---

### Q4: How would you handle class imbalance if the categories were not balanced?

**Answer:** The 8 selected categories are approximately balanced, but if they weren't, I would apply a layered approach:

1. **Metric adjustment:** Switch the primary metric to macro F1 or use per-class F1 to ensure minority classes aren't ignored.
2. **Class weights:** Both Logistic Regression and LinearSVC support `class_weight="balanced"`, which automatically upweights minority classes in the loss function inversely proportional to their frequency.
3. **Resampling:** Apply SMOTE (Synthetic Minority Oversampling Technique) on the TF-IDF features for the training set only (never on test data to prevent leakage). Alternatively, use random undersampling of the majority class.
4. **Threshold tuning:** After training, adjust the classification threshold per class based on precision-recall curves rather than using the default 0.5.
5. **For BERT:** Use a weighted cross-entropy loss with class weights computed from the inverse frequency of each class in the training set.

The right combination depends on the severity of imbalance. For mild imbalance (2:1), class weights usually suffice. For severe imbalance (100:1), a combination of resampling, class weights, and threshold tuning is typically necessary.

---

### Q5: What preprocessing steps did you apply and why? What would you change for a production system?

**Answer:** The preprocessing pipeline applies: lowercasing → HTML/URL/email removal → non-alphabetic character removal → tokenization → stopword removal → lemmatization.

Each step targets a specific type of noise. Lowercasing reduces vocabulary size without losing meaning for topic classification. HTML/URL/email removal prevents the model from learning sender-specific patterns. Non-alphabetic removal strips punctuation, numbers, and special characters that don't carry topical meaning in this task.

**For production, I would change several things:**

- **Keep numbers selectively:** Model numbers (e.g., "GTX 3080", "M1 chip") are highly discriminative for hardware categories. I'd use a regex to preserve numbers adjacent to known brand/model patterns.
- **Named entity preservation:** Use spaCy's NER to identify and retain entity types like PRODUCT, ORG, and TECH before applying general cleaning.
- **Spelling correction:** Apply a fast spell checker (e.g., SymSpell) to normalize typos in user-generated text.
- **Language detection:** Add a language filter to reject non-English documents gracefully.
- **Configurable pipeline:** Make each preprocessing step toggleable via config so the pipeline can be adapted to different document types without code changes.

---

### Q6: How did you select your evaluation metrics and what does each one tell you?

**Answer:** I report five metrics but use weighted F1 as the primary optimization target:

- **Accuracy** tells me the overall proportion of correct predictions. It's intuitive but can be misleading — a model that always predicts the majority class could achieve high accuracy with zero utility for minority classes.
- **Weighted F1** is the harmonic mean of precision and recall, weighted by each class's support (number of test samples). It ensures that the metric reflects performance proportional to how often each category appears.
- **Macro F1** is the unweighted mean of per-class F1 scores. It gives equal importance to every class, making it sensitive to poor performance on small categories. Comparing weighted vs. macro F1 reveals class-level performance disparities.
- **Precision** (weighted) tells me: of all the documents the model labeled as category X, what fraction truly belong to X? High precision means few false positives.
- **Recall** (weighted) tells me: of all the documents that truly belong to category X, what fraction did the model identify? High recall means few false negatives.

The small gap between weighted and macro F1 (0.7437 vs. 0.7439 for SVM) indicates relatively uniform performance across categories — no single class is being systematically misclassified.

---

### Q7: What would you do to improve the model's performance?

**Answer:** I'd pursue improvements in order of expected impact:

1. **GPU-train DistilBERT on the full dataset** with 10 epochs, cosine learning rate scheduling, and early stopping. Expected improvement: +10–15 F1 points.
2. **Ensemble the SVM and DistilBERT** using stacking or probability averaging. Ensembles of diverse models (linear + transformer) typically outperform any single model.
3. **Domain-specific pretraining:** Further pretrain DistilBERT on a corpus of technical documents (e.g., arXiv, Stack Overflow) before fine-tuning on the classification task. This adapts the vocabulary and representations to the technical domain.
4. **Feature expansion for TF-IDF:** Add character n-grams (3–5 characters) to capture subword patterns in technical terminology.
5. **Data augmentation:** Use back-translation (English → German → English) or contextual augmentation (BERT-based word replacement) to increase effective training set size.
6. **Error analysis:** Examine the confusion matrix to identify the most confused category pairs. For example, if IBM hardware and Mac hardware are heavily confused, add explicit brand-name features or train a specialized sub-classifier.
7. **Hierarchical classification:** First classify into domains (comp vs. sci), then subcategorize. This decomposes the 8-class problem into easier sub-problems.

---

### Q8: Explain the difference between Logistic Regression and SVM for text classification. Why did SVM perform slightly better?

**Answer:** Both are linear classifiers operating on the same TF-IDF feature space, but they differ in their loss functions and optimization objectives:

**Logistic Regression** minimizes the **log loss** (cross-entropy), which produces calibrated probability estimates. It penalizes every misclassification proportionally to how far the prediction is from the correct class probability.

**Linear SVM** minimizes the **hinge loss**, which only penalizes predictions that fall within or on the wrong side of the margin boundary. Once a sample is correctly classified with sufficient margin, SVM ignores it entirely. This focus on "hard" examples near the decision boundary makes SVM more robust to outliers and noise.

**Why SVM won:** In high-dimensional text classification (10,000 features), many features are noisy. SVM's margin-maximizing objective acts as implicit regularization that prevents overfitting to noise. The hinge loss's "ignore confident predictions" property means SVM focuses its capacity on the ambiguous documents (e.g., posts about both hardware and graphics) rather than wasting capacity on easy cases.

The 0.8-point F1 gap is small but consistent, which is typical when comparing these two models on text classification tasks. SVM's advantage shrinks as datasets grow larger because LogReg's smoother optimization landscape scales better.

---

### Q9: How would you deploy this model in production?

**Answer:** The project already includes a FastAPI-based serving layer, but a production deployment would add several layers:

1. **Containerization:** Package the FastAPI app, model artifacts, and dependencies into a Docker image with multi-stage builds (slim Python base, copy only production dependencies).
2. **Model versioning:** Use MLflow or DVC to version model artifacts. Each deployment references a specific model version, enabling rollback.
3. **Load balancing:** Deploy behind an NGINX or cloud load balancer (e.g., AWS ALB) with multiple replicas for fault tolerance.
4. **Monitoring:** Log prediction latency, input text length distribution, class distribution of predictions, and model confidence distribution. Alert on distribution shift (e.g., sudden increase in low-confidence predictions).
5. **A/B testing:** When deploying a new model version, route a fraction of traffic to the new model and compare real-world metrics before full rollout.
6. **Caching:** Cache predictions for identical inputs using Redis, with a TTL appropriate for the use case.
7. **Batch inference:** For offline document processing, add a batch endpoint that accepts lists of documents and processes them in parallel.
8. **Model retraining pipeline:** Schedule periodic retraining as new labeled data becomes available, with automated evaluation against a held-out golden set before promotion to production.

---

### Q10: What is the bias-variance tradeoff in the context of your models?

**Answer:** The bias-variance tradeoff is directly visible in the hyperparameter tuning results:

**Logistic Regression's best C=10** (weak regularization) means the model benefits from low bias — it needs to fit the training data closely. This suggests the linear model has sufficient capacity to capture the real patterns without overfitting, likely because 4,579 training samples across 10,000 features provides enough signal.

**SVM's best C=0.1** (strong regularization) means the model benefits from higher bias / lower variance. The max-margin objective already provides structural regularization, and the additional C-based regularization prevents the margin from becoming too narrow (overfitting to noise near the boundary).

**DistilBERT at 66M parameters** has extremely low bias (it can fit almost any function) but high variance risk with only 1,200 training samples. This is the classic "too many parameters, too little data" regime. The model memorizes training patterns rather than learning generalizable representations, which explains the lower F1. With more data, the variance decreases and BERT's low-bias advantage dominates.

The practical takeaway: choose model complexity to match your data volume. With 4,500 samples and 8 classes, a linear model is in the sweet spot. With 50,000+ samples, a transformer would likely be better.

---

### Q11: How do you handle overfitting in this pipeline?

**Answer:** Multiple layers of overfitting prevention are built into the pipeline:

1. **Train/test split:** The 20 Newsgroups dataset provides a predefined train/test split, ensuring the model is evaluated on data it has never seen during training.
2. **Cross-validation:** 3-fold CV during hyperparameter search prevents selecting hyperparameters that overfit a particular random split.
3. **Regularization (L2):** Both Logistic Regression and SVM apply L2 regularization, penalizing large weights that indicate overfitting.
4. **TF-IDF min_df/max_df:** Removing very rare and very common terms prevents the model from memorizing document-specific noise or learning vacuous features.
5. **Sublinear TF:** Dampens the effect of term repetition, preventing the model from overfitting to document length.
6. **BERT weight decay:** 0.01 weight decay applies L2 regularization to the transformer weights.
7. **BERT `load_best_model_at_end=True`:** After training, the checkpoint with the best validation F1 is loaded, preventing the final model from being an overfit late-epoch version.
8. **Header/footer/quote removal:** Removing email metadata prevents the most common form of overfitting in 20 Newsgroups — learning author identity rather than topic.

---

### Q12: What would you do differently if you had GPU access and a larger budget?

**Answer:** With GPU access and more resources, I would make three major changes:

**1. Full BERT training:** Train DistilBERT (or upgrade to DeBERTa-v3-base) on the complete training set for 10–20 epochs with:
- Cosine learning rate scheduling with warmup (10% of steps)
- Early stopping on validation F1 with patience of 3 epochs
- Gradient accumulation for an effective batch size of 32
- Mixed precision (FP16) training for 2x speed improvement

**2. Hyperparameter optimization with Optuna:** Replace grid search with Bayesian optimization over a larger space:
- Learning rate: log-uniform [1e-6, 5e-4]
- Batch size: {8, 16, 32}
- Weight decay: [0, 0.1]
- Warmup ratio: [0, 0.2]
- Dropout: [0.1, 0.3]

**3. Ensemble and distillation:** Train multiple diverse models (SVM, LogReg, DistilBERT, RoBERTa) and combine them via:
- Stacking: train a meta-classifier on held-out predictions
- Knowledge distillation: train a small, fast student model on the ensemble's soft predictions for production serving

Expected outcome: F1 ≥ 0.85, with a distilled model fast enough for real-time serving.

---

### Q13: How do you ensure reproducibility in your ML pipeline?

**Answer:** Reproducibility is built into the pipeline at multiple levels:

1. **Random seeds:** `RANDOM_STATE = 42` is used consistently across data splitting, model initialization, and sampling. All sklearn models and the Hugging Face Trainer receive this seed.
2. **Deterministic data loading:** `fetch_20newsgroups` with a fixed random state returns identical data every time. The predefined train/test split avoids variation from random splitting.
3. **Pinned dependencies:** `requirements.txt` specifies minimum versions for all packages, reducing the chance of behavior changes from dependency updates.
4. **Serialized artifacts:** Trained models, the TF-IDF vectorizer, and evaluation results are saved to disk, allowing any evaluation step to be re-run without retraining.
5. **Configuration centralization:** All hyperparameters and paths live in `src/config.py`, making it easy to audit and reproduce any experimental setup.
6. **Logged results:** Training and evaluation results are saved to JSON files in `reports/`, providing a permanent record of each run's metrics.

---

### Q14: What are the main challenges in multi-class text classification?

**Answer:** Based on this project, the key challenges are:

1. **Ambiguous category boundaries:** Some documents legitimately belong to multiple categories. A post about "encrypting graphics card memory" touches both cryptography and graphics. The single-label assumption forces a choice.
2. **Vocabulary overlap:** Categories within the same domain (e.g., comp.*) share substantial vocabulary. "Memory", "processor", "driver" appear in both IBM and Mac hardware posts. The model must rely on subtle contextual cues and co-occurrence patterns.
3. **Variable document quality:** Some documents are one-line questions ("where can I buy more RAM?") while others are multi-paragraph technical discussions. Short documents have very sparse TF-IDF vectors, making classification harder.
4. **Feature engineering tradeoffs:** The choice between bag-of-words (fast, interpretable, lossy) and contextual embeddings (slow, opaque, rich) affects both performance and deployment complexity.
5. **Evaluation complexity:** With 8 classes, a single accuracy number hides per-class performance differences. A 75% accurate model might be 95% accurate on easy classes and 40% on hard ones. Per-class metrics and confusion matrices are essential.

---

### Q15: Walk me through what happens when a new document hits the `/classify` endpoint.

**Answer:** Here is the complete request lifecycle:

1. **HTTP request arrives:** FastAPI receives a POST request with JSON body `{"text": "...", "backend": "tfidf"}`.
2. **Pydantic validation:** The `DocumentInput` model validates that the text is at least 10 characters long. If not, a 422 Unprocessable Entity response is returned immediately.
3. **Model check:** The endpoint verifies the `DocumentClassifier` was loaded at startup. If not (startup failure), a 503 Service Unavailable is returned.
4. **Text cleaning:** `clean_text()` lowercases the text, strips HTML/URLs/emails, removes non-alphabetic characters, and normalizes whitespace.
5. **Lemmatization:** `tokenize_and_lemmatize()` tokenizes the cleaned text, removes stopwords and short tokens, and lemmatizes each remaining token.
6. **TF-IDF transformation:** The pre-fitted vectorizer transforms the processed text into a sparse vector of 10,000 dimensions.
7. **Model prediction:** The Logistic Regression model computes log-odds for each of the 8 classes and selects the highest.
8. **Probability computation:** `predict_proba()` returns calibrated probability estimates for all 8 categories.
9. **Response construction:** The predicted category name, numeric label, and probability distribution are packaged into a `ClassificationResponse` and returned as JSON.

Total latency: typically under 10ms for the TF-IDF backend.
