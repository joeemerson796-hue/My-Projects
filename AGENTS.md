## Cursor Cloud specific instructions

This repository contains two production-quality ML portfolio projects:

### Projects

1. **`supply_chain_risk_prediction/`** — Predicts supplier risk levels (Low/Medium/High) using supply chain features. Models: Logistic Regression, Random Forest, XGBoost, LightGBM. Includes SHAP explainability and FastAPI endpoint.

2. **`document_classification/`** — Classifies technical documents into 8 categories using NLP. Models: TF-IDF + Logistic Regression, TF-IDF + SVM, DistilBERT fine-tuning. Includes FastAPI endpoint.

### Running the pipelines

Each project has a self-contained `run_pipeline.py` that runs the full ML pipeline (data → preprocessing → feature engineering → training → evaluation → inference demo). Run from the project root:

```bash
cd supply_chain_risk_prediction && python3 run_pipeline.py
cd document_classification && python3 run_pipeline.py
```

### Running the APIs

```bash
cd supply_chain_risk_prediction && python3 -m uvicorn app.main:app --port 8001
cd document_classification && python3 -m uvicorn app.main:app --port 8002
```

### Non-obvious caveats

- **scikit-learn 1.8+**: The `multi_class` and `penalty` parameters in `LogisticRegression` are deprecated. The code uses `l1_ratio` instead.
- **transformers 5.x**: Requires `accelerate` package. Uses `use_cpu=True` instead of the old `no_cuda=True`.
- **BERT training on CPU**: DistilBERT fine-tuning uses a small subset (1200 samples, 3 epochs) to remain CPU-feasible (~10 min). With GPU, use the full dataset.
- **LightGBM + Optuna**: Use `n_jobs=1` inside `cross_val_score` when called from Optuna to avoid process deadlocks in constrained environments.
- **uvicorn**: Installed to `~/.local/bin` — ensure `PATH` includes `$HOME/.local/bin`.
- **PyTorch**: CPU-only variant installed via `--index-url https://download.pytorch.org/whl/cpu` to save disk space.
