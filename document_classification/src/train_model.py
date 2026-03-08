"""Model training for document classification: TF-IDF models and BERT fine-tuning."""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_TEST_SAMPLES,
    BERT_TRAIN_SAMPLES,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def train_tfidf_logreg(X_train, y_train, X_test, y_test) -> dict:
    """Train Logistic Regression on TF-IDF features with grid search."""
    logger.info("Training TF-IDF + Logistic Regression …")
    param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"]}
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
    grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    model_path = MODELS_DIR / "tfidf_logreg.pkl"
    joblib.dump(best_model, model_path)

    logger.info("TF-IDF+LR best params: %s | Test F1: %.4f", grid.best_params_, test_f1)
    return {
        "name": "tfidf_logreg",
        "model": best_model,
        "best_params": grid.best_params_,
        "test_f1": test_f1,
        "model_path": str(model_path),
    }


def train_tfidf_svm(X_train, y_train, X_test, y_test) -> dict:
    """Train Linear SVM on TF-IDF features with grid search."""
    logger.info("Training TF-IDF + SVM …")
    param_grid = {"C": [0.1, 1, 10]}
    model = LinearSVC(max_iter=5000, random_state=RANDOM_STATE, dual="auto")
    grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    model_path = MODELS_DIR / "tfidf_svm.pkl"
    joblib.dump(best_model, model_path)

    logger.info("TF-IDF+SVM best params: %s | Test F1: %.4f", grid.best_params_, test_f1)
    return {
        "name": "tfidf_svm",
        "model": best_model,
        "best_params": grid.best_params_,
        "test_f1": test_f1,
        "model_path": str(model_path),
    }


class NewsDataset(torch.utils.data.Dataset):
    """PyTorch dataset for tokenized documents."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_bert_classifier(train_df, test_df, num_classes: int) -> dict:
    """Fine-tune DistilBERT for document classification."""
    logger.info("Fine-tuning %s (CPU-friendly subset) …", BERT_MODEL_NAME)

    train_sub = train_df.sample(n=min(BERT_TRAIN_SAMPLES, len(train_df)), random_state=RANDOM_STATE)
    test_sub = test_df.sample(n=min(BERT_TEST_SAMPLES, len(test_df)), random_state=RANDOM_STATE)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_enc = tokenizer(
        list(train_sub["clean_text"]), truncation=True, padding=True, max_length=BERT_MAX_LENGTH,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        list(test_sub["clean_text"]), truncation=True, padding=True, max_length=BERT_MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = NewsDataset(train_enc, train_sub["label"].values)
    test_dataset = NewsDataset(test_enc, test_sub["label"].values)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_classes,
    )

    output_dir = str(MODELS_DIR / "bert_checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        report_to="none",
        seed=RANDOM_STATE,
        use_cpu=True,
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        return {"f1_weighted": f1_score(eval_pred.label_ids, preds, average="weighted")}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    test_f1 = eval_result.get("eval_f1_weighted", 0.0)

    bert_dir = MODELS_DIR / "bert_model"
    model.save_pretrained(bert_dir)
    tokenizer.save_pretrained(bert_dir)

    logger.info("BERT test F1: %.4f", test_f1)
    return {
        "name": "bert_distilbert",
        "model": model,
        "tokenizer": tokenizer,
        "test_f1": test_f1,
        "model_path": str(bert_dir),
        "trainer": trainer,
    }


def train_all_models(artifacts: dict) -> dict:
    """Train all candidate models."""
    results = {}

    lr_result = train_tfidf_logreg(
        artifacts["X_train_tfidf"], artifacts["y_train"],
        artifacts["X_test_tfidf"], artifacts["y_test"],
    )
    results["tfidf_logreg"] = lr_result

    svm_result = train_tfidf_svm(
        artifacts["X_train_tfidf"], artifacts["y_train"],
        artifacts["X_test_tfidf"], artifacts["y_test"],
    )
    results["tfidf_svm"] = svm_result

    bert_result = train_bert_classifier(
        artifacts["train_df"], artifacts["test_df"], artifacts["num_classes"],
    )
    results["bert_distilbert"] = bert_result

    best_name = max(results, key=lambda k: results[k]["test_f1"])
    logger.info("Best model: %s (F1=%.4f)", best_name, results[best_name]["test_f1"])

    summary = {k: {"test_f1": v["test_f1"], "model_path": v["model_path"]}
               for k, v in results.items()}
    if "best_params" in lr_result:
        summary["tfidf_logreg"]["best_params"] = lr_result["best_params"]
    if "best_params" in svm_result:
        summary["tfidf_svm"]["best_params"] = svm_result["best_params"]

    with open(REPORTS_DIR / "training_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    artifacts["trained_models"] = results
    artifacts["best_model_name"] = best_name
    return artifacts
