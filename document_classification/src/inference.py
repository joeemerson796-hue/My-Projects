"""Inference pipeline for document classification."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BERT_MAX_LENGTH, DATA_PROCESSED, MODELS_DIR, SELECTED_CATEGORIES
from src.data_preprocessing import clean_text, tokenize_and_lemmatize

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Production-ready document classifier supporting TF-IDF and BERT backends."""

    def __init__(self, model_dir: Path | str = MODELS_DIR, backend: str = "auto"):
        model_dir = Path(model_dir)

        with open(model_dir / "model_metadata.json") as f:
            meta = json.load(f)
        self.categories = meta["categories"]
        best = meta["best_model"]

        if backend == "auto":
            backend = "bert" if "bert" in best else "tfidf"

        self.backend = backend

        if self.backend == "tfidf":
            self.model = joblib.load(model_dir / "tfidf_logreg.pkl")
            self.vectorizer = joblib.load(DATA_PROCESSED / "tfidf_vectorizer.pkl")
            logger.info("Loaded TF-IDF backend (Logistic Regression)")
        else:
            bert_dir = model_dir / "bert_model"
            self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
            self.model.eval()
            logger.info("Loaded BERT backend (%s)", bert_dir)

    def predict(self, text: str) -> dict:
        """Classify a single document."""
        if self.backend == "tfidf":
            return self._predict_tfidf(text)
        return self._predict_bert(text)

    def _predict_tfidf(self, text: str) -> dict:
        cleaned = clean_text(text)
        processed = tokenize_and_lemmatize(cleaned)
        features = self.vectorizer.transform([processed])
        pred = self.model.predict(features)[0]

        proba = {}
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features)[0]
            proba = {self.categories[i]: round(float(p), 4) for i, p in enumerate(probas)}

        return {
            "category": self.categories[int(pred)],
            "label": int(pred),
            "probabilities": proba,
        }

    def _predict_bert(self, text: str) -> dict:
        cleaned = clean_text(text)
        inputs = self.tokenizer(cleaned, truncation=True, padding=True,
                                max_length=BERT_MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).numpy()
        pred = int(np.argmax(probs))

        return {
            "category": self.categories[pred],
            "label": pred,
            "probabilities": {self.categories[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }


def run_inference_demo():
    """Quick demonstration of the inference pipeline."""
    classifier = DocumentClassifier(backend="tfidf")

    sample_docs = [
        "The new GPU architecture features improved ray tracing cores and CUDA "
        "performance for rendering complex 3D graphics in real time.",
        "Researchers developed a novel encryption protocol based on elliptic curve "
        "cryptography that provides enhanced security for IoT devices.",
        "The patient was diagnosed with acute myocardial infarction and treated with "
        "percutaneous coronary intervention within the first hour.",
    ]

    print("\n" + "=" * 60)
    print("INFERENCE DEMO")
    print("=" * 60)
    for doc in sample_docs:
        result = classifier.predict(doc)
        print(f"\nText: {doc[:80]}…")
        print(f"Category: {result['category']}")
        if result["probabilities"]:
            top3 = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top-3: {top3}")
    return classifier
