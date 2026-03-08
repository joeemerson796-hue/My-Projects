"""Feature engineering: TF-IDF and BERT tokenization for document classification."""

import logging

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_PROCESSED, MAX_TFIDF_FEATURES

logger = logging.getLogger(__name__)


def create_tfidf_features(train_texts, test_texts, max_features: int = MAX_TFIDF_FEATURES):
    """Fit TF-IDF vectorizer on training text and transform both splits."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    joblib.dump(vectorizer, DATA_PROCESSED / "tfidf_vectorizer.pkl")
    logger.info("TF-IDF: %d features, train shape %s", X_train.shape[1], X_train.shape)
    return X_train, X_test, vectorizer


def run_feature_engineering(artifacts: dict) -> dict:
    """Build TF-IDF features from preprocessed documents."""
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(
        artifacts["train_df"]["processed_text"],
        artifacts["test_df"]["processed_text"],
    )
    artifacts["X_train_tfidf"] = X_train_tfidf
    artifacts["X_test_tfidf"] = X_test_tfidf
    artifacts["tfidf_vectorizer"] = vectorizer
    artifacts["y_train"] = artifacts["train_df"]["label"].values
    artifacts["y_test"] = artifacts["test_df"]["label"].values
    return artifacts
