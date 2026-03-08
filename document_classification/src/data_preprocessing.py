"""Data loading and NLP preprocessing for document classification."""

import logging
import re
from typing import Tuple

import nltk
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW,
    RANDOM_STATE,
    SELECTED_CATEGORIES,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def load_20newsgroups() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 20 Newsgroups dataset with selected technical categories."""
    logger.info("Loading 20 Newsgroups – categories: %s", SELECTED_CATEGORIES)
    train_data = fetch_20newsgroups(
        subset="train", categories=SELECTED_CATEGORIES,
        remove=("headers", "footers", "quotes"), random_state=RANDOM_STATE,
    )
    test_data = fetch_20newsgroups(
        subset="test", categories=SELECTED_CATEGORIES,
        remove=("headers", "footers", "quotes"), random_state=RANDOM_STATE,
    )

    df_train = pd.DataFrame({"text": train_data.data, "label": train_data.target})
    df_test = pd.DataFrame({"text": test_data.data, "label": test_data.target})

    target_names = train_data.target_names
    label_map = {i: name for i, name in enumerate(target_names)}
    df_train["category"] = df_train["label"].map(label_map)
    df_test["category"] = df_test["label"].map(label_map)

    df_train.to_csv(DATA_RAW / "train.csv", index=False)
    df_test.to_csv(DATA_RAW / "test.csv", index=False)

    logger.info("Train: %d documents | Test: %d documents", len(df_train), len(df_test))
    logger.info("Categories: %s", target_names)
    return df_train, df_test


def clean_text(text: str) -> str:
    """Clean a document string for NLP processing."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize, remove stopwords, and lemmatize."""
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Full NLP preprocessing pipeline for a DataFrame."""
    df = df.copy()
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df["processed_text"] = df["clean_text"].apply(tokenize_and_lemmatize)
    df = df[df["processed_text"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Preprocessed %d documents", len(df))
    return df


def run_preprocessing_pipeline() -> dict:
    """Execute the full preprocessing pipeline."""
    df_train, df_test = load_20newsgroups()
    df_train = preprocess_documents(df_train)
    df_test = preprocess_documents(df_test)

    return {
        "train_df": df_train,
        "test_df": df_test,
        "num_classes": df_train["label"].nunique(),
    }
