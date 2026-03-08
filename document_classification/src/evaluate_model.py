"""Model evaluation and comparison for document classification."""

import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SELECTED_CATEGORIES

logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")

SHORT_NAMES = [c.split(".")[-1] for c in SELECTED_CATEGORIES]


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, labels=None) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    display_labels = labels or SHORT_NAMES[: cm.shape[0]]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=display_labels, yticklabels=display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_model_comparison(results: dict) -> str:
    names = list(results.keys())
    f1s = [results[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(names, f1s, color=colors[: len(names)])
    ax.set_ylabel("Test F1 (weighted)")
    ax.set_title("Model Comparison – Document Classification")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def evaluate_all_models(artifacts: dict) -> dict:
    """Evaluate every trained model on the test set."""
    results = artifacts["trained_models"]
    y_test = artifacts["y_test"]
    eval_results = {}

    for name in ["tfidf_logreg", "tfidf_svm"]:
        info = results[name]
        model = info["model"]
        y_pred = model.predict(artifacts["X_test_tfidf"])
        metrics = compute_metrics(y_test, y_pred)
        eval_results[name] = metrics
        plot_confusion_matrix(y_test, y_pred, name)
        print(f"\n{'='*60}")
        print(f"Classification Report – {name}")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=SHORT_NAMES))

    bert_info = results["bert_distilbert"]
    trainer = bert_info.get("trainer")
    if trainer is not None:
        eval_out = trainer.evaluate()
        bert_f1 = eval_out.get("eval_f1_weighted", 0.0)
        eval_results["bert_distilbert"] = {"f1_weighted": bert_f1, "accuracy": eval_out.get("eval_accuracy", 0.0)}
        logger.info("BERT eval: %s", eval_out)
    else:
        eval_results["bert_distilbert"] = {"f1_weighted": bert_info["test_f1"]}

    f1_map = {n: eval_results[n]["f1_weighted"] for n in eval_results}
    plot_model_comparison(f1_map)

    best_name = artifacts["best_model_name"]
    meta = {
        "best_model": best_name,
        "all_results": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in eval_results.items()},
    }

    with open(REPORTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump({"best_model": best_name, "num_classes": artifacts["num_classes"],
                    "categories": SELECTED_CATEGORIES}, f, indent=2)

    artifacts["eval_results"] = eval_results
    logger.info("Evaluation complete – best: %s", best_name)
    return artifacts
