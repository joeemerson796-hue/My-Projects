"""Model evaluation, comparison, and explainability for supply chain risk prediction."""

import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RISK_LABELS

logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }
    if y_proba is not None:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
        except ValueError:
            pass
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=RISK_LABELS.values(), yticklabels=RISK_LABELS.values())
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", path)
    return str(path)


def plot_feature_importance(model, feature_names: list, model_name: str) -> str:
    """Plot feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        logger.warning("Model %s has no feature_importances_ or coef_", model_name)
        return ""

    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), importances[idx][::-1], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top Feature Importances – {model_name}")
    path = FIGURES_DIR / f"feature_importance_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot: %s", path)
    return str(path)


def shap_analysis(model, X_sample: pd.DataFrame, feature_names: list, model_name: str) -> str:
    """Generate SHAP summary plot for model explainability."""
    logger.info("Running SHAP analysis for %s …", model_name)
    sample = X_sample.iloc[:500] if len(X_sample) > 500 else X_sample

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, sample)
        shap_values = explainer.shap_values(sample)
    except Exception as e:
        logger.warning("SHAP failed for %s: %s – falling back to KernelExplainer", model_name, e)
        bg = shap.kmeans(sample, 10)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[2], sample, feature_names=feature_names,
                          show=False, plot_size=(10, 7))
    else:
        shap.summary_plot(shap_values, sample, feature_names=feature_names,
                          show=False, plot_size=(10, 7))
    path = FIGURES_DIR / f"shap_summary_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary: %s", path)
    return str(path)


def plot_model_comparison(results: dict) -> str:
    """Bar chart comparing validation F1 across models."""
    names = list(results.keys())
    f1s = [results[n]["val_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, f1s, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylabel("Validation F1 (weighted)")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model comparison: %s", path)
    return str(path)


def evaluate_all_models(artifacts: dict) -> dict:
    """Evaluate every trained model on the test set."""
    results = artifacts["trained_models"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    feature_names = artifacts["feature_names"]
    eval_results = {}

    for name, info in results.items():
        model = info["model"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_proba)
        eval_results[name] = metrics

        plot_confusion_matrix(y_test, y_pred, name)
        plot_feature_importance(model, feature_names, name)

        logger.info("%s test metrics: %s", name, {k: f"{v:.4f}" for k, v in metrics.items()})
        print(f"\n{'='*60}")
        print(f"Classification Report – {name}")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=list(RISK_LABELS.values())))

    plot_model_comparison({n: {"val_f1": eval_results[n]["f1_weighted"]} for n in eval_results})

    best_name = artifacts["best_model_name"]
    shap_analysis(artifacts["best_model"], X_test, feature_names, best_name)

    with open(REPORTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    best_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(artifacts["best_model"], best_model_path)

    meta = {
        "best_model": best_name,
        "feature_names": feature_names,
        "test_metrics": eval_results[best_name],
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    artifacts["eval_results"] = eval_results
    logger.info("Evaluation complete – best model: %s", best_name)
    return artifacts
