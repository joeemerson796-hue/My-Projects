"""Model training with hyperparameter tuning for supply chain risk prediction."""

import json
import logging
import warnings
from typing import Any

import joblib
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

from src.config import CV_FOLDS, MODELS_DIR, N_OPTUNA_TRIALS, RANDOM_STATE, REPORTS_DIR

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def _objective_lr(trial, X, y):
    C = trial.suggest_float("C", 1e-3, 100, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    model = LogisticRegression(
        C=C, l1_ratio=l1_ratio, solver="saga", max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return cross_val_score(model, X, y, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=1).mean()


def _objective_rf(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
    return cross_val_score(model, X, y, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=1).mean()


def _objective_xgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = xgb.XGBClassifier(
        **params, objective="multi:softprob", eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    return cross_val_score(model, X, y, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=1).mean()


def _objective_lgbm(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = lgb.LGBMClassifier(
        **params, objective="multiclass", random_state=RANDOM_STATE,
        n_jobs=1, verbose=-1,
    )
    return cross_val_score(model, X, y, cv=CV_FOLDS, scoring="f1_weighted", n_jobs=1).mean()


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "objective": _objective_lr,
        "builder": lambda params: LogisticRegression(
            **params, solver="saga", max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    },
    "random_forest": {
        "objective": _objective_rf,
        "builder": lambda params: RandomForestClassifier(
            **params, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    },
    "xgboost": {
        "objective": _objective_xgb,
        "builder": lambda params: xgb.XGBClassifier(
            **params, objective="multi:softprob", eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
    },
    "lightgbm": {
        "objective": _objective_lgbm,
        "builder": lambda params: lgb.LGBMClassifier(
            **params, objective="multiclass", random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        ),
    },
}


def tune_and_train(name: str, X_train, y_train, X_val, y_val, n_trials: int = N_OPTUNA_TRIALS) -> dict:
    """Run Optuna hyperparameter search, then train the best model."""
    logger.info("Tuning %s with %d trials …", name, n_trials)
    entry = MODEL_REGISTRY[name]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: entry["objective"](trial, X_train, y_train), n_trials=n_trials)

    best_params = study.best_params
    logger.info("%s best CV F1: %.4f | params: %s", name, study.best_value, best_params)

    model = entry["builder"](best_params)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="weighted")
    logger.info("%s validation F1: %.4f", name, val_f1)

    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)

    return {
        "name": name,
        "model": model,
        "best_params": best_params,
        "best_cv_f1": study.best_value,
        "val_f1": val_f1,
        "model_path": str(model_path),
    }


def train_all_models(artifacts: dict) -> dict:
    """Train all candidate models and select the best one."""
    results = {}
    for name in MODEL_REGISTRY:
        result = tune_and_train(
            name,
            artifacts["X_train"], artifacts["y_train"],
            artifacts["X_val"], artifacts["y_val"],
        )
        results[name] = result

    best_name = max(results, key=lambda k: results[k]["val_f1"])
    logger.info("Best model: %s (val F1=%.4f)", best_name, results[best_name]["val_f1"])

    summary = {k: {"best_params": v["best_params"], "best_cv_f1": v["best_cv_f1"],
                    "val_f1": v["val_f1"]} for k, v in results.items()}
    with open(REPORTS_DIR / "training_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    artifacts["trained_models"] = results
    artifacts["best_model_name"] = best_name
    artifacts["best_model"] = results[best_name]["model"]
    return artifacts
