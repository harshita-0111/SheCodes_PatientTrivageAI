"""
ml/train_model.py
====================

PatientTriage.ai — Ensemble Training
------------------------------------------
Trains a STACKED ensemble (XGBoost + LightGBM + CatBoost +
HistGradientBoosting base learners, logistic-regression meta-learner) to
predict emergency acuity from `triage_stays`.

Why P1-P3, not the full P1-P5 label space
------------------------------------------------
The labeled cohort has 207 stays: P1=18, P2=97, P3=90, P4=2, P5=0. A
model cannot learn a class boundary from 2 examples, and it certainly
cannot learn one from zero. Training a 5-class model on this data would
produce a P4/P5 boundary that LOOKS like a learned clinical judgement
and is actually noise wearing a label. This module trains genuine
multiclass on P1/P2/P3 (n=205, real signal in every class) and excludes
the 2 P4 rows from training entirely — they are reported, not hidden,
in the training metrics.

P4/P5 assignment for a patient the ensemble does not flag as P1-P3-risk
is handled by a documented rule-based floor in `ml/predict.py`, not by
an ML boundary this data cannot support. This is the same pattern
`ml/rule_engine.py` uses elsewhere in this project: rules set floors:
ML cannot invent a boundary claim it hasn't earned.

Run:
    cd backend
    python -m ml.train_model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.core.config import get_settings
from app.core.logging_config import configure_logging, get_logger
from ml.model_utils import FEATURE_NAMES, ModelArtifact, load_labeled_stays, save_artifact

configure_logging()
logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_SPLITS = 5
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 60  # Increased trials for a more thorough hyperparameter search


def _tune_xgboost(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> dict:
    from xgboost import XGBClassifier

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 80, 250),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        )
        model = XGBClassifier(**params, eval_metric="mlogloss", random_state=RANDOM_STATE)
        preds = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        return log_loss(y, preds, labels=sorted(set(y)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("xgboost best log-loss=%.4f params=%s", study.best_value, study.best_params)
    return study.best_params


def _tune_lightgbm(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> dict:
    from lightgbm import LGBMClassifier

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 80, 250),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 31),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 20),
        )
        model = LGBMClassifier(**params, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
        preds = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        return log_loss(y, preds, labels=sorted(set(y)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("lightgbm best log-loss=%.4f params=%s", study.best_value, study.best_params)
    return study.best_params


def _tune_catboost(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> dict:
    from catboost import CatBoostClassifier

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            iterations=trial.suggest_int("iterations", 80, 250),
            depth=trial.suggest_int("depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 8.0, log=True),
        )
        model = CatBoostClassifier(**params, auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=0)
        preds = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        return log_loss(y, preds, labels=sorted(set(y)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("catboost best log-loss=%.4f params=%s", study.best_value, study.best_params)
    return study.best_params


def _build_base_models(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> list[tuple[str, object]]:
    """Optuna-tune each base learner, then return fitted-ready (unfit) estimators for the stack."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    xgb_params = _tune_xgboost(X, y, cv)
    lgbm_params = _tune_lightgbm(X, y, cv)
    cat_params = _tune_catboost(X, y, cv)

    return [
        ("xgboost", XGBClassifier(**xgb_params, eval_metric="mlogloss", random_state=RANDOM_STATE)),
        ("lightgbm", LGBMClassifier(**lgbm_params, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)),
        ("catboost", CatBoostClassifier(**cat_params, auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=0)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(
            max_iter=150, max_depth=3, learning_rate=0.08, class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ]


def _evaluate_oof(stack: StackingClassifier, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, classes: list[int]) -> dict:
    """
    Out-of-fold predictions from the full stack — honest evaluation on
    205 rows needs CV, not one split. `y` here is already remapped to
    0..n-1 (see `train()`); metrics are translated back to true priority
    labels (`classes`) before being returned, so the saved report reads
    in clinical terms (P1/P2/P3), not internal model indices.
    """
    oof_proba = cross_val_predict(stack, X, y, cv=cv, method="predict_proba")
    oof_pred_idx = oof_proba.argmax(axis=1)

    y_true = np.array(classes)[y.to_numpy()]
    y_pred = np.array(classes)[oof_pred_idx]

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro")), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "log_loss": round(float(log_loss(y, oof_proba, labels=list(range(len(classes))))), 4),
        "per_class_recall": {
            str(c): round(float(recall_score(y_true == c, y_pred == c, zero_division=0)), 4) for c in classes
        },
        "confusion_matrix": {
            f"true_{t}_pred_{p}": int(((y_true == t) & (y_pred == p)).sum())
            for t in classes for p in classes
        },
    }


def train(verbose: bool = True) -> dict:
    settings = get_settings()
    X, y_raw, classes = load_labeled_stays()
    if verbose:
        logger.info(
            "Training frame: %d labeled stays across classes %s (prevalence %s)",
            len(X), classes, y_raw.value_counts().sort_index().to_dict(),
        )

    if len(classes) < 2:
        raise RuntimeError(f"Only {len(classes)} trainable class(es) found — need at least 2 to train a classifier.")

    # XGBoost requires 0..n-1 integer class labels; our priority classes
    # are e.g. [1, 2, 3]. Fit everything on the remapped labels and keep
    # `classes` (the true priority integers, sorted ascending) as the
    # public contract — sklearn's predict_proba column order follows
    # sorted(model.classes_), so column i always corresponds to
    # classes[i] as long as every model is fit on this same remapping.
    label_to_index = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(label_to_index)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    if verbose:
        logger.info("Tuning base learners with Optuna (%d trials each)...", N_OPTUNA_TRIALS)
    base_specs = _build_base_models(X, y, cv)

    stack = StackingClassifier(
        estimators=base_specs,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    if verbose:
        logger.info("Evaluating stacked ensemble via %d-fold out-of-fold CV...", N_SPLITS)
    oof_metrics = _evaluate_oof(stack, X, y, cv, classes)
    if verbose:
        logger.info(
            "OOF: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f log_loss=%.4f",
            oof_metrics["accuracy"], oof_metrics["macro_f1"], oof_metrics["weighted_f1"], oof_metrics["log_loss"],
        )

    # Fit base models individually (kept for ensemble-agreement scoring in
    # uncertainty.py) and the full stack (the actual predictor) on all data.
    base_fitted = {}
    for name, model in base_specs:
        from sklearn.base import clone
        fitted = clone(model)
        fitted.fit(X, y)
        base_fitted[name] = fitted

    stack.fit(X, y)

    # Calibrate the fitted stack's probabilities via 5-fold CV on the same data.
    calibrated = CalibratedClassifierCV(stack, method="sigmoid", cv=cv)
    calibrated.fit(X, y)

    version = f"stacking-ensemble-{datetime.now(timezone.utc):%Y%m%d-%H%M}"
    metrics = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_labeled": int(len(X)),
        "classes_trained": classes,
        "classes_excluded_sparse": "see logs — fewer than 10 examples",
        "evaluation_method": f"{N_SPLITS}-fold stratified out-of-fold CV, NOT held-out test (n={len(X)} is too small to trust a single split)",
        "oof_metrics": oof_metrics,
        "known_limitations": [
            "age is absent from this MIMIC-IV-ED demo extract (no anchor_age in edstays.csv.gz) — "
            "every stay is scored with age=None, so age-banded rule logic falls back to the adult band "
            "for every training row. Age-specific ML patterns cannot currently be learned from this data.",
            "P4 (n=2) and P5 (n=0) are excluded from ML training as statistically unsupportable. "
            "P4/P5 assignment for low-risk patients is a documented rule-based floor in ml/predict.py, "
            "not a learned boundary.",
            f"n={len(X)} labeled stays is small for a clinical ensemble; OOF metrics carry real uncertainty.",
        ],
    }

    artifact = ModelArtifact(
        base_models=base_fitted,
        meta_learner=calibrated,  # the calibrated stack IS the servable predictor
        calibrators={},  # calibration is applied to the whole stack above, not per-base-model
        classes=classes,
        feature_names=FEATURE_NAMES,
        version=version,
        metrics=metrics,
    )
    save_artifact(artifact)

    if verbose:
        logger.info("Training complete. Artifact + metrics saved under %s", settings.REPORTS_DIR)
    return metrics


if __name__ == "__main__":
    result = train()
    print("\n=== PatientTriage.ai — Ensemble Training Summary ===")
    print(f"Version        : {result['version']}")
    print(f"Classes trained: {result['classes_trained']}  (n={result['n_labeled']})")
    m = result["oof_metrics"]
    print(f"Accuracy       : {m['accuracy']}")
    print(f"Macro F1       : {m['macro_f1']}")
    print(f"Weighted F1    : {m['weighted_f1']}")
    print(f"Macro recall   : {m['macro_recall']}")
    print(f"Log loss       : {m['log_loss']}")
    print("Per-class recall:", json.dumps(m["per_class_recall"]))
    print(f"\nFull report: backend/reports/{__import__('ml.model_utils', fromlist=['METRICS_NAME']).METRICS_NAME}")
