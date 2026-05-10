"""
Trainer — fits an XGBoost classifier and saves all artefacts.

Why XGBoost?
────────────
  • Tabular data advantage: on datasets of this size (~600 rows, ~40 features),
    gradient-boosted trees consistently outperform neural networks because they
    do not require large amounts of data to avoid overfitting.
  • Class imbalance: the `scale_pos_weight` parameter handles the low positive
    rate (~6% biopsy positive) natively without oversampling.
  • Explainability: Tree SHAP produces fast, exact feature attribution — a
    requirement for any tool used in a clinical decision-support context.
  • Regularisation: L1 (reg_alpha) and L2 (reg_lambda) penalties reduce
    overfitting on small datasets.
  • Auditability: the model can be inspected, versioned, and re-loaded without
    a deep learning framework.

Saved artefacts (all in models/):
  xgboost_model.joblib     — serialised model
  feature_names.json       — ordered feature column list (required at inference time)
  metrics.json             — CV AUC, train AUC, F1, accuracy, positive rate
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    TARGET_COLUMN, MODEL_PATH, FEATURE_NAMES_PATH,
    MODEL_METRICS_PATH, MODELS_DIR,
)
from src.features.feature_engineer import get_feature_columns

logger = logging.getLogger(__name__)


def _scale_pos_weight(y: pd.Series) -> float:
    """Ratio of negative to positive samples for XGBoost class-imbalance correction."""
    neg, pos = (y == 0).sum(), (y == 1).sum()
    if pos == 0:
        return 1.0
    ratio = neg / pos
    logger.info("Class ratio (neg/pos): %.1f → scale_pos_weight=%.1f", ratio, ratio)
    return float(ratio)


def train(df: pd.DataFrame) -> dict:
    """
    Train XGBoost on the given feature-engineered DataFrame.

    Args:
        df: DataFrame including all feature columns and the target column.

    Returns:
        Dictionary of training and cross-validation metrics.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    logger.info(
        "Training on %d samples, %d features. Positive rate: %.1f%%",
        len(X), len(feature_cols), 100 * y.mean(),
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=_scale_pos_weight(y),
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation provides an unbiased performance estimate before final fit
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    logger.info("5-fold CV AUC: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # Final fit on full dataset for deployment
    model.fit(X, y, verbose=False)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "trained_at":     datetime.utcnow().isoformat(),
        "n_samples":      int(len(X)),
        "n_features":     int(len(feature_cols)),
        "positive_rate":  round(float(y.mean()), 4),
        "cv_auc_mean":    round(float(cv_scores.mean()), 4),
        "cv_auc_std":     round(float(cv_scores.std()),  4),
        "train_auc":      round(float(roc_auc_score(y, y_prob)), 4),
        "train_f1":       round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "train_accuracy": round(float(accuracy_score(y, y_pred)), 4),
    }

    logger.info(
        "CV AUC: %.3f ± %.3f | Train AUC: %.3f | Train F1: %.3f",
        metrics["cv_auc_mean"], metrics["cv_auc_std"],
        metrics["train_auc"],   metrics["train_f1"],
    )

    # ── Save artefacts ─────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved → %s", MODEL_PATH)

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(MODEL_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_model():
    """Load and return the saved XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. "
            "Run `python scripts/initial_train.py` first."
        )
    return joblib.load(MODEL_PATH)


def load_feature_names() -> list:
    """Load the ordered list of feature column names expected by the model."""
    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Feature names not found at {FEATURE_NAMES_PATH}."
        )
    with open(FEATURE_NAMES_PATH) as f:
        return json.load(f)
