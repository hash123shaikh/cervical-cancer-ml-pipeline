"""
Predictor — runs inference using the saved XGBoost model.

Each prediction includes:
  • Binary classification (0 = low risk, 1 = elevated risk)
  • Calibrated probability (P(biopsy positive))
  • Risk tier: low (<30%), moderate (30–60%), high (>60%)
  • Top-5 SHAP feature attributions for clinical explainability

Model and SHAP explainer are loaded lazily on first call and cached in module
scope, so repeated calls (e.g. batch predictions via the API) pay the
deserialisation cost only once.

Clinical note: this output is a risk-stratification aid, not a diagnostic tool.
Final clinical decisions must be made by a qualified healthcare professional.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_model        = None
_explainer    = None
_feature_names = None


def _load_artifacts():
    global _model, _explainer, _feature_names
    if _model is not None:
        return
    from src.training.trainer import load_model, load_feature_names
    _model         = load_model()
    _feature_names = load_feature_names()
    _explainer     = shap.TreeExplainer(_model)
    logger.info("Model and SHAP explainer loaded.")


def predict(df: pd.DataFrame) -> list:
    """
    Run batch prediction on a feature-engineered DataFrame.

    Args:
        df: DataFrame with at minimum the required feature columns.

    Returns:
        List of dicts, one per row:
          {
            "prediction":   int,          # 0 or 1
            "probability":  float,        # P(biopsy positive)
            "risk_level":   str,          # "low" | "moderate" | "high"
            "top_features": list[dict],   # top-5 SHAP attributions
          }
    """
    _load_artifacts()

    # Fill any missing feature columns with 0 (conservative — absence of risk factor)
    missing = set(_feature_names) - set(df.columns)
    if missing:
        logger.warning("Missing %d feature columns; filling with 0: %s", len(missing), missing)
        for col in missing:
            df[col] = 0.0

    X = df[_feature_names].astype(float)

    predictions  = _model.predict(X)
    probabilities = _model.predict_proba(X)[:, 1]
    shap_values  = _explainer.shap_values(X)

    results = []
    for i in range(len(X)):
        prob = float(probabilities[i])
        risk_level = "high" if prob >= 0.6 else "moderate" if prob >= 0.3 else "low"

        # SHAP attributions — sorted by absolute magnitude
        shap_pairs = zip(_feature_names, shap_values[i])
        top_features = sorted(
            [{"feature": k, "shap_value": round(float(v), 5)} for k, v in shap_pairs],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )[:5]

        results.append({
            "prediction":   int(predictions[i]),
            "probability":  round(prob, 4),
            "risk_level":   risk_level,
            "top_features": top_features,
        })

    return results


def predict_single(features: dict) -> dict:
    """Convenience wrapper: predict from a flat feature dictionary."""
    return predict(pd.DataFrame([features]))[0]
