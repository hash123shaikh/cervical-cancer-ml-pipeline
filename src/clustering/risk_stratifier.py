"""
Risk Stratifier — unsupervised population risk stratification.

Why unsupervised learning alongside the supervised classifier?
──────────────────────────────────────────────────────────────
The supervised XGBoost model answers a patient-level question:
  "What is the probability that THIS individual has a positive biopsy?"

But a cervical cancer screening programme must answer a population-level question:
  "How should we allocate limited screening resources across ALL patients?"

These are different problems. A national screening guideline does not assign a
unique follow-up interval to each of the millions of women in the programme based
on a continuous probability score. It defines discrete risk strata and maps each
stratum to a recall policy:

  Low risk    → routine recall (3 years)
  Moderate    → earlier recall (12 months)
  High risk   → immediate colposcopy referral

This is exactly what unsupervised clustering produces: a data-driven stratification
of the patient population into groups with coherent risk profiles. The stratum is
then COMBINED with the individual probability from XGBoost to generate the final
screening recommendation — using both the population context and the individual
signal.

This two-branch design directly mirrors the approach described in the doctoral
project brief: developing risk-based screening recommendations by identifying
risk patterns in population-scale data, rather than just classifying individuals.

Clustering approach: K-means
─────────────────────────────
K-means is chosen over hierarchical or density-based methods because:
  1. It scales to national registry size (millions of patients)
  2. Cluster boundaries are interpretable by clinicians and epidemiologists
  3. New patients can be assigned to a stratum with a single distance computation
  4. It is the standard approach in population health stratification literature

k=3 strata (low / moderate / high) is the default because it maps directly to
the three-tier recall structure used in most European cervical screening programmes.
The optimal k is validated at training time using silhouette score across k=2..5.

How strata are labelled:
  Clusters are sorted by their mean predicted cancer probability (from XGBoost).
  The cluster with the lowest mean probability → 'low', highest → 'high'.
  This makes the labels stable across model retrains.

Screening interval mapping:
  These intervals are illustrative. In a production system they would be
  calibrated against the specific programme's outcome data and validated
  against national guidelines (e.g. IARC, ECDC, NBHW Sweden).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    CLUSTER_MODEL_PATH, CLUSTER_SUMMARY_PATH,
    N_RISK_STRATA, TARGET_COLUMN, AUXILIARY_TARGETS,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)

_EXCLUDE = {TARGET_COLUMN, *AUXILIARY_TARGETS, "ingested_at", "ingestion_id", "id"}

# Screening interval policy — illustrative; calibrate against programme data
_INTERVAL_POLICY = {
    "low":      {"recall_months": 36, "action": "routine_recall",      "label": "Routine 3-year recall"},
    "moderate": {"recall_months": 12, "action": "early_recall",         "label": "Early 12-month recall"},
    "high":     {"recall_months":  0, "action": "colposcopy_referral",  "label": "Immediate colposcopy referral"},
}

_STRATUM_NAMES = ["low", "moderate", "high"]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _feature_matrix(df: pd.DataFrame) -> tuple:
    """Return (X array, feature column list) excluding target and bookkeeping cols."""
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    X = df[feature_cols].astype(float).values
    return X, feature_cols


def _select_k(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 5) -> int:
    """
    Select optimal number of clusters using silhouette score.
    Falls back to N_RISK_STRATA if validation fails.
    """
    if len(X_scaled) < k_max * 2:
        logger.warning(
            "Dataset too small (%d rows) for k-selection — using k=%d.",
            len(X_scaled), N_RISK_STRATA,
        )
        return N_RISK_STRATA

    best_k, best_score = N_RISK_STRATA, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels, sample_size=min(1000, len(X_scaled)))
        logger.info("  k=%d → silhouette=%.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score

    logger.info("Optimal k=%d (silhouette=%.4f)", best_k, best_score)
    return best_k


def _label_clusters(kmeans: KMeans, X_scaled: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Map raw KMeans cluster IDs to 'low'/'moderate'/'high' labels.

    Strategy: use the mean biopsy rate per cluster as the ordering signal.
    If biopsy column is absent, fall back to mean distance to centroid.
    This ensures labels are stable and clinically meaningful across retrains.
    """
    cluster_ids = kmeans.predict(X_scaled)
    k = kmeans.n_clusters

    if TARGET_COLUMN in df.columns:
        mean_risk = {
            i: df[TARGET_COLUMN].iloc[cluster_ids == i].mean()
            for i in range(k)
        }
    else:
        # Fall back: mean distance to nearest centroid (lower = more tightly packed = lower risk)
        distances = np.min(
            np.array([np.linalg.norm(X_scaled - c, axis=1) for c in kmeans.cluster_centers_]),
            axis=0,
        )
        mean_risk = {i: float(distances[cluster_ids == i].mean()) for i in range(k)}

    # Sort clusters by ascending mean risk → assign low/moderate/high
    sorted_clusters = sorted(mean_risk.items(), key=lambda x: x[1])
    n = len(sorted_clusters)

    if n == 2:
        mapping = {sorted_clusters[0][0]: "low", sorted_clusters[1][0]: "high"}
    elif n == 3:
        mapping = {
            sorted_clusters[0][0]: "low",
            sorted_clusters[1][0]: "moderate",
            sorted_clusters[2][0]: "high",
        }
    else:
        # For k > 3, compress to 3 strata by thirds
        mapping = {}
        for idx, (cid, _) in enumerate(sorted_clusters):
            if idx < n // 3:
                mapping[cid] = "low"
            elif idx < 2 * n // 3:
                mapping[cid] = "moderate"
            else:
                mapping[cid] = "high"

    return mapping


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_stratifier(df: pd.DataFrame) -> dict:
    """
    Fit the K-means risk stratifier on the full population dataset.

    Saves:
      models/risk_stratifier.joblib   — (scaler, kmeans, cluster_to_stratum mapping)
      models/stratum_summary.json     — per-stratum statistics for monitoring

    Returns a summary dict suitable for logging and the pipeline run record.
    """
    X, feature_cols = _feature_matrix(df)

    # ── Scale features (K-means is distance-based → scaling is mandatory) ──────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Select k ──────────────────────────────────────────────────────────────
    k = _select_k(X_scaled)

    # ── Fit K-means ───────────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)

    # ── Label clusters clinically ─────────────────────────────────────────────
    cluster_to_stratum = _label_clusters(kmeans, X_scaled, df)
    cluster_ids        = kmeans.predict(X_scaled)
    strata             = np.array([cluster_to_stratum[c] for c in cluster_ids])

    # ── Per-stratum statistics ────────────────────────────────────────────────
    stratum_stats = {}
    for stratum in set(cluster_to_stratum.values()):
        mask = strata == stratum
        n    = int(mask.sum())
        stats: dict = {
            "n_patients":  n,
            "pct_patients": round(100 * n / len(df), 1),
            "policy": _INTERVAL_POLICY[stratum],
        }
        if TARGET_COLUMN in df.columns:
            stats["mean_biopsy_rate"] = round(
                float(df[TARGET_COLUMN].iloc[mask].mean()), 4
            )
        stratum_stats[stratum] = stats

    # ── Persist ───────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artefact = {
        "scaler":            scaler,
        "kmeans":            kmeans,
        "cluster_to_stratum": cluster_to_stratum,
        "feature_cols":      feature_cols,
    }
    joblib.dump(artefact, CLUSTER_MODEL_PATH)

    summary = {
        "k": k,
        "n_patients": len(df),
        "stratum_distribution": stratum_stats,
        "silhouette_score": round(
            float(silhouette_score(X_scaled, cluster_ids, sample_size=min(1000, len(X_scaled)))),
            4,
        ),
    }
    with open(CLUSTER_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "✓ Risk stratifier fitted: k=%d, silhouette=%.3f | distribution: %s",
        k,
        summary["silhouette_score"],
        {s: v["n_patients"] for s, v in stratum_stats.items()},
    )
    return summary


def assign_stratum(df: pd.DataFrame) -> list:
    """
    Assign each row in df to a risk stratum using the saved stratifier.

    Returns a list of dicts, one per row:
      {
        "stratum":        str,   # "low" | "moderate" | "high"
        "recall_months":  int,   # 0 = immediate referral
        "action":         str,
        "label":          str,
      }
    """
    artefact = _load_artefact()
    scaler, kmeans = artefact["scaler"], artefact["kmeans"]
    cluster_to_stratum = artefact["cluster_to_stratum"]
    feature_cols       = artefact["feature_cols"]

    # Align columns — fill missing with 0 (conservative)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        logger.warning("Missing %d feature columns; filling with 0.", len(missing))
        for col in missing:
            df[col] = 0.0

    X        = df[feature_cols].astype(float).values
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)

    results = []
    for c in clusters:
        stratum = cluster_to_stratum.get(int(c), "moderate")
        results.append({"stratum": stratum, **_INTERVAL_POLICY[stratum]})
    return results


def get_stratum_summary() -> dict:
    """Return the saved stratum summary for the monitoring endpoint."""
    if not CLUSTER_SUMMARY_PATH.exists():
        return {
            "message": "Stratifier not yet fitted. "
                       "Run pipeline via POST /pipeline/trigger."
        }
    with open(CLUSTER_SUMMARY_PATH) as f:
        return json.load(f)


def _load_artefact() -> dict:
    if not CLUSTER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Risk stratifier not found at {CLUSTER_MODEL_PATH}. "
            "Run `python scripts/initial_train.py` first."
        )
    return joblib.load(CLUSTER_MODEL_PATH)
