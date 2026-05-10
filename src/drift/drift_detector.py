"""
Drift Detector — monitors for data distribution shift using Evidently AI.

Strategy overview
──────────────────
1.  Baseline
    After initial training, the reference dataset (first 600 rows) is saved
    as a Parquet file. This represents the expected population distribution.

2.  Detection
    Each new batch is compared against the reference using Evidently's
    DataDriftPreset, which applies per-column statistical tests:
      • Wasserstein distance for continuous features
      • Jensen-Shannon divergence for binary/categorical features
    These tests are distribution-free and do not assume normality.

3.  Decision rule
    If > DRIFT_THRESHOLD (default 15%) of feature columns show drift, we flag
    retraining. 15% is intentionally conservative:
      - Cervical cancer screening data changes slowly (patient demographics,
        screening guidelines shift over years, not weeks)
      - False alarms on small daily batches (15 rows) are noisy; a stricter
        threshold reduces operational churn from spurious alerts
      - In a production system, a rolling 7-day window would be used for more
        stable estimates

4.  Response
    a. Full HTML + JSON report saved to /reports/
    b. Pipeline scheduler triggers retraining on the same cycle
    c. New model's reference statistics are updated to the current distribution

5.  Limitations acknowledged
    We monitor feature distributions (covariate shift), not label distributions
    (concept drift). A complete production system would also track:
      - Prediction calibration over time (reliability diagrams)
      - Positive-rate shifts in arriving labels once ground-truth is available
      - Model performance degradation on a labelled holdout
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    REFERENCE_STATS_PATH, DRIFT_THRESHOLD,
    DRIFT_REPORT_PATH, DRIFT_STATUS_PATH,
    TARGET_COLUMN, AUXILIARY_TARGETS,
)

logger = logging.getLogger(__name__)

_EXCLUDE = {TARGET_COLUMN, *AUXILIARY_TARGETS, "ingested_at", "ingestion_id", "id"}


def save_reference(df: pd.DataFrame) -> None:
    """Persist the reference dataset for future drift comparisons."""
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    REFERENCE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[feature_cols].to_parquet(str(REFERENCE_STATS_PATH), index=False)
    logger.info("Reference saved: %d rows × %d features", len(df), len(feature_cols))


def load_reference() -> pd.DataFrame:
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_STATS_PATH}. "
            "Run `python scripts/initial_train.py` first."
        )
    return pd.read_parquet(str(REFERENCE_STATS_PATH))


def check_drift(current_df: pd.DataFrame) -> dict:
    """
    Compare a new batch against the reference distribution.

    Args:
        current_df: Processed + feature-engineered batch DataFrame.

    Returns:
        Status dict with drift flag, drifted column names, share drifted,
        and paths to the saved HTML and JSON reports.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        raise ImportError(
            "Evidently is not installed. Run: pip install evidently"
        )

    reference_df = load_reference()
    feature_cols = [c for c in current_df.columns if c not in _EXCLUDE]
    common_cols  = [c for c in feature_cols if c in reference_df.columns]

    ref = reference_df[common_cols].astype(float)
    cur = current_df[common_cols].astype(float)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(DRIFT_REPORT_PATH))

    result       = report.as_dict()
    drift_result = result["metrics"][0]["result"]

    drifted_columns = [
        col for col, data in drift_result.get("drift_by_columns", {}).items()
        if data.get("drift_detected", False)
    ]
    share_drifted  = drift_result.get("share_of_drifted_columns", 0.0)
    drift_detected = float(share_drifted) > DRIFT_THRESHOLD

    status = {
        "drift_detected":   drift_detected,
        "drifted_columns":  drifted_columns,
        "share_drifted":    round(float(share_drifted), 4),
        "threshold":        DRIFT_THRESHOLD,
        "n_columns_tested": len(common_cols),
        "report_path":      str(DRIFT_REPORT_PATH),
        "checked_at":       datetime.utcnow().isoformat(),
        "reference_rows":   len(ref),
        "current_rows":     len(cur),
    }

    with open(DRIFT_STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

    if drift_detected:
        logger.warning(
            "⚠ Data drift detected: %.0f%% of columns shifted "
            "(threshold %.0f%%). Drifted: %s",
            100 * float(share_drifted), 100 * DRIFT_THRESHOLD, drifted_columns,
        )
    else:
        logger.info(
            "✓ No drift detected: %.0f%% of columns shifted (threshold %.0f%%).",
            100 * float(share_drifted), 100 * DRIFT_THRESHOLD,
        )

    return status


def get_current_status() -> dict:
    """Return the most recently saved drift status dict."""
    if not DRIFT_STATUS_PATH.exists():
        return {
            "drift_detected": False,
            "message": "No drift check has been run yet. "
                       "Trigger the pipeline via POST /pipeline/trigger.",
        }
    with open(DRIFT_STATUS_PATH) as f:
        return json.load(f)
