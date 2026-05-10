"""
Preprocessor — cleans raw data and produces processed feature tables.

Why CSV flat files for processed data (not SQLite)?
────────────────────────────────────────────────────
Processed feature tensors are accessed in one way: bulk sequential reads by
pandas for model training and drift detection. For this access pattern CSV is:
  • Faster: no query planner, no row deserialisation
  • Trivially versioned: one timestamped file per pipeline run
  • Inspectable: any data scientist can open it in Excel, pandas, or a notebook
  • Simple: the preprocessor is a pure function (DataFrame → CSV); adding a
    database layer would introduce complexity with no operational benefit here

SQLite would be appropriate if we needed random access, concurrent writes from
multiple processes, or complex aggregation queries — none of which apply here.

Processing steps (in order):
  1. Replace "?" placeholders (UCI missing-value convention) with NaN
  2. Cast all columns to numeric (every field in this dataset is numeric)
  3. Impute missing values:
       • Median imputation for continuous features (robust to outliers)
       • Mode imputation for binary (0/1) features (preserve majority class)
  4. Clip extreme outliers at the 99th percentile of the training distribution
     (prevents a handful of extreme values from dominating tree splits)
  5. Save stats (medians, modes, caps) so inference-time preprocessing is
     identical to training-time preprocessing — avoids data leakage
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    PROCESSED_DIR, TARGET_COLUMN, AUXILIARY_TARGETS,
    PREPROCESSING_STATS_PATH,
)

logger = logging.getLogger(__name__)

ALL_TARGETS = [TARGET_COLUMN] + AUXILIARY_TARGETS

BINARY_COLUMNS = [
    "smokes", "hormonal_contraceptives", "iud", "stds",
    "stds_condylomatosis", "stds_cervical_condylomatosis",
    "stds_vaginal_condylomatosis", "stds_vulvo_perineal_condylomatosis",
    "stds_syphilis", "stds_pelvic_inflammatory_disease",
    "stds_genital_herpes", "stds_molluscum_contagiosum",
    "stds_aids", "stds_hiv", "stds_hepatitis_b", "stds_hpv",
    "dx_cancer", "dx_cin", "dx_hpv", "dx",
]

CONTINUOUS_COLUMNS = [
    "age", "num_sexual_partners", "first_sexual_intercourse",
    "num_of_pregnancies", "smokes_years", "smokes_packs_year",
    "hormonal_contraceptives_years", "iud_years", "stds_number",
    "stds_num_diagnosis", "stds_time_first_diagnosis",
    "stds_time_last_diagnosis",
]


def preprocess(
    df: pd.DataFrame,
    fit: bool = True,
    stats: Optional[dict] = None,
) -> tuple:
    """
    Clean and preprocess a raw DataFrame.

    Args:
        df:    Input DataFrame (raw rows from SQLite or reference set)
        fit:   If True, compute imputation statistics from this data (training time).
               If False, apply pre-computed stats (inference/drift time).
               Using pre-computed stats at inference time prevents data leakage.
        stats: Required when fit=False. Pass the dict returned by a prior
               fit=True call, or load with load_stats().

    Returns:
        (cleaned_df, stats_dict)
    """
    if not fit and stats is None:
        raise ValueError(
            "`stats` must be supplied when fit=False. "
            "Load saved stats with preprocessor.load_stats()."
        )

    df = df.copy()

    # ── Step 1: Replace UCI missing-value placeholder ──────────────────────────
    df = df.replace("?", np.nan)

    # ── Step 2: Cast to numeric ────────────────────────────────────────────────
    for col in df.columns:
        if col not in ALL_TARGETS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Step 3: Imputation ─────────────────────────────────────────────────────
    if fit:
        stats = {}

    for col in CONTINUOUS_COLUMNS:
        if col not in df.columns:
            continue
        if fit:
            stats[f"{col}_median"] = float(df[col].median())
        df[col] = df[col].fillna(stats[f"{col}_median"])

    for col in BINARY_COLUMNS:
        if col not in df.columns or col in ALL_TARGETS:
            continue
        if fit:
            mode_vals = df[col].mode()
            stats[f"{col}_mode"] = float(mode_vals[0]) if len(mode_vals) > 0 else 0.0
        df[col] = df[col].fillna(stats.get(f"{col}_mode", 0.0))

    # ── Step 4: Outlier clipping (99th percentile of training distribution) ─────
    for col in CONTINUOUS_COLUMNS:
        if col not in df.columns:
            continue
        if fit:
            stats[f"{col}_cap"] = float(df[col].quantile(0.99))
        cap = stats.get(f"{col}_cap")
        if cap is not None:
            df[col] = df[col].clip(upper=cap)

    n_missing = df.drop(columns=ALL_TARGETS, errors="ignore").isnull().sum().sum()
    logger.info(
        "Preprocessing complete: %d rows × %d cols | residual NaN: %d",
        len(df), len(df.columns), n_missing,
    )
    return df, stats


# ── Persistence ────────────────────────────────────────────────────────────────

def save_stats(stats: dict) -> None:
    """Persist imputation statistics so inference uses identical transforms."""
    PREPROCESSING_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSING_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Preprocessing stats saved to %s", PREPROCESSING_STATS_PATH)


def load_stats() -> dict:
    """Load saved imputation statistics for inference-time preprocessing."""
    if not PREPROCESSING_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing stats not found at {PREPROCESSING_STATS_PATH}. "
            "Run `python scripts/initial_train.py` first."
        )
    with open(PREPROCESSING_STATS_PATH) as f:
        return json.load(f)


def save_processed(df: pd.DataFrame, split: str = "train") -> Path:
    """
    Save a processed DataFrame as a timestamped CSV and update the 'latest' pointer.

    Args:
        df:    Processed DataFrame
        split: 'train' | 'stream' | 'reference' — used in the filename

    Returns:
        Path to the versioned file.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned = PROCESSED_DIR / f"features_{split}_{ts}.csv"
    latest    = PROCESSED_DIR / f"features_{split}_latest.csv"

    df.to_csv(versioned, index=False)
    df.to_csv(latest,    index=False)

    logger.info("Saved processed features: %s (%d rows)", versioned.name, len(df))
    return versioned


def load_latest(split: str = "train") -> pd.DataFrame:
    """Load the most recent processed CSV for a given split."""
    path = PROCESSED_DIR / f"features_{split}_latest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No processed '{split}' data at {path}. "
            "Run the pipeline first."
        )
    return pd.read_csv(path)
