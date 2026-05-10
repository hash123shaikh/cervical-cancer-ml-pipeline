"""
Feature Engineer — derives domain-informed composite features.

New features and their clinical rationale:
───────────────────────────────────────────
  stds_total       Sum of all 12 STD indicator columns.
                   STD history is a well-established composite risk factor;
                   exposing the aggregate count makes this signal explicit.

  is_smoker        Binary flag (smokes == 1).
                   Smoking is an independent risk factor for cervical cancer;
                   a clean binary flag avoids encoding noise from smokes_years.

  age_group        Ordinal: 0 = young (<25), 1 = middle (25–40), 2 = older (>40).
                   Cervical cancer incidence peaks in the 35–44 age group; an
                   ordinal grouping lets the model learn non-linear age effects
                   without relying solely on the continuous age feature.

  contraceptive_iud  Interaction: hormonal_contraceptives × iud.
                   Combined contraceptive use affects HPV persistence; the
                   interaction captures this joint effect.

  high_risk_stds   Binary: 1 if HIV or HPV detected.
                   HPV is the primary causal agent; HIV suppresses immune
                   clearance of HPV. Together they represent the highest-risk
                   STD profile and warrant a dedicated feature.
"""

import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TARGET_COLUMN, AUXILIARY_TARGETS

logger = logging.getLogger(__name__)

_STD_INDICATORS = [
    "stds_condylomatosis", "stds_cervical_condylomatosis",
    "stds_vaginal_condylomatosis", "stds_vulvo_perineal_condylomatosis",
    "stds_syphilis", "stds_pelvic_inflammatory_disease",
    "stds_genital_herpes", "stds_molluscum_contagiosum",
    "stds_aids", "stds_hiv", "stds_hepatitis_b", "stds_hpv",
]

_ALL_TARGETS = {TARGET_COLUMN, *AUXILIARY_TARGETS}
_EXCLUDE     = _ALL_TARGETS | {"ingested_at", "ingestion_id", "id"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to a preprocessed DataFrame.
    All original columns are preserved.
    """
    df = df.copy()

    # 1. Total STD count
    present = [c for c in _STD_INDICATORS if c in df.columns]
    df["stds_total"] = df[present].sum(axis=1)

    # 2. Is smoker
    if "smokes" in df.columns:
        df["is_smoker"] = (df["smokes"] == 1).astype(float)

    # 3. Age group (ordinal encoding)
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 40, 200],
            labels=[0, 1, 2],
            right=False,
        ).astype(float)

    # 4. Contraceptive × IUD interaction
    if {"hormonal_contraceptives", "iud"} <= set(df.columns):
        df["contraceptive_iud"] = df["hormonal_contraceptives"] * df["iud"]

    # 5. High-risk STD flag (HIV or HPV)
    high_risk = [c for c in ["stds_hiv", "stds_hpv"] if c in df.columns]
    if high_risk:
        df["high_risk_stds"] = (df[high_risk].sum(axis=1) > 0).astype(float)

    logger.info(
        "Feature engineering complete: %d features (%d new)",
        len(df.columns), len(df.columns) - len(df.columns) + 5,
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return columns to use as model inputs.
    Excludes all target columns and pipeline bookkeeping fields.
    """
    return [c for c in df.columns if c not in _EXCLUDE]
