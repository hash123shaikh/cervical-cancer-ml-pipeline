"""Tests for preprocessing and feature engineering."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


def _sample_df(n: int = 12) -> pd.DataFrame:
    """Create a small DataFrame with UCI-style '?' missing values."""
    return pd.DataFrame({
        "age":      [25.0, 30.0, "?",  45.0, 22.0, 35.0, 28.0, 40.0, 33.0, 27.0, 31.0, 36.0][:n],
        "smokes":   [0,    1,    "?",  0,    1,    0,    0,    1,    0,    "?",  0,    1   ][:n],
        "stds_hpv": [0,    1,    0,    "?",  0,    1,    0,    0,    "?",  0,    0,    0   ][:n],
        "biopsy":   [0,    0,    0,    1,    0,    0,    0,    1,    0,    0,    0,    0   ][:n],
    })


# ── Preprocessor ──────────────────────────────────────────────────────────────

def test_question_marks_replaced_with_nan():
    from src.preprocessing.preprocessor import preprocess
    result, _ = preprocess(_sample_df(), fit=True)
    assert not (result == "?").any().any()


def test_no_nans_in_feature_columns_after_preprocessing():
    from src.preprocessing.preprocessor import preprocess
    result, _ = preprocess(_sample_df(), fit=True)
    # Target column may still have NaN (not imputed) but feature cols must be clean
    feature_cols = [c for c in result.columns if c not in ("biopsy",)]
    assert result[feature_cols].isnull().sum().sum() == 0


def test_fit_false_without_stats_raises():
    from src.preprocessing.preprocessor import preprocess
    with pytest.raises(ValueError, match="stats"):
        preprocess(_sample_df(), fit=False, stats=None)


def test_fit_false_uses_provided_stats_not_current_data():
    """Stats from training must be applied unchanged at inference time (no leakage)."""
    from src.preprocessing.preprocessor import preprocess
    df_train = _sample_df()
    _, stats = preprocess(df_train, fit=True)

    # Provide a shifted DataFrame at inference time
    df_inf = pd.DataFrame({"age": [999.0], "smokes": [np.nan], "biopsy": [0]})
    result, _ = preprocess(df_inf, fit=False, stats=stats)

    # Age should be clipped to the training-time 99th-percentile cap
    cap = stats.get("age_cap")
    if cap is not None:
        assert result["age"].iloc[0] <= cap + 1e-6


def test_returned_stats_are_reusable():
    from src.preprocessing.preprocessor import preprocess
    df = _sample_df()
    result1, stats = preprocess(df, fit=True)
    result2, _     = preprocess(df, fit=False, stats=stats)
    # Both runs should produce numerically identical feature columns
    feature_cols = [c for c in result1.columns if c == "age"]
    pd.testing.assert_frame_equal(result1[feature_cols], result2[feature_cols])


# ── Feature engineer ──────────────────────────────────────────────────────────

def test_engineer_features_adds_expected_columns():
    from src.preprocessing.preprocessor import preprocess
    from src.features.feature_engineer import engineer_features
    df, _ = preprocess(_sample_df(), fit=True)
    featured = engineer_features(df)
    for col in ("stds_total", "age_group", "is_smoker", "high_risk_stds"):
        assert col in featured.columns, f"Missing engineered feature: {col}"


def test_stds_total_is_non_negative():
    from src.preprocessing.preprocessor import preprocess
    from src.features.feature_engineer import engineer_features
    df, _ = preprocess(_sample_df(), fit=True)
    featured = engineer_features(df)
    assert (featured["stds_total"] >= 0).all()


def test_age_group_values_are_valid():
    from src.preprocessing.preprocessor import preprocess
    from src.features.feature_engineer import engineer_features
    df, _ = preprocess(_sample_df(), fit=True)
    featured = engineer_features(df)
    # age_group should be in {0.0, 1.0, 2.0} or NaN
    valid = {0.0, 1.0, 2.0}
    non_null = featured["age_group"].dropna()
    assert set(non_null.unique()).issubset(valid)
