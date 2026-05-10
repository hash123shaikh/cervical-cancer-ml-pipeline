"""Tests for the risk stratifier (unsupervised branch)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


def _sample_df(n: int = 80) -> pd.DataFrame:
    """Generate a synthetic patient population large enough for clustering."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age":                   rng.integers(20, 70, n).astype(float),
        "smokes":                rng.integers(0, 2, n).astype(float),
        "stds_hpv":              rng.integers(0, 2, n).astype(float),
        "stds_hiv":              rng.integers(0, 2, n).astype(float),
        "stds_total":            rng.integers(0, 5, n).astype(float),
        "is_smoker":             rng.integers(0, 2, n).astype(float),
        "age_group":             rng.integers(0, 3, n).astype(float),
        "high_risk_stds":        rng.integers(0, 2, n).astype(float),
        "hormonal_contraceptives": rng.integers(0, 2, n).astype(float),
        "iud":                   rng.integers(0, 2, n).astype(float),
        "contraceptive_iud":     rng.integers(0, 2, n).astype(float),
        "biopsy":                rng.integers(0, 2, n).astype(float),
    })


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Redirect model artefacts to tmp dir so tests don't touch real models."""
    import config
    monkeypatch.setattr(config, "CLUSTER_MODEL_PATH",  tmp_path / "stratifier.joblib")
    monkeypatch.setattr(config, "CLUSTER_SUMMARY_PATH", tmp_path / "summary.json")
    monkeypatch.setattr(config, "MODELS_DIR",           tmp_path)


def test_fit_returns_summary_with_expected_keys():
    from src.clustering.risk_stratifier import fit_stratifier
    summary = fit_stratifier(_sample_df())
    for key in ("k", "n_patients", "stratum_distribution", "silhouette_score"):
        assert key in summary, f"Missing key: {key}"


def test_fit_produces_valid_strata():
    from src.clustering.risk_stratifier import fit_stratifier
    summary = fit_stratifier(_sample_df())
    strata = set(summary["stratum_distribution"].keys())
    assert strata.issubset({"low", "moderate", "high"})
    assert len(strata) >= 2


def test_silhouette_score_between_minus1_and_1():
    from src.clustering.risk_stratifier import fit_stratifier
    summary = fit_stratifier(_sample_df())
    assert -1.0 <= summary["silhouette_score"] <= 1.0


def test_stratum_counts_sum_to_n_patients():
    from src.clustering.risk_stratifier import fit_stratifier
    df = _sample_df(80)
    summary = fit_stratifier(df)
    total = sum(v["n_patients"] for v in summary["stratum_distribution"].values())
    assert total == len(df)


def test_assign_stratum_returns_one_result_per_row():
    from src.clustering.risk_stratifier import fit_stratifier, assign_stratum
    df = _sample_df(80)
    fit_stratifier(df)
    results = assign_stratum(df.head(10))
    assert len(results) == 10


def test_assign_stratum_valid_actions():
    from src.clustering.risk_stratifier import fit_stratifier, assign_stratum
    df = _sample_df(80)
    fit_stratifier(df)
    results = assign_stratum(df.head(20))
    valid_actions = {"routine_recall", "early_recall", "colposcopy_referral"}
    for r in results:
        assert r["action"] in valid_actions
        assert r["stratum"] in {"low", "moderate", "high"}
        assert isinstance(r["recall_months"], int)


def test_assign_stratum_recall_months_by_stratum():
    """Verify the policy mapping is internally consistent."""
    from src.clustering.risk_stratifier import fit_stratifier, assign_stratum
    df = _sample_df(80)
    fit_stratifier(df)
    results = assign_stratum(df)
    for r in results:
        if r["stratum"] == "low":
            assert r["recall_months"] == 36
        elif r["stratum"] == "moderate":
            assert r["recall_months"] == 12
        elif r["stratum"] == "high":
            assert r["recall_months"] == 0


def test_get_stratum_summary_after_fit():
    from src.clustering.risk_stratifier import fit_stratifier, get_stratum_summary
    fit_stratifier(_sample_df())
    summary = get_stratum_summary()
    assert "k" in summary
    assert "stratum_distribution" in summary


def test_get_stratum_summary_before_fit_returns_message():
    from src.clustering.risk_stratifier import get_stratum_summary
    result = get_stratum_summary()
    assert "message" in result
