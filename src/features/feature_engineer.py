import logging
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TARGET_COLUMN, AUXILIARY_TARGETS

logger = logging.getLogger(__name__)
_STD_INDICATORS = ["stds_condylomatosis","stds_cervical_condylomatosis","stds_vaginal_condylomatosis",
    "stds_vulvo_perineal_condylomatosis","stds_syphilis","stds_pelvic_inflammatory_disease",
    "stds_genital_herpes","stds_molluscum_contagiosum","stds_aids","stds_hiv","stds_hepatitis_b","stds_hpv"]
_EXCLUDE = {TARGET_COLUMN, *AUXILIARY_TARGETS, "ingested_at","ingestion_id","id"}

def engineer_features(df):
    df = df.copy()
    present = [c for c in _STD_INDICATORS if c in df.columns]
    df["stds_total"] = df[present].sum(axis=1)
    if "smokes" in df.columns:
        df["is_smoker"] = (df["smokes"] == 1).astype(float)
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[0,25,40,200], labels=[0,1,2], right=False).astype(float)
    if {"hormonal_contraceptives","iud"} <= set(df.columns):
        df["contraceptive_iud"] = df["hormonal_contraceptives"] * df["iud"]
    high_risk = [c for c in ["stds_hiv","stds_hpv"] if c in df.columns]
    if high_risk:
        df["high_risk_stds"] = (df[high_risk].sum(axis=1) > 0).astype(float)
    return df

def get_feature_columns(df):
    return [c for c in df.columns if c not in _EXCLUDE]
