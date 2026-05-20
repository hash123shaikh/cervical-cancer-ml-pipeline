import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR, TARGET_COLUMN, AUXILIARY_TARGETS, PREPROCESSING_STATS_PATH

logger = logging.getLogger(__name__)
ALL_TARGETS = [TARGET_COLUMN] + AUXILIARY_TARGETS
BINARY_COLUMNS = ["smokes","hormonal_contraceptives","iud","stds","stds_condylomatosis",
    "stds_cervical_condylomatosis","stds_vaginal_condylomatosis","stds_vulvo_perineal_condylomatosis",
    "stds_syphilis","stds_pelvic_inflammatory_disease","stds_genital_herpes","stds_molluscum_contagiosum",
    "stds_aids","stds_hiv","stds_hepatitis_b","stds_hpv","dx_cancer","dx_cin","dx_hpv","dx"]
CONTINUOUS_COLUMNS = ["age","num_sexual_partners","first_sexual_intercourse","num_of_pregnancies",
    "smokes_years","smokes_packs_year","hormonal_contraceptives_years","iud_years","stds_number",
    "stds_num_diagnosis","stds_time_first_diagnosis","stds_time_last_diagnosis"]

def preprocess(df, fit=True, stats=None):
    if not fit and stats is None:
        raise ValueError("`stats` must be supplied when fit=False.")
    df = df.copy().replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if fit:
        stats = {}
    for col in CONTINUOUS_COLUMNS:
        if col not in df.columns: continue
        if fit: stats[f"{col}_median"] = float(df[col].median())
        df[col] = df[col].fillna(stats[f"{col}_median"])
    for col in BINARY_COLUMNS:
        if col not in df.columns or col in ALL_TARGETS: continue
        if fit:
            mode_vals = df[col].mode()
            stats[f"{col}_mode"] = float(mode_vals[0]) if len(mode_vals) > 0 else 0.0
        df[col] = df[col].fillna(stats.get(f"{col}_mode", 0.0))
    for col in CONTINUOUS_COLUMNS:
        if col not in df.columns: continue
        if fit: stats[f"{col}_cap"] = float(df[col].quantile(0.99))
        cap = stats.get(f"{col}_cap")
        if cap is not None: df[col] = df[col].clip(upper=cap)
    return df, stats

def save_stats(stats):
    PREPROCESSING_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSING_STATS_PATH, "w") as f: json.dump(stats, f, indent=2)

def load_stats():
    if not PREPROCESSING_STATS_PATH.exists():
        raise FileNotFoundError(f"Stats not found at {PREPROCESSING_STATS_PATH}.")
    with open(PREPROCESSING_STATS_PATH) as f: return json.load(f)

def save_processed(df, split="train"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned = PROCESSED_DIR / f"features_{split}_{ts}.csv"
    latest    = PROCESSED_DIR / f"features_{split}_latest.csv"
    df.to_csv(versioned, index=False)
    df.to_csv(latest,    index=False)
    logger.info("Saved %s (%d rows)", versioned.name, len(df))
    return versioned

def load_latest(split="train"):
    path = PROCESSED_DIR / f"features_{split}_latest.csv"
    if not path.exists():
        raise FileNotFoundError(f"No processed '{split}' data at {path}.")
    return pd.read_csv(path)
