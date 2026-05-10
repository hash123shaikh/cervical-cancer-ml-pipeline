"""
Central configuration for the cervical cancer ML pipeline.

All paths, constants, and environment-variable overrides live here.
Importing this module also ensures required directories exist.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
LOGS_DIR      = BASE_DIR / "logs"

for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00383/risk_factors_cervical_cancer.csv"
)
RAW_CSV_PATH   = RAW_DIR / "risk_factors_cervical_cancer.csv"
RAW_DB_PATH    = RAW_DIR / "cervical_cancer.db"
STREAM_STATE_PATH = RAW_DIR / "stream_state.json"

TARGET_COLUMN     = "biopsy"
AUXILIARY_TARGETS = ["hinselmann", "schiller", "citology"]

# Rows reserved as the historical / initial-training set;
# everything beyond this index simulates the arriving stream.
REFERENCE_SIZE   = 600
DAILY_BATCH_SIZE = int(os.getenv("DAILY_BATCH_SIZE", "15"))

# ── Column renaming: raw UCI header → snake_case ───────────────────────────────
COLUMN_RENAME: dict = {
    "Age":                                  "age",
    "Number of sexual partners":            "num_sexual_partners",
    "First sexual intercourse":             "first_sexual_intercourse",
    "Num of pregnancies":                   "num_of_pregnancies",
    "Smokes":                               "smokes",
    "Smokes (years)":                       "smokes_years",
    "Smokes (packs/year)":                  "smokes_packs_year",
    "Hormonal Contraceptives":              "hormonal_contraceptives",
    "Hormonal Contraceptives (years)":      "hormonal_contraceptives_years",
    "IUD":                                  "iud",
    "IUD (years)":                          "iud_years",
    "STDs":                                 "stds",
    "STDs (number)":                        "stds_number",
    "STDs:condylomatosis":                  "stds_condylomatosis",
    "STDs:cervical condylomatosis":         "stds_cervical_condylomatosis",
    "STDs:vaginal condylomatosis":          "stds_vaginal_condylomatosis",
    "STDs:vulvo-perineal condylomatosis":   "stds_vulvo_perineal_condylomatosis",
    "STDs:syphilis":                        "stds_syphilis",
    "STDs:pelvic inflammatory disease":     "stds_pelvic_inflammatory_disease",
    "STDs:genital herpes":                  "stds_genital_herpes",
    "STDs:molluscum contagiosum":           "stds_molluscum_contagiosum",
    "STDs:AIDS":                            "stds_aids",
    "STDs:HIV":                             "stds_hiv",
    "STDs:Hepatitis B":                     "stds_hepatitis_b",
    "STDs:HPV":                             "stds_hpv",
    "STDs: Number of diagnosis":            "stds_num_diagnosis",
    "STDs: Time since first diagnosis":     "stds_time_first_diagnosis",
    "STDs: Time since last diagnosis":      "stds_time_last_diagnosis",
    "Dx:Cancer":                            "dx_cancer",
    "Dx:CIN":                               "dx_cin",
    "Dx:HPV":                               "dx_hpv",
    "Dx":                                   "dx",
    "Hinselmann":                           "hinselmann",
    "Schiller":                             "schiller",
    "Citology":                             "citology",
    "Biopsy":                               "biopsy",
}

# ── Model artefacts ────────────────────────────────────────────────────────────
MODEL_PATH              = MODELS_DIR / "xgboost_model.joblib"
FEATURE_NAMES_PATH      = MODELS_DIR / "feature_names.json"
MODEL_METRICS_PATH      = MODELS_DIR / "metrics.json"
PREPROCESSING_STATS_PATH = MODELS_DIR / "preprocessing_stats.json"

# ── Clustering / risk stratification ──────────────────────────────────────────
CLUSTER_MODEL_PATH   = MODELS_DIR / "risk_stratifier.joblib"
CLUSTER_SUMMARY_PATH = MODELS_DIR / "stratum_summary.json"
N_RISK_STRATA        = int(os.getenv("N_RISK_STRATA", "3"))   # low / moderate / high

# ── Drift detection ────────────────────────────────────────────────────────────
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.parquet"
DRIFT_THRESHOLD      = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
DRIFT_REPORT_PATH    = REPORTS_DIR / "drift_report.html"
DRIFT_STATUS_PATH    = REPORTS_DIR / "drift_status.json"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULE_HOUR   = int(os.getenv("SCHEDULE_HOUR",   "6"))
SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "0"))

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = LOGS_DIR / "pipeline.log"
