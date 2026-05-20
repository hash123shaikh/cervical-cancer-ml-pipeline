import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import REFERENCE_STATS_PATH, DRIFT_THRESHOLD, DRIFT_REPORT_PATH, DRIFT_STATUS_PATH, TARGET_COLUMN, AUXILIARY_TARGETS

logger = logging.getLogger(__name__)
_EXCLUDE = {TARGET_COLUMN, *AUXILIARY_TARGETS, "ingested_at","ingestion_id","id"}

def save_reference(df):
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    REFERENCE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[feature_cols].to_parquet(str(REFERENCE_STATS_PATH), index=False)
    logger.info("Reference saved: %d rows", len(df))

def load_reference():
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(f"Reference not found at {REFERENCE_STATS_PATH}.")
    return pd.read_parquet(str(REFERENCE_STATS_PATH))

def check_drift(current_df):
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
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
    drifted_columns = [col for col,data in drift_result.get("drift_by_columns",{}).items() if data.get("drift_detected",False)]
    share_drifted   = drift_result.get("share_of_drifted_columns", 0.0)
    drift_detected  = float(share_drifted) > DRIFT_THRESHOLD
    status = {"drift_detected":drift_detected,"drifted_columns":drifted_columns,
        "share_drifted":round(float(share_drifted),4),"threshold":DRIFT_THRESHOLD,
        "n_columns_tested":len(common_cols),"report_path":str(DRIFT_REPORT_PATH),
        "checked_at":datetime.utcnow().isoformat(),"reference_rows":len(ref),"current_rows":len(cur)}
    with open(DRIFT_STATUS_PATH,"w") as f: json.dump(status, f, indent=2)
    logger.info("Drift check: detected=%s (%.0f%% columns)", drift_detected, 100*float(share_drifted))
    return status

def get_current_status():
    if not DRIFT_STATUS_PATH.exists():
        return {"drift_detected":False,"message":"No drift check run yet. POST /pipeline/trigger to start."}
    with open(DRIFT_STATUS_PATH) as f: return json.load(f)
