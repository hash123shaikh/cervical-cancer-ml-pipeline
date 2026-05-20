import json
import logging
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_CSV_PATH, STREAM_STATE_PATH, REFERENCE_SIZE, DAILY_BATCH_SIZE, COLUMN_RENAME

logger = logging.getLogger(__name__)

def _load_state():
    if STREAM_STATE_PATH.exists():
        with open(STREAM_STATE_PATH) as f:
            return json.load(f)
    return {"current_index": REFERENCE_SIZE, "last_run_date": None, "batches_generated": 0}

def _save_state(state):
    STREAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STREAM_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

def _load_dataset():
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_CSV_PATH}. Run scripts/download_data.py first.")
    df = pd.read_csv(RAW_CSV_PATH)
    return df.rename(columns=COLUMN_RENAME)

def get_reference_data():
    df = _load_dataset()
    return df.iloc[:REFERENCE_SIZE].copy().reset_index(drop=True)

def get_next_batch(batch_size=None):
    batch_size = batch_size or DAILY_BATCH_SIZE
    df = _load_dataset()
    state = _load_state()
    idx = state["current_index"]
    if idx >= len(df):
        logger.warning("Stream exhausted. Wrapping to start.")
        idx = REFERENCE_SIZE
        state["current_index"] = idx
    batch = df.iloc[idx: idx + batch_size].copy().reset_index(drop=True)
    batch["ingested_at"] = datetime.utcnow().isoformat()
    state["current_index"] = idx + len(batch)
    state["last_run_date"] = date.today().isoformat()
    state["batches_generated"] = state.get("batches_generated", 0) + 1
    _save_state(state)
    logger.info("Batch #%d: %d rows (idx %d-%d)", state["batches_generated"], len(batch), idx, idx+len(batch)-1)
    return batch

def get_stream_status():
    df = _load_dataset()
    state = _load_state()
    total = len(df) - REFERENCE_SIZE
    consumed = state["current_index"] - REFERENCE_SIZE
    return {"total_stream_rows": total, "consumed_rows": consumed,
            "remaining_rows": max(0, total - consumed),
            "batches_generated": state.get("batches_generated", 0),
            "last_run_date": state.get("last_run_date")}

def reset_stream():
    _save_state({"current_index": REFERENCE_SIZE, "last_run_date": None, "batches_generated": 0})
