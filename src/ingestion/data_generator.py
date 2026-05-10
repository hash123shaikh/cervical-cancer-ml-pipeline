"""
Data Generator — simulates a daily batch data stream.

Why daily batches (not real-time streaming)?
─────────────────────────────────────────────
In a cervical cancer screening programme, data does not arrive continuously:
  • Biopsy results require 2–5 days of lab processing before entry
  • Screening programmes process patients in cohort batches (per clinic, per region)
  • A daily ingestion cadence matches the realistic data latency in this domain
  • Real-time streaming would be appropriate for an on-device triage tool; for
    population-level risk modelling, daily aggregation is the right fit

Simulation strategy:
  1. The full UCI dataset (858 patients) is treated as the historical record
  2. The first REFERENCE_SIZE (600) rows are the "historical" set used for
     initial model training
  3. The remaining 258 rows are released DAILY_BATCH_SIZE rows per simulated day
  4. Position is persisted in a JSON state file so the generator survives restarts

In a real deployment this function would be replaced by a call to a clinical
data warehouse API, an HL7 FHIR endpoint, or a message queue consumer.
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    RAW_CSV_PATH, STREAM_STATE_PATH, REFERENCE_SIZE,
    DAILY_BATCH_SIZE, COLUMN_RENAME,
)

logger = logging.getLogger(__name__)


# ── State persistence ──────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STREAM_STATE_PATH.exists():
        with open(STREAM_STATE_PATH) as f:
            return json.load(f)
    return {"current_index": REFERENCE_SIZE, "last_run_date": None, "batches_generated": 0}


def _save_state(state: dict) -> None:
    STREAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STREAM_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _load_dataset() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_CSV_PATH}. "
            "Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(RAW_CSV_PATH)
    return df.rename(columns=COLUMN_RENAME)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_reference_data() -> pd.DataFrame:
    """
    Return the reference dataset (first REFERENCE_SIZE rows).

    Used for:
      • Initial model training
      • Drift detection baseline (the "expected" distribution)
    """
    df = _load_dataset()
    return df.iloc[:REFERENCE_SIZE].copy().reset_index(drop=True)


def get_next_batch(batch_size: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Return the next daily batch from the stream set.

    Returns None if the stream is exhausted (wraps around for continuous demo).
    In production this would call an external API / read from a message queue.
    """
    batch_size = batch_size or DAILY_BATCH_SIZE
    df = _load_dataset()
    state = _load_state()
    idx = state["current_index"]

    if idx >= len(df):
        logger.warning(
            "Stream exhausted (%d/%d rows consumed). "
            "Wrapping to start of stream for continuous demo operation.", idx, len(df)
        )
        idx = REFERENCE_SIZE
        state["current_index"] = idx

    batch = df.iloc[idx: idx + batch_size].copy().reset_index(drop=True)
    batch["ingested_at"] = datetime.utcnow().isoformat()

    state["current_index"] = idx + len(batch)
    state["last_run_date"] = date.today().isoformat()
    state["batches_generated"] = state.get("batches_generated", 0) + 1
    _save_state(state)

    logger.info(
        "Batch #%d generated: %d rows (dataset rows %d–%d)",
        state["batches_generated"], len(batch), idx, idx + len(batch) - 1,
    )
    return batch


def get_stream_status() -> dict:
    """Return a progress summary of the simulated stream."""
    df = _load_dataset()
    state = _load_state()
    total  = len(df) - REFERENCE_SIZE
    consumed = state["current_index"] - REFERENCE_SIZE
    return {
        "total_stream_rows":  total,
        "consumed_rows":      consumed,
        "remaining_rows":     max(0, total - consumed),
        "batches_generated":  state.get("batches_generated", 0),
        "last_run_date":      state.get("last_run_date"),
    }


def reset_stream() -> None:
    """Reset the stream pointer to the beginning of the stream set."""
    _save_state({"current_index": REFERENCE_SIZE, "last_run_date": None, "batches_generated": 0})
    logger.info("Stream state reset to index %d.", REFERENCE_SIZE)
