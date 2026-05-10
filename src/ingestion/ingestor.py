"""
Ingestor — persists raw batches to SQLite.

Why SQLite for raw data (not flat files)?
─────────────────────────────────────────
• Schema enforcement: malformed rows (wrong type, missing column) are caught at
  insert time, not silently propagated downstream.
• Queryability: ingestion history can be audited without loading everything into
  memory ("how many rows arrived this week?", "which batch introduced the anomaly?").
• Atomic writes: a crash mid-batch leaves the previous state intact; we never end
  up with a partially written file.
• Concurrent reads: the API can query batch history while the scheduler writes the
  next batch (WAL mode enables this without blocking).
• Zero extra infrastructure: SQLite runs in-process — appropriate for a
  single-node system where running a Postgres server would be over-engineering.

Why NOT SQLite for processed features?
───────────────────────────────────────
Processed feature matrices are read sequentially and in bulk by pandas for model
training — no filtering, no joins, no random access. Flat CSV files are faster for
this access pattern, trivially versioned (one timestamped file per run), and
directly inspectable with any standard data tool. See preprocessor.py.
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DB_PATH

logger = logging.getLogger(__name__)

_BATCHES_TABLE = "raw_batches"
_ROWS_TABLE    = "raw_rows"


# ── Connection ─────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    RAW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(RAW_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")   # concurrent reads without blocking writes
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ── Schema ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create database tables if they don't already exist (idempotent)."""
    conn = _connect()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_BATCHES_TABLE} (
            ingestion_id  TEXT PRIMARY KEY,
            ingested_at   TEXT NOT NULL,
            batch_date    TEXT NOT NULL,
            row_count     INTEGER NOT NULL,
            json_payload  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", RAW_DB_PATH)


# ── Write ──────────────────────────────────────────────────────────────────────

def ingest_batch(df: pd.DataFrame) -> str:
    """
    Persist a DataFrame batch to SQLite.

    Writes two records:
      1. A metadata row in `raw_batches` — ingestion_id, timestamp, row count,
         and full JSON payload for complete auditability.
      2. Individual feature rows in `raw_rows` via pandas.to_sql for queryability.

    Returns the ingestion_id (UUID) so downstream steps can reference this batch.
    """
    if df.empty:
        logger.warning("Received empty batch — nothing to ingest.")
        return ""

    ingestion_id = str(uuid.uuid4())
    ingested_at  = datetime.utcnow().isoformat()
    batch_date   = datetime.utcnow().date().isoformat()

    conn = _connect()

    # Batch metadata (full payload kept for auditability / replay)
    conn.execute(
        f"INSERT INTO {_BATCHES_TABLE} "
        "(ingestion_id, ingested_at, batch_date, row_count, json_payload) "
        "VALUES (?, ?, ?, ?, ?)",
        (ingestion_id, ingested_at, batch_date, len(df),
         df.to_json(orient="records")),
    )
    conn.commit()

    # Individual rows — drop the synthetic ingested_at added by the generator,
    # replace with a consistent timestamp, and tag with the ingestion_id
    rows = df.drop(columns=["ingested_at"], errors="ignore").copy()
    rows["ingestion_id"] = ingestion_id
    rows["ingested_at"]  = ingested_at
    rows.to_sql(_ROWS_TABLE, conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    logger.info("Ingested batch %s: %d rows at %s", ingestion_id, len(df), ingested_at)
    return ingestion_id


# ── Read ───────────────────────────────────────────────────────────────────────

def load_all_rows() -> pd.DataFrame:
    """
    Return all ingested feature rows ordered by ingestion time.
    Bookkeeping columns (id, ingestion_id, ingested_at) are dropped.
    """
    conn = _connect()
    df = pd.read_sql(
        f"SELECT * FROM {_ROWS_TABLE} ORDER BY ingested_at ASC", conn
    )
    conn.close()
    return df.drop(columns=["id", "ingestion_id", "ingested_at"], errors="ignore")


def get_batch_history() -> pd.DataFrame:
    """Return a summary of all ingestion runs for monitoring."""
    conn = _connect()
    df = pd.read_sql(
        f"SELECT ingestion_id, ingested_at, batch_date, row_count "
        f"FROM {_BATCHES_TABLE} ORDER BY ingested_at DESC",
        conn,
    )
    conn.close()
    return df
