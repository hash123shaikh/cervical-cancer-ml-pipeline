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

def _connect():
    RAW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(RAW_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    conn = _connect()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_BATCHES_TABLE} (
            ingestion_id TEXT PRIMARY KEY, ingested_at TEXT NOT NULL,
            batch_date TEXT NOT NULL, row_count INTEGER NOT NULL, json_payload TEXT NOT NULL)""")
    conn.commit(); conn.close()
    logger.info("Database initialised at %s", RAW_DB_PATH)

def ingest_batch(df):
    if df.empty:
        logger.warning("Empty batch — nothing to ingest.")
        return ""
    ingestion_id = str(uuid.uuid4())
    ingested_at  = datetime.utcnow().isoformat()
    batch_date   = datetime.utcnow().date().isoformat()
    conn = _connect()
    conn.execute(f"INSERT INTO {_BATCHES_TABLE} (ingestion_id,ingested_at,batch_date,row_count,json_payload) VALUES (?,?,?,?,?)",
                 (ingestion_id, ingested_at, batch_date, len(df), df.to_json(orient="records")))
    conn.commit()
    rows = df.drop(columns=["ingested_at"], errors="ignore").copy()
    rows["ingestion_id"] = ingestion_id
    rows["ingested_at"]  = ingested_at
    rows.to_sql(_ROWS_TABLE, conn, if_exists="append", index=False)
    conn.commit(); conn.close()
    logger.info("Ingested batch %s: %d rows", ingestion_id, len(df))
    return ingestion_id

def load_all_rows():
    conn = _connect()
    df = pd.read_sql(f"SELECT * FROM {_ROWS_TABLE} ORDER BY ingested_at ASC", conn)
    conn.close()
    return df.drop(columns=["id","ingestion_id","ingested_at"], errors="ignore")

def get_batch_history():
    conn = _connect()
    df = pd.read_sql(f"SELECT ingestion_id,ingested_at,batch_date,row_count FROM {_BATCHES_TABLE} ORDER BY ingested_at DESC", conn)
    conn.close()
    return df
