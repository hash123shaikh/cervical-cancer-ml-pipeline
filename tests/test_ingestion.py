"""Tests for the ingestion layer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect DB to a temp path so tests don't touch the real database."""
    import config
    monkeypatch.setattr(config, "RAW_DB_PATH", tmp_path / "test.db")


def test_init_db_is_idempotent():
    from src.ingestion.ingestor import init_db
    init_db()
    init_db()  # calling twice must not raise


def test_ingest_and_retrieve():
    from src.ingestion.ingestor import init_db, ingest_batch, load_all_rows
    init_db()

    df = pd.DataFrame({
        "age":    [25.0, 32.0],
        "smokes": [0.0,  1.0],
        "biopsy": [0.0,  0.0],
    })
    ingestion_id = ingest_batch(df)

    assert isinstance(ingestion_id, str)
    assert len(ingestion_id) == 36            # UUID format

    result = load_all_rows()
    assert len(result) == 2
    assert "age" in result.columns


def test_empty_batch_returns_empty_string():
    from src.ingestion.ingestor import init_db, ingest_batch
    init_db()
    assert ingest_batch(pd.DataFrame()) == ""


def test_batch_history():
    from src.ingestion.ingestor import init_db, ingest_batch, get_batch_history
    init_db()
    ingest_batch(pd.DataFrame({"age": [20.0]}))
    history = get_batch_history()
    assert len(history) == 1
    assert "row_count" in history.columns
