"""
Pipeline Scheduler — orchestrates the complete daily ML cycle.

Cycle steps (runs daily at SCHEDULE_HOUR:SCHEDULE_MINUTE UTC):
  1. Generate next daily batch from the stream
  2. Ingest batch → SQLite
  3. Load all raw rows from SQLite
  4. Preprocess (impute, clip)
  5. Feature engineer
  6. Save processed CSV
  7. Run drift detection on the incoming batch
  8. If drift detected (or no model exists yet) → retrain, update reference
  9. Log a structured JSONL summary to logs/pipeline_runs.jsonl

Why APScheduler (not Airflow / Prefect)?
─────────────────────────────────────────
This system runs as a single Docker container on a single node. APScheduler
runs in-process with zero extra infrastructure (no broker, no worker fleet).
For a production system with multi-step dependencies, SLAs, and distributed
execution, a DAG-based orchestrator (Airflow, Prefect, or a cloud scheduler)
would be the right replacement. APScheduler is the pragmatic choice here.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import LOG_FILE, LOG_LEVEL, MODEL_METRICS_PATH, SCHEDULE_HOUR, SCHEDULE_MINUTE

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(str(LOG_FILE)),
            logging.StreamHandler(),
        ],
        force=True,
    )


def run_pipeline_cycle() -> dict:
    """
    Execute one full pipeline cycle.
    Safe to call manually (e.g. via POST /pipeline/trigger or scripts/run_pipeline.py).
    Returns a structured summary dict.
    """
    started_at = datetime.utcnow().isoformat()
    logger.info("══════════════════════════════════════════════")
    logger.info("Pipeline cycle started — %s", started_at)

    from src.ingestion.data_generator import get_next_batch
    from src.ingestion.ingestor import ingest_batch, load_all_rows, init_db
    from src.preprocessing.preprocessor import preprocess, save_processed, save_stats
    from src.features.feature_engineer import engineer_features
    from src.drift.drift_detector import check_drift

    summary: dict = {"started_at": started_at, "steps": {}}

    try:
        # ── 1: Ensure DB exists ────────────────────────────────────────────────
        init_db()

        # ── 2: Ingest next batch ───────────────────────────────────────────────
        batch = get_next_batch()
        if batch is None or batch.empty:
            logger.warning("No new data available — skipping cycle.")
            return {"started_at": started_at, "skipped": True}

        ingestion_id = ingest_batch(batch)
        summary["steps"]["ingestion"] = {"ingestion_id": ingestion_id, "rows": len(batch)}
        logger.info("✓ Step 1/5 — Ingested %d rows (id=%s)", len(batch), ingestion_id[:8])

        # ── 3: Load all raw data + preprocess ─────────────────────────────────
        raw_df = load_all_rows()
        processed_df, stats = preprocess(raw_df, fit=True)
        save_stats(stats)
        summary["steps"]["preprocessing"] = {"total_rows": len(processed_df)}
        logger.info("✓ Step 2/5 — Preprocessed %d total rows", len(processed_df))

        # ── 4: Feature engineering ─────────────────────────────────────────────
        featured_df = engineer_features(processed_df)
        save_processed(featured_df, split="train")
        summary["steps"]["feature_engineering"] = {"n_features": len(featured_df.columns)}
        logger.info("✓ Step 3/5 — Feature engineering (%d features)", len(featured_df.columns))

        # ── 5: Drift detection on the incoming batch ───────────────────────────
        batch_processed, _ = preprocess(batch, fit=False, stats=stats)
        batch_featured      = engineer_features(batch_processed)
        drift_status        = check_drift(batch_featured)
        summary["steps"]["drift_detection"] = drift_status
        logger.info(
            "✓ Step 4/5 — Drift check: detected=%s (%.0f%% columns shifted)",
            drift_status["drift_detected"],
            100 * drift_status.get("share_drifted", 0),
        )

        # ── 6: Conditional supervised retraining ──────────────────────────────
        should_retrain = drift_status["drift_detected"] or not MODEL_METRICS_PATH.exists()
        if should_retrain:
            reason = "drift detected" if drift_status["drift_detected"] else "no existing model"
            logger.info("↻ Step 5/6 — Retraining supervised model (%s)…", reason)
            from src.training.trainer import train
            from src.drift.drift_detector import save_reference
            metrics = train(featured_df)
            save_reference(featured_df)
            summary["steps"]["training"] = {**metrics, "reason": reason}
            logger.info(
                "✓ Step 5/6 — Retrained. CV AUC: %.3f", metrics["cv_auc_mean"]
            )
        else:
            logger.info("✓ Step 5/6 — No supervised retraining needed.")
            summary["steps"]["training"] = "skipped"

        # ── 7: Refit risk stratifier (unsupervised) ────────────────────────────
        # Refitted on every cycle — not just on drift — because the full
        # population distribution shifts as new batches accumulate over time.
        logger.info("↻ Step 6/6 — Refitting risk stratifier (K-means)…")
        from src.clustering.risk_stratifier import fit_stratifier
        cluster_summary = fit_stratifier(featured_df)
        summary["steps"]["risk_stratification"] = cluster_summary
        logger.info(
            "✓ Step 6/6 — Stratifier fitted: k=%d | silhouette=%.3f",
            cluster_summary["k"],
            cluster_summary["silhouette_score"],
        )

    except Exception as exc:
        logger.exception("Pipeline cycle failed: %s", exc)
        summary["error"] = str(exc)

    summary["completed_at"] = datetime.utcnow().isoformat()
    logger.info("Pipeline cycle completed — %s", summary["completed_at"])
    logger.info("══════════════════════════════════════════════")

    # Append structured run log
    run_log = LOG_FILE.parent / "pipeline_runs.jsonl"
    with open(run_log, "a") as f:
        f.write(json.dumps(summary) + "\n")

    return summary


def start_scheduler() -> None:
    """Start the APScheduler cron job. Blocks until interrupted."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    _setup_logging()

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_pipeline_cycle,
        trigger=CronTrigger(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE),
        id="daily_pipeline",
        name="Daily ML pipeline cycle",
        misfire_grace_time=3600,  # run up to 1 h late if the container was down
        replace_existing=True,
    )

    logger.info(
        "Scheduler running — pipeline fires daily at %02d:%02d UTC.",
        SCHEDULE_HOUR, SCHEDULE_MINUTE,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
