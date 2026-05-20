import json
import logging
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import LOG_FILE, LOG_LEVEL, MODEL_METRICS_PATH, SCHEDULE_HOUR, SCHEDULE_MINUTE

logger = logging.getLogger(__name__)

def _setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.FileHandler(str(LOG_FILE)), logging.StreamHandler()], force=True)

def run_pipeline_cycle():
    started_at = datetime.utcnow().isoformat()
    logger.info("Pipeline cycle started — %s", started_at)
    from src.ingestion.data_generator import get_next_batch
    from src.ingestion.ingestor import ingest_batch, load_all_rows, init_db
    from src.preprocessing.preprocessor import preprocess, save_processed, save_stats
    from src.features.feature_engineer import engineer_features
    from src.drift.drift_detector import check_drift
    summary = {"started_at":started_at,"steps":{}}
    try:
        init_db()
        batch = get_next_batch()
        if batch is None or batch.empty:
            return {"started_at":started_at,"skipped":True}
        ingestion_id = ingest_batch(batch)
        summary["steps"]["ingestion"] = {"ingestion_id":ingestion_id,"rows":len(batch)}
        raw_df = load_all_rows()
        processed_df, stats = preprocess(raw_df, fit=True)
        save_stats(stats)
        featured_df = engineer_features(processed_df)
        save_processed(featured_df, split="train")
        summary["steps"]["feature_engineering"] = {"n_features":len(featured_df.columns)}
        batch_processed, _ = preprocess(batch, fit=False, stats=stats)
        batch_featured      = engineer_features(batch_processed)
        drift_status        = check_drift(batch_featured)
        summary["steps"]["drift_detection"] = drift_status
        should_retrain = drift_status["drift_detected"] or not MODEL_METRICS_PATH.exists()
        if should_retrain:
            from src.training.trainer import train
            from src.drift.drift_detector import save_reference
            metrics = train(featured_df)
            save_reference(featured_df)
            summary["steps"]["training"] = {**metrics,"reason":"drift detected" if drift_status["drift_detected"] else "no existing model"}
        else:
            summary["steps"]["training"] = "skipped"
        from src.clustering.risk_stratifier import fit_stratifier
        cluster_summary = fit_stratifier(featured_df)
        summary["steps"]["risk_stratification"] = cluster_summary
    except Exception as exc:
        logger.exception("Pipeline cycle failed: %s", exc)
        summary["error"] = str(exc)
    summary["completed_at"] = datetime.utcnow().isoformat()
    run_log = LOG_FILE.parent / "pipeline_runs.jsonl"
    with open(run_log,"a") as f: f.write(json.dumps(summary)+"\n")
    return summary

def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    _setup_logging()
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(run_pipeline_cycle, trigger=CronTrigger(hour=SCHEDULE_HOUR,minute=SCHEDULE_MINUTE),
        id="daily_pipeline", misfire_grace_time=3600, replace_existing=True)
    logger.info("Scheduler running — pipeline fires daily at %02d:%02d UTC.", SCHEDULE_HOUR, SCHEDULE_MINUTE)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
