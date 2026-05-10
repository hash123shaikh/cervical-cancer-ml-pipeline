"""
Bootstrap the pipeline on the historical reference dataset.

Run this once before starting the API server:
  python scripts/initial_train.py

Steps:
  1. Download dataset (if needed)
  2. Initialise SQLite database
  3. Ingest reference set (first 600 rows)
  4. Preprocess + feature engineer
  5. Train XGBoost model
  6. Save reference distribution for drift detection
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("initial_train")


def main() -> None:
    import scripts.download_data as dl
    dl.download()

    from src.ingestion.ingestor         import init_db, ingest_batch
    from src.ingestion.data_generator   import get_reference_data
    from src.preprocessing.preprocessor import preprocess, save_processed, save_stats
    from src.features.feature_engineer  import engineer_features
    from src.training.trainer           import train
    from src.drift.drift_detector       import save_reference

    logger.info("─── 1/5  Initialising database")
    init_db()

    logger.info("─── 2/5  Loading and ingesting reference data")
    ref_df = get_reference_data()
    ingest_batch(ref_df)
    logger.info("      %d reference rows ingested.", len(ref_df))

    logger.info("─── 3/5  Preprocessing")
    processed_df, stats = preprocess(ref_df, fit=True)
    save_stats(stats)

    logger.info("─── 4/5  Feature engineering")
    featured_df = engineer_features(processed_df)
    save_processed(featured_df, split="train")
    logger.info(
        "      %d rows × %d features saved.",
        len(featured_df), len(featured_df.columns),
    )

    logger.info("─── 5/6  Training model + saving drift reference")
    metrics = train(featured_df)
    save_reference(featured_df)

    logger.info("─── 6/6  Fitting risk stratifier (K-means)")
    from src.clustering.risk_stratifier import fit_stratifier
    cluster_summary = fit_stratifier(featured_df)

    logger.info("══════════════════════════════════════════")
    logger.info("Bootstrap complete!")
    logger.info("  CV AUC :  %.3f ± %.3f", metrics["cv_auc_mean"], metrics["cv_auc_std"])
    logger.info("  Train AUC: %.3f", metrics["train_auc"])
    logger.info("  Train F1:  %.3f", metrics["train_f1"])
    logger.info("  Risk strata (k=%d): %s", cluster_summary["k"],
                {s: v["n_patients"] for s, v in cluster_summary["stratum_distribution"].items()})
    logger.info("")
    logger.info("Start the server:  python src/serving/api.py")
    logger.info("Or with Docker:    docker-compose up")


if __name__ == "__main__":
    main()
