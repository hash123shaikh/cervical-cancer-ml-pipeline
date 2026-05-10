"""Download the UCI Cervical Cancer Risk Factors dataset."""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from config import DATA_URL, RAW_CSV_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def download() -> None:
    if RAW_CSV_PATH.exists():
        logger.info("Dataset already at %s — skipping.", RAW_CSV_PATH)
        return

    logger.info("Downloading from %s …", DATA_URL)
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    RAW_CSV_PATH.write_bytes(resp.content)

    import pandas as pd
    df = pd.read_csv(RAW_CSV_PATH)
    logger.info(
        "Downloaded %d rows × %d columns → %s",
        len(df), len(df.columns), RAW_CSV_PATH,
    )


if __name__ == "__main__":
    download()
