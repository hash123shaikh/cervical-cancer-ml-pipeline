"""
Manually trigger one pipeline cycle (ingest → preprocess → drift → retrain).
Useful for demos, testing, and operational verification.

Usage:
    python scripts/run_pipeline.py
"""

import sys
import json
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from src.pipeline.scheduler import run_pipeline_cycle

if __name__ == "__main__":
    summary = run_pipeline_cycle()
    print("\n── Pipeline summary ──────────────────────────")
    print(json.dumps(summary, indent=2))
