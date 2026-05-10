#!/bin/bash
# entrypoint.sh — Bootstrap on first run, then start the API.
#
# On first `docker-compose up`:
#   1. Downloads the UCI dataset
#   2. Runs initial training on the reference set
#   3. Starts the FastAPI server (which also launches the background scheduler)
#
# On subsequent runs the model already exists (persisted via Docker volume),
# so steps 1–2 are skipped and the server starts immediately.

set -e

echo "══════════════════════════════════════════════════"
echo "  Cervical Cancer ML Pipeline"
echo "══════════════════════════════════════════════════"

if [ ! -f "models/xgboost_model.joblib" ]; then
    echo ""
    echo "  First run detected — bootstrapping pipeline..."
    echo ""
    python scripts/download_data.py
    python scripts/initial_train.py
    echo ""
    echo "  Bootstrap complete. Starting API server."
    echo ""
else
    echo ""
    echo "  Model found — starting API server."
    echo ""
fi

exec python src/serving/api.py
