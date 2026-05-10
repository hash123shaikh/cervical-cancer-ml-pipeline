# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches the pip install step.
# Re-runs only when requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ────────────────────────────────────────────────────────────────
COPY . .

# Create the directory tree expected at runtime.
# These will be shadowed by the Docker volumes defined in docker-compose.yml,
# so artefacts written here persist on the host across restarts.
RUN mkdir -p data/raw data/processed models logs reports

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app

RUN chmod +x entrypoint.sh

# ── Networking ─────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
# start_period is generous: first-run bootstrap (download + train) can take
# 60–120 s before the server is ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=150s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entry point ────────────────────────────────────────────────────────────────
# entrypoint.sh bootstraps the model if needed, then exec's the API server.
ENTRYPOINT ["./entrypoint.sh"]
