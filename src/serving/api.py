"""
FastAPI serving layer.

Endpoints:
  GET  /health                      — Liveness check
  GET  /metrics                     — Current model evaluation metrics
  GET  /stream/status               — Data stream progress
  GET  /drift/status                — Latest drift detection result
  GET  /drift/report                — Interactive HTML drift report (Evidently)
  GET  /strata/summary              — Population risk stratum distribution
  POST /predict                     — Individual cancer risk prediction (supervised)
  POST /predict/batch               — Multi-patient batch prediction
  POST /screening-recommendation    — Full recommendation: risk + stratum + recall interval
  POST /pipeline/trigger            — Manually run one pipeline cycle (demo / ops)

Two-branch ML design:
  /predict                  → supervised branch only (XGBoost probability)
  /screening-recommendation → both branches combined:
                              supervised probability + unsupervised stratum
                              → screening interval policy output
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, List

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MODEL_METRICS_PATH, DRIFT_REPORT_PATH, API_HOST, API_PORT
from src.drift.drift_detector import get_current_status
from src.features.feature_engineer import engineer_features
from src.inference.predictor import predict

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cervical Cancer Risk Prediction API",
    description=(
        "Production-oriented ML pipeline for cervical cancer biopsy risk stratification.\n\n"
        "Built on the UCI Cervical Cancer Risk Factors dataset "
        "(Fernandes et al., 2017, Hospital Geral de Santo António, Porto).\n\n"
        "**Clinical note:** This tool provides risk stratification support only. "
        "All clinical decisions must be made by a qualified healthcare professional."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Request / Response models ──────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    """
    Feature set for a single patient.
    All fields except `age` have sensible zero-defaults so callers can
    supply only the fields they have.
    """
    age: float = Field(..., ge=0, le=120, description="Age in years")
    num_sexual_partners:           float = Field(default=0)
    first_sexual_intercourse:      float = Field(default=0)
    num_of_pregnancies:            float = Field(default=0)
    smokes:                        float = Field(default=0, ge=0, le=1)
    smokes_years:                  float = Field(default=0)
    smokes_packs_year:             float = Field(default=0)
    hormonal_contraceptives:       float = Field(default=0, ge=0, le=1)
    hormonal_contraceptives_years: float = Field(default=0)
    iud:                           float = Field(default=0, ge=0, le=1)
    iud_years:                     float = Field(default=0)
    stds:                          float = Field(default=0, ge=0, le=1)
    stds_number:                   float = Field(default=0)
    stds_condylomatosis:           float = Field(default=0)
    stds_cervical_condylomatosis:  float = Field(default=0)
    stds_vaginal_condylomatosis:   float = Field(default=0)
    stds_vulvo_perineal_condylomatosis: float = Field(default=0)
    stds_syphilis:                 float = Field(default=0)
    stds_pelvic_inflammatory_disease: float = Field(default=0)
    stds_genital_herpes:           float = Field(default=0)
    stds_molluscum_contagiosum:    float = Field(default=0)
    stds_aids:                     float = Field(default=0)
    stds_hiv:                      float = Field(default=0)
    stds_hepatitis_b:              float = Field(default=0)
    stds_hpv:                      float = Field(default=0)
    stds_num_diagnosis:            float = Field(default=0)
    stds_time_first_diagnosis:     float = Field(default=0)
    stds_time_last_diagnosis:      float = Field(default=0)
    dx_cancer:                     float = Field(default=0)
    dx_cin:                        float = Field(default=0)
    dx_hpv:                        float = Field(default=0)
    dx:                            float = Field(default=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 29,
                "smokes": 1,
                "smokes_years": 6,
                "hormonal_contraceptives": 1,
                "hormonal_contraceptives_years": 4,
                "stds": 1,
                "stds_hpv": 1,
                "num_sexual_partners": 5,
                "first_sexual_intercourse": 16,
                "num_of_pregnancies": 2,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction:   int
    probability:  float
    risk_level:   str
    top_features: List[dict]


class BatchRequest(BaseModel):
    patients: List[PatientFeatures]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _df_from_patient(patient: PatientFeatures):
    import pandas as pd
    row = patient.model_dump()
    df  = pd.DataFrame([row])
    return engineer_features(df)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/strata/summary", tags=["screening"])
async def strata_summary():
    """
    Return the current population risk stratum distribution.

    Shows how the full ingested patient population is distributed across
    low / moderate / high risk strata, with the screening policy for each.
    Useful for programme-level monitoring and capacity planning.
    """
    from src.clustering.risk_stratifier import get_stratum_summary
    return get_stratum_summary()


@app.post("/screening-recommendation", tags=["screening"])
async def screening_recommendation(patient: PatientFeatures):
    """
    Generate a complete screening recommendation for a single patient.

    Combines both ML branches:
      1. Supervised (XGBoost) → individual cancer probability + risk level
      2. Unsupervised (K-means) → population stratum assignment

    The recommendation integrates both signals:
      - A high individual probability overrides a low stratum → escalates to high
      - A low individual probability in a high stratum → keeps moderate recall

    This mirrors how a clinical decision support tool would present information:
    not just a raw probability, but a contextualised, actionable recommendation.

    Clinical note: all recommendations must be reviewed by a qualified clinician.
    """
    from src.clustering.risk_stratifier import assign_stratum

    df = _df_from_patient(patient)

    # Branch A: individual supervised prediction
    individual = predict(df)[0]

    # Branch B: population stratum assignment
    stratum_result = assign_stratum(df)[0]

    # Integration logic: escalate if individual risk disagrees with stratum
    ind_risk    = individual["risk_level"]
    pop_stratum = stratum_result["stratum"]

    risk_rank = {"low": 0, "moderate": 1, "high": 2}
    final_stratum = pop_stratum
    escalated     = False

    if risk_rank.get(ind_risk, 1) > risk_rank.get(pop_stratum, 1):
        # Individual signal is higher than population stratum — escalate
        final_stratum = ind_risk
        escalated = True

    policy = {
        "low":      {"recall_months": 36, "action": "routine_recall",     "label": "Routine 3-year recall"},
        "moderate": {"recall_months": 12, "action": "early_recall",        "label": "Early 12-month recall"},
        "high":     {"recall_months":  0, "action": "colposcopy_referral", "label": "Immediate colposcopy referral"},
    }[final_stratum]

    return {
        "individual_risk": {
            "prediction":  individual["prediction"],
            "probability": individual["probability"],
            "risk_level":  ind_risk,
            "top_features": individual["top_features"],
        },
        "population_stratum": {
            "stratum":        pop_stratum,
            "recall_months":  stratum_result["recall_months"],
        },
        "recommendation": {
            "final_stratum":   final_stratum,
            "recall_months":   policy["recall_months"],
            "action":          policy["action"],
            "label":           policy["label"],
            "escalated":       escalated,
            "escalation_note": (
                "Individual risk signal elevated above population stratum — "
                "recall interval shortened." if escalated else None
            ),
        },
        "disclaimer": (
            "This recommendation is generated by a research prototype. "
            "All clinical decisions must be made by a qualified healthcare professional."
        ),
    }


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "model_ready": MODEL_METRICS_PATH.exists()}


@app.get("/metrics", tags=["ops"])
async def metrics():
    if not MODEL_METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="No trained model found.")
    with open(MODEL_METRICS_PATH) as f:
        return json.load(f)


@app.get("/stream/status", tags=["ops"])
async def stream_status():
    from src.ingestion.data_generator import get_stream_status
    return get_stream_status()


@app.get("/drift/status", tags=["monitoring"])
async def drift_status():
    return get_current_status()


@app.get("/drift/report", tags=["monitoring"], response_class=HTMLResponse)
async def drift_report():
    if not DRIFT_REPORT_PATH.exists():
        return HTMLResponse(
            "<h2>No drift report available yet.</h2>"
            "<p>Run a pipeline cycle first: "
            "<code>POST /pipeline/trigger</code></p>"
        )
    return FileResponse(str(DRIFT_REPORT_PATH), media_type="text/html")


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_single(patient: PatientFeatures):
    """Predict biopsy risk for a single patient."""
    df      = _df_from_patient(patient)
    results = predict(df)
    return results[0]


@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(request: BatchRequest):
    """Predict biopsy risk for a list of patients in one call."""
    import pandas as pd
    rows = [p.model_dump() for p in request.patients]
    df   = pd.DataFrame(rows)
    df   = engineer_features(df)
    return {"predictions": predict(df), "count": len(rows)}


@app.post("/pipeline/trigger", tags=["ops"])
async def trigger_pipeline(background_tasks: BackgroundTasks):
    """
    Manually trigger one pipeline cycle (ingest → preprocess → drift → retrain).
    Executes in the background; returns immediately.
    """
    def _run():
        from src.pipeline.scheduler import run_pipeline_cycle
        run_pipeline_cycle()

    background_tasks.add_task(_run)
    return {
        "message": "Pipeline cycle started in background.",
        "check": "GET /drift/status for results once complete.",
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def start():
    """Start the API with the background daily scheduler."""
    import logging as _log
    _log.basicConfig(
        level=_log.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from src.pipeline.scheduler import start_scheduler
    t = threading.Thread(target=start_scheduler, daemon=True)
    t.start()
    logger.info("Background scheduler started (daily at %s UTC).", "06:00")

    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    start()
