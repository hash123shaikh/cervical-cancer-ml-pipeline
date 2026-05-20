import json
import logging
import threading
from pathlib import Path
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
app = FastAPI(title="Cervical Cancer Risk Prediction API", version="1.0.0")

class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    num_sexual_partners: float = Field(default=0)
    first_sexual_intercourse: float = Field(default=0)
    num_of_pregnancies: float = Field(default=0)
    smokes: float = Field(default=0, ge=0, le=1)
    smokes_years: float = Field(default=0)
    smokes_packs_year: float = Field(default=0)
    hormonal_contraceptives: float = Field(default=0, ge=0, le=1)
    hormonal_contraceptives_years: float = Field(default=0)
    iud: float = Field(default=0, ge=0, le=1)
    iud_years: float = Field(default=0)
    stds: float = Field(default=0, ge=0, le=1)
    stds_number: float = Field(default=0)
    stds_condylomatosis: float = Field(default=0)
    stds_cervical_condylomatosis: float = Field(default=0)
    stds_vaginal_condylomatosis: float = Field(default=0)
    stds_vulvo_perineal_condylomatosis: float = Field(default=0)
    stds_syphilis: float = Field(default=0)
    stds_pelvic_inflammatory_disease: float = Field(default=0)
    stds_genital_herpes: float = Field(default=0)
    stds_molluscum_contagiosum: float = Field(default=0)
    stds_aids: float = Field(default=0)
    stds_hiv: float = Field(default=0)
    stds_hepatitis_b: float = Field(default=0)
    stds_hpv: float = Field(default=0)
    stds_num_diagnosis: float = Field(default=0)
    stds_time_first_diagnosis: float = Field(default=0)
    stds_time_last_diagnosis: float = Field(default=0)
    dx_cancer: float = Field(default=0)
    dx_cin: float = Field(default=0)
    dx_hpv: float = Field(default=0)
    dx: float = Field(default=0)

def _df_from_patient(patient):
    import pandas as pd
    return engineer_features(pd.DataFrame([patient.model_dump()]))

@app.get("/health", tags=["ops"])
async def health():
    return {"status":"ok","model_ready":MODEL_METRICS_PATH.exists()}

@app.get("/metrics", tags=["ops"])
async def metrics():
    if not MODEL_METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="No trained model found.")
    with open(MODEL_METRICS_PATH) as f: return json.load(f)

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
        return HTMLResponse("<h2>No drift report yet.</h2><p>POST /pipeline/trigger first.</p>")
    return FileResponse(str(DRIFT_REPORT_PATH), media_type="text/html")

@app.get("/strata/summary", tags=["screening"])
async def strata_summary():
    from src.clustering.risk_stratifier import get_stratum_summary
    return get_stratum_summary()

@app.post("/predict", tags=["prediction"])
async def predict_single(patient: PatientFeatures):
    return predict(_df_from_patient(patient))[0]

@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(request: dict):
    import pandas as pd
    patients = request.get("patients", [])
    if not patients: return {"predictions":[],"count":0}
    df = engineer_features(pd.DataFrame(patients))
    return {"predictions":predict(df),"count":len(patients)}

@app.post("/screening-recommendation", tags=["screening"])
async def screening_recommendation(patient: PatientFeatures):
    from src.clustering.risk_stratifier import assign_stratum
    df = _df_from_patient(patient)
    individual     = predict(df)[0]
    stratum_result = assign_stratum(df)[0]
    risk_rank  = {"low":0,"moderate":1,"high":2}
    ind_risk   = individual["risk_level"]
    pop_stratum = stratum_result["stratum"]
    final_stratum = ind_risk if risk_rank.get(ind_risk,1) > risk_rank.get(pop_stratum,1) else pop_stratum
    escalated = final_stratum != pop_stratum
    policy = {"low":{"recall_months":36,"action":"routine_recall","label":"Routine 3-year recall"},
              "moderate":{"recall_months":12,"action":"early_recall","label":"Early 12-month recall"},
              "high":{"recall_months":0,"action":"colposcopy_referral","label":"Immediate colposcopy referral"}}[final_stratum]
    return {"individual_risk":individual,"population_stratum":stratum_result,
            "recommendation":{**policy,"final_stratum":final_stratum,"escalated":escalated},
            "disclaimer":"Research prototype. All clinical decisions must be made by a qualified clinician."}

@app.post("/pipeline/trigger", tags=["ops"])
async def trigger_pipeline(background_tasks: BackgroundTasks):
    def _run():
        from src.pipeline.scheduler import run_pipeline_cycle
        run_pipeline_cycle()
    background_tasks.add_task(_run)
    return {"message":"Pipeline cycle started in background.","check":"GET /drift/status for results."}

def start():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    from src.pipeline.scheduler import start_scheduler
    t = threading.Thread(target=start_scheduler, daemon=True)
    t.start()
    uvicorn.run(app, host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    start()
