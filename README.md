# Cervical Cancer Risk-Based Screening — End-to-End ML Pipeline

---------------------------
**Step 1 — Initialise Git and make your first commit**

git init
git add .
git commit -m "Initial commit: end-to-end ML pipeline for cervical cancer risk screening"

**Step 2 — Connect to GitHub and push**
Copy the two lines under "…or push an existing repository from the command line".

git remote add origin https://github.com/YOUR_USERNAME/cervical-cancer-ml-pipeline.git
git branch -M main
git push -u origin main

*GitHub will ask for your username and password. Important: GitHub no longer accepts your account password here. You need a Personal Access Token instead.*

*Username:* hash123shaikh
*Password:* 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

------------------------

A production-oriented machine learning system for cervical cancer risk-based
screening stratification, combining a **supervised individual risk model** and
an **unsupervised population stratification model** to generate actionable
screening interval recommendations.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Design Decisions](#3-design-decisions)
4. [Quick Start](#4-quick-start)
5. [API Reference](#5-api-reference)
6. [Data Drift Handling](#6-data-drift-handling)
7. [Running Tests](#7-running-tests)
8. [Code Structure](#8-code-structure)
9. [Project Structure](#9-project-structure)
10. [Extension: National Registry Scale](#10-extension-national-registry-scale)

---

## 1. Overview

### Dataset

The **UCI Cervical Cancer (Risk Factors) dataset** (Fernandes et al., 2017,
Hospital Geral de Santo António, Porto) was chosen because it directly reflects
a real clinical screening context: heterogeneous tabular features, substantial
missingness (~30% in some columns), a low positive rate (~6%), and
multiple screening test outcomes — characteristics shared with national
registry-scale screening data.

- **858 patients**, 36 features (demographic, behavioural, STD history, screening test results)
- **Target**: `Biopsy` (binary, gold-standard diagnostic label)
- **Positive rate**: ~6% (clinically realistic class imbalance)
- **Missing values**: encoded as `"?"` (UCI convention)
- **Screening test results**: `Hinselmann`, `Schiller`, `Citology` — each
  representing a different screening modality, analogous to HPV genotyping +
  cytology in a national programme

### Two-branch ML design

This system deliberately implements **both** supervised and unsupervised ML,
mirroring the design of a real risk-based screening programme:

| Branch | Method | Question answered |
|---|---|---|
| **Supervised** | XGBoost | What is THIS patient's individual cancer probability? |
| **Unsupervised** | K-means | Which risk stratum does this patient belong to in the POPULATION? |
| **Combined** | Integration logic | What is the recommended screening interval? |

The two outputs are fused in the `/screening-recommendation` endpoint to produce
a policy-level output: routine recall (3 years), early recall (12 months), or
immediate colposcopy referral.

### What this system demonstrates

| Concern | Implementation |
|---|---|
| Supervised individual risk | XGBoost + SHAP attributions |
| Unsupervised population stratification | K-means with silhouette-optimised k |
| Screening interval recommendation | Integration of both ML branches |
| Realistic data simulation | Daily batch release from a held-out stream set |
| Auditability | SQLite raw-data store with ingestion provenance |
| Reproducibility | Pinned requirements, Docker, versioned CSVs |
| Monitoring | Evidently AI drift reports per batch |
| Explainability | SHAP feature attributions on every prediction |
| Operability | Health check, metrics endpoint, manual pipeline trigger |

---

## 2. Architecture

```
                    UCI Cervical Cancer Dataset (858 patients)
                                      │
                    ┌─────────────────┴──────────────────┐
                    │                                    │
              rows 0–599                          rows 600–857
           (reference set)                       (stream set)
                    │                                    │
                    └──────── initial_train.py ──────────┘
                                      │
                                      ▼
          ┌───────────────────────────────────────────────────────┐
          │            DAILY PIPELINE CYCLE                       │
          │            APScheduler  ·  06:00 UTC                  │
          │                                                       │
          │  ┌──────────────┐     ┌──────────────┐               │
          │  │  Data        │     │ Preprocessor │               │
          │  │  Generator   │────▶│  • NaN impute│               │
          │  │  (+15 rows)  │     │  • clip P99  │               │
          │  └──────────────┘     └──────┬───────┘               │
          │         │                    │                        │
          │         ▼                    ▼                        │
          │    ┌──────────┐     ┌────────────────────┐           │
          │    │  SQLite  │     │  Feature Engineer  │           │
          │    │ raw_rows │     │  • stds_total      │           │
          │    └──────────┘     │  • is_smoker       │           │
          │                     │  • age_group       │           │
          │                     │  • high_risk_stds  │           │
          │                     └────────┬───────────┘           │
          │                              │                        │
          │              ┌───────────────┴───────────────┐        │
          │              │                               │        │
          │              ▼                               ▼        │
          │   ┌─────────────────────┐   ┌──────────────────────┐ │
          │   │  BRANCH A           │   │  BRANCH B            │ │
          │   │  Supervised         │   │  Unsupervised        │ │
          │   │                     │   │                      │ │
          │   │  XGBoost Classifier │   │  K-means Clustering  │ │
          │   │  • 5-fold CV        │   │  • k selected by     │ │
          │   │  • scale_pos_weight │   │    silhouette score  │ │
          │   │  • SHAP attributions│   │  • strata labelled   │ │
          │   │                     │   │    by biopsy rate    │ │
          │   │  Output:            │   │                      │ │
          │   │  P(cancer)          │   │  Output:             │ │
          │   │  per patient        │   │  low / moderate /    │ │
          │   └──────────┬──────────┘   │  high risk stratum   │ │
          │              │              └──────────┬───────────┘ │
          │              │                         │             │
          │              └────────────┬────────────┘             │
          │                           │                           │
          │                           ▼                           │
          │              ┌────────────────────────┐              │
          │              │  Drift Detector        │              │
          │              │  (Evidently AI)        │              │
          │              │  → retrain if drift    │              │
          │              └────────────────────────┘              │
          └───────────────────────────────────────────────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │      FastAPI  :8000        │
                        │                            │
                        │  POST /predict             │ ← supervised only
                        │  POST /screening-          │ ← BOTH branches fused
                        │       recommendation       │
                        │  GET  /strata/summary      │ ← population view
                        │  GET  /drift/status        │
                        │  GET  /drift/report        │
                        │  GET  /metrics             │
                        │  GET  /health              │
                        │  POST /pipeline/trigger    │
                        └────────────────────────────┘
```

---

## 3. Design Decisions

### 3.1 Daily batch cadence (not real-time)

In a cervical cancer screening programme, data does not arrive as a continuous
stream. Biopsy results require 2–5 days of lab processing before entry into a
clinical information system. Screening programmes process patients in cohort
batches (by clinic, by region, by referral date). A daily aggregation cadence
accurately reflects this data latency.

Real-time streaming would be appropriate for an on-device triage tool operating
during a colposcopy appointment. For a population-level risk model that learns
from completed screening episodes, daily batch is the right fit.

### 3.2 Two storage layers (SQLite + CSV)

| Layer | Store | Why |
|---|---|---|
| Raw ingested data | **SQLite** | Schema enforcement, atomic writes, queryable audit log, concurrent reads (WAL mode), zero extra infrastructure |
| Processed features | **CSV flat files** | Bulk sequential reads are faster than SQL; trivially versioned (one timestamped file per run); directly inspectable in pandas/Excel |

SQLite is used where we need auditability and structured queries.
CSV is used where we need bulk read throughput and human-inspectable artefacts.

### 3.3 Why XGBoost

- **Small dataset performance**: with ~600 training rows, gradient-boosted trees
  consistently outperform neural networks. Deep models require far more data to
  avoid overfitting; XGBoost regularisation (L1/L2) handles it natively.
- **Class imbalance**: `scale_pos_weight` (ratio of negatives to positives)
  corrects the ~6% positive rate without oversampling or SMOTE, which can
  introduce artefacts on a small dataset.
- **Explainability**: Tree SHAP produces fast, exact feature attributions — a
  non-negotiable property for any tool that influences clinical decision-making.
- **Auditability**: the serialised model (`joblib`) is a single file that can
  be versioned, diffed, and loaded without a training framework.

### 3.4 Why both supervised AND unsupervised learning

A supervised model alone answers the wrong question for a screening programme.

Giving a clinician a probability score of 0.43 is not actionable. What a screening programme needs is a **policy decision**: recall this patient in 12 months, or 3 years, or refer immediately. That requires mapping individuals onto a discrete set of strata — which is precisely what unsupervised clustering produces.

The supervised and unsupervised branches answer fundamentally different questions:

| Branch | Question | Output |
|---|---|---|
| XGBoost (supervised) | What is this individual's cancer probability? | 0.0 – 1.0 continuous score |
| K-means (unsupervised) | Which population risk group does this patient belong to? | low / moderate / high stratum |
| Combined | What should we do? | Screening interval + action |

The two are deliberately kept separate in the codebase (`src/training/` vs `src/clustering/`) because they are fitted differently — the supervised model retrains only on drift, while the stratifier retrains on every cycle because the population distribution shifts as new patients accumulate, even when the latest batch does not show significant drift.

### 3.5 Why K-means for stratification

- **Scalability**: K-means runs in linear time with respect to dataset size. At national registry scale (millions of patients), hierarchical and density-based methods are computationally infeasible.
- **Interpretability**: Each cluster is defined by a centroid in feature space, which can be inspected by clinicians and epidemiologists. The mean feature values per stratum translate directly into clinical profile descriptions.
- **Assignability**: New patients are assigned to a stratum with a single distance computation. No retraining is needed.
- **Precedent**: K-means-based risk stratification is established in population health literature (e.g., ACG patient risk groups, Johns Hopkins ACG system).

k is selected automatically using silhouette score across k=2..5, defaulting to k=3 to preserve the clinically natural three-tier structure (routine / early / urgent).

### 3.6 In-process scheduler (APScheduler)

APScheduler runs inside the same Python process as the FastAPI server —
no separate worker, no message broker. This is the correct choice for a
single-node system. If the pipeline were to grow (multi-step DAGs, SLAs,
distributed execution), the natural replacement is Apache Airflow or Prefect,
with the same `run_pipeline_cycle()` function becoming an Airflow task.

---

## 4. Quick Start

### Option A — Docker (recommended)

```bash
# Clone and enter
git clone <repo-url>
cd cervical-cancer-ml

# First run: builds image, downloads UCI dataset, trains model, starts API
docker-compose up
```

The first run downloads the dataset (~150 KB) and trains the model inside
the container. Subsequent `docker-compose up` calls skip training because the
model is persisted via the mounted `./models` volume.

API is available at **http://localhost:8000** · Docs at **http://localhost:8000/docs**

### Option B — Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the Makefile
make setup

# Download data + train
make train

# Start the API server (includes background scheduler)
make run
```

### Trigger a pipeline cycle manually

```bash
# Via script (no API needed)
make pipeline

# Via API (runs in background)
curl -X POST http://localhost:8000/pipeline/trigger
```

---

## 5. API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `GET` | `/metrics` | Current supervised model metrics (CV AUC, Train F1, etc.) |
| `GET` | `/stream/status` | Simulated stream progress |
| `GET` | `/drift/status` | Latest drift detection result |
| `GET` | `/drift/report` | Interactive Evidently HTML report |
| `GET` | `/strata/summary` | Population risk stratum distribution + policy per stratum |
| `POST` | `/predict` | Individual cancer risk prediction (supervised branch only) |
| `POST` | `/predict/batch` | Multi-patient batch prediction |
| `POST` | `/screening-recommendation` | **Full recommendation: both branches fused** |
| `POST` | `/pipeline/trigger` | Manually run one pipeline cycle |

Full interactive docs: **http://localhost:8000/docs** (Swagger UI)

### Prediction request / response

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 29,
    "smokes": 1,
    "smokes_years": 6,
    "stds_hpv": 1,
    "hormonal_contraceptives": 1,
    "num_sexual_partners": 5
  }'
```

```json
{
  "prediction":  1,
  "probability": 0.7231,
  "risk_level":  "high",
  "top_features": [
    {"feature": "stds_hpv",    "shap_value":  0.42310},
    {"feature": "stds_total",  "shap_value":  0.31204},
    {"feature": "age_group",   "shap_value": -0.11008},
    {"feature": "is_smoker",   "shap_value":  0.08843},
    {"feature": "smokes_years","shap_value":  0.06102}
  ]
}
```

Risk tiers: `low` (< 30%) · `moderate` (30–60%) · `high` (> 60%)

---

## 6. Data Drift Handling

### Detection

Each daily batch is compared against the reference distribution (first 600 rows)
using **Evidently AI's DataDriftPreset**:

- **Wasserstein distance** for continuous features (age, smokes_years, …)
- **Jensen-Shannon divergence** for binary features (stds flags, iud, …)

These are distribution-free tests — no normality assumption required.

### Decision rule

If **> 15% of feature columns** show drift → flag retraining.

15% is intentionally conservative. Cervical cancer screening data shifts slowly
(population demographics and screening guidelines evolve over years). With only
15 rows per day, a more aggressive threshold would generate noisy false alarms
from sampling variance rather than real distribution change.

### Response to detected drift

1. Full HTML report saved to `reports/drift_report.html`
2. Model retrained on all accumulated data in the same pipeline cycle
3. New reference statistics updated with the current distribution

### Acknowledged limitations

- We monitor **covariate shift** (feature distributions), not **concept drift**
  (label distribution). A complete production system would additionally track:
  - Prediction calibration over time (reliability diagrams)
  - Positive-rate trends in confirmed labels once ground-truth is available
  - Rolling-window estimates (7-day / 150-row) for more stable drift signals

---

## 7. Running Tests

```bash
# All tests
make test

# Verbose
pytest tests/ -v

# Specific module
pytest tests/test_preprocessing.py -v
```

Tests cover:
- `test_ingestion.py` — DB init idempotency, ingest/retrieve, empty batch, history
- `test_preprocessing.py` — `"?"` replacement, NaN-free output, stats reuse, age_group validity
- `test_api.py` — health check, validation errors, batch predict count; model-dependent tests auto-skip if no model

---

## 8. Code Structure

### Organising principle: one concern, one module

Each directory under `src/` maps to exactly one pipeline concern. No module reaches into another module's responsibility. This makes the system easy to reason about, test in isolation, and replace incrementally — for example, swapping the SQLite ingestor for a PostgreSQL one requires changing only `src/ingestion/ingestor.py`, not touching preprocessing or training.

```
src/
├── ingestion/    ← How data enters the system (source-agnostic interface)
├── preprocessing/← How raw data is cleaned (stateful: saves imputation stats)
├── features/     ← How domain knowledge is encoded (pure transformation)
├── training/     ← How models are fitted and saved (input: DataFrame, output: artefacts)
├── inference/    ← How predictions are generated (lazy loads model, stateless per call)
├── drift/        ← How distribution shift is detected (reads reference, writes reports)
├── serving/      ← How predictions are exposed (HTTP layer only, no business logic)
└── pipeline/     ← How stages are orchestrated (calls the above in sequence)
```

### Why `scripts/` is separate from `src/`

`src/` contains importable library code — functions and classes with no side effects at import time. `scripts/` contains executable entry points that *do* things when run: download files, run training, trigger pipelines. This separation means the full library is safely importable in tests without triggering any I/O.

### Why `config.py` is at the root

All paths, constants, and environment variable overrides live in one file at the root. Any module that needs a path imports it from `config` — nothing hard-codes paths inline. This means the entire system can be relocated or containerised by changing one file.

### Logging strategy

Every module uses Python's standard `logging` library (not `print`). The scheduler writes structured JSONL to `logs/pipeline_runs.jsonl` — one JSON object per run — which means run history is machine-queryable, not just human-readable. Operational messages use clear visual markers (`✓`, `⚠`, `✗`) so log tailing during a live run is immediately scannable.

---

## 9. Project Structure

```
cervical-cancer-ml/
├── config.py                      # All paths, constants, env overrides
├── requirements.txt               # Pinned dependencies
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh                  # Bootstrap on first run, then start API
├── Makefile
│
├── src/
│   ├── ingestion/
│   │   ├── data_generator.py      # Simulates daily batch stream
│   │   └── ingestor.py            # SQLite write/read layer
│   ├── preprocessing/
│   │   └── preprocessor.py        # NaN impute, outlier clip, CSV save
│   ├── features/
│   │   └── feature_engineer.py    # Derived features (stds_total, age_group, …)
│   ├── training/
│   │   └── trainer.py             # XGBoost, 5-fold CV, SHAP, artefact save
│   ├── inference/
│   │   └── predictor.py           # Lazy model load, SHAP top-5 per prediction
│   ├── drift/
│   │   └── drift_detector.py      # Evidently AI drift reports
│   ├── serving/
│   │   └── api.py                 # FastAPI — prediction + monitoring endpoints
│   └── pipeline/
│       └── scheduler.py           # APScheduler daily cycle + manual trigger
│
├── scripts/
│   ├── download_data.py           # One-time UCI dataset download
│   ├── initial_train.py           # Bootstrap: ingest reference → train → save
│   └── run_pipeline.py            # Manual pipeline trigger
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── data/
│   ├── raw/                       # SQLite DB + raw CSV + stream state JSON
│   └── processed/                 # Timestamped feature CSVs + latest pointers
│
├── models/                        # xgboost_model.joblib, feature_names.json,
│                                  # metrics.json, preprocessing_stats.json,
│                                  # reference_stats.parquet
├── reports/                       # drift_report.html, drift_status.json
└── logs/                          # pipeline.log, pipeline_runs.jsonl
```

### Screening recommendation request / response

```bash
curl -X POST http://localhost:8000/screening-recommendation \
  -H "Content-Type: application/json" \
  -d '{"age": 29, "smokes": 1, "stds_hpv": 1, "hormonal_contraceptives": 1}'
```

```json
{
  "individual_risk": {
    "prediction":   1,
    "probability":  0.7231,
    "risk_level":   "high",
    "top_features": [
      {"feature": "stds_hpv",   "shap_value": 0.42310},
      {"feature": "stds_total", "shap_value": 0.31204}
    ]
  },
  "population_stratum": {
    "stratum":       "moderate",
    "recall_months": 12
  },
  "recommendation": {
    "final_stratum":   "high",
    "recall_months":   0,
    "action":          "colposcopy_referral",
    "label":           "Immediate colposcopy referral",
    "escalated":       true,
    "escalation_note": "Individual risk signal elevated above population stratum — recall interval shortened."
  },
  "disclaimer": "This recommendation is generated by a research prototype. All clinical decisions must be made by a qualified healthcare professional."
}
```

The `escalated: true` flag shows the integration logic at work — the patient's individual probability was high even though their population stratum was moderate, so the recommendation escalated to the higher tier.

---

## 10. Extension: National Registry Scale

This section documents how the pipeline would extend to the full research scenario described in the doctoral project — Sweden's national quality registry for cervical cancer prevention, with HPV genotyping, longitudinal screening history, and population-scale data.

### Data layer changes

| Current (demo) | National registry |
|---|---|
| UCI CSV, 858 patients | Sweden NKCK registry, millions of patients |
| Single snapshot per patient | Longitudinal records — multiple screening episodes per patient |
| Biopsy as target | HPV genotype progression, CIN grade, cancer incidence |
| Daily batch of 15 rows | Real-time feeds from regional laboratories |

The `src/ingestion/` module is the only layer that changes. All downstream modules (preprocessing, feature engineering, training, clustering, serving) operate on DataFrames and are completely data-source-agnostic.

### Feature layer changes

The feature engineering module would gain:

- **Longitudinal features**: time since last screening, number of previous HPV-positive results, HPV genotype sequence (HPV16/18 vs other high-risk vs low-risk)
- **Screening history features**: previous abnormal cytology, previous colposcopy, previous treatment (LEEP/cone biopsy)
- **Registry linkage features**: age at first sexual intercourse (from social registry), parity, socioeconomic indicators

### Imaging fusion extension

With access to Cervix-RT (TCIA) or similar datasets containing CT scans and expert GTV contours, the pipeline would add a third branch:

```
CT + RTSTRUCT (DICOM)
      │
  pydicom / SimpleITK
      │
  Mask extraction from RTSTRUCT
      │
  PyRadiomics feature extraction
  (shape, first-order, texture)
      │
  Feature selection (GA / filter)
      │
  Fusion with clinical tabular features
      ↓
  XGBoost (multimodal input)
```

This extension is not implemented here because the freely accessible TCGA-CESC imaging dataset (54 MRI scans) lacks GTV contour annotations, which are a prerequisite for scientifically valid radiomic feature extraction. PyRadiomics on whole-volume MRI without a tumour mask produces features that describe the entire pelvic anatomy rather than the tumour — an approach that would be scientifically indefensible in a clinical context.

### Scale infrastructure changes

At national registry scale, two components would be replaced:

| Current | Production |
|---|---|
| APScheduler (in-process) | Apache Airflow or Prefect (DAG-based, distributed) |
| SQLite (single-node) | PostgreSQL or BigQuery (concurrent writes, query scale) |
| Single Docker container | Kubernetes cluster with auto-scaling |

The `run_pipeline_cycle()` function in `src/pipeline/scheduler.py` is already structured as a self-contained unit that maps cleanly to an Airflow DAG task — this was a deliberate design choice.

---

## References

Fernandes, K., Cardoso, J. S., & Fernandes, J. (2017). *Transfer learning with
partial observability applied to cervical cancer screening.* In Iberian Conference
on Pattern Recognition and Image Analysis. Springer, Cham.

Dataset: [UCI ML Repository #383](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+(Risk+Factors))

---

**Clinical disclaimer**: This system is a research prototype for risk stratification.
It is not a validated medical device. All clinical decisions must be made by a
qualified healthcare professional.
