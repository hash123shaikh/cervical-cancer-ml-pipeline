import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CLUSTER_MODEL_PATH, CLUSTER_SUMMARY_PATH, N_RISK_STRATA, TARGET_COLUMN, AUXILIARY_TARGETS, MODELS_DIR

logger = logging.getLogger(__name__)
_EXCLUDE = {TARGET_COLUMN, *AUXILIARY_TARGETS, "ingested_at","ingestion_id","id"}
_INTERVAL_POLICY = {
    "low":      {"recall_months":36,"action":"routine_recall","label":"Routine 3-year recall"},
    "moderate": {"recall_months":12,"action":"early_recall","label":"Early 12-month recall"},
    "high":     {"recall_months":0, "action":"colposcopy_referral","label":"Immediate colposcopy referral"}}

def _feature_matrix(df):
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    return df[feature_cols].astype(float).values, feature_cols

def _select_k(X_scaled, k_min=2, k_max=5):
    if len(X_scaled) < k_max*2: return N_RISK_STRATA
    best_k, best_score = N_RISK_STRATA, -1.0
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels, sample_size=min(1000,len(X_scaled)))
        if score > best_score: best_k, best_score = k, score
    return best_k

def _label_clusters(kmeans, X_scaled, df):
    cluster_ids = kmeans.predict(X_scaled)
    k = kmeans.n_clusters
    if TARGET_COLUMN in df.columns:
        mean_risk = {i: float(df[TARGET_COLUMN].iloc[cluster_ids==i].mean()) for i in range(k)}
    else:
        distances = np.min(np.array([np.linalg.norm(X_scaled-c,axis=1) for c in kmeans.cluster_centers_]),axis=0)
        mean_risk = {i: float(distances[cluster_ids==i].mean()) for i in range(k)}
    sorted_clusters = sorted(mean_risk.items(), key=lambda x: x[1])
    n = len(sorted_clusters)
    if n == 2:
        return {sorted_clusters[0][0]:"low", sorted_clusters[1][0]:"high"}
    elif n == 3:
        return {sorted_clusters[0][0]:"low", sorted_clusters[1][0]:"moderate", sorted_clusters[2][0]:"high"}
    else:
        mapping = {}
        for idx,(cid,_) in enumerate(sorted_clusters):
            mapping[cid] = "low" if idx < n//3 else "moderate" if idx < 2*n//3 else "high"
        return mapping

def fit_stratifier(df):
    X, feature_cols = _feature_matrix(df)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k        = _select_k(X_scaled)
    kmeans   = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    cluster_to_stratum = _label_clusters(kmeans, X_scaled, df)
    cluster_ids = kmeans.predict(X_scaled)
    strata      = np.array([cluster_to_stratum[c] for c in cluster_ids])
    stratum_stats = {}
    for stratum in set(cluster_to_stratum.values()):
        mask = strata == stratum
        n    = int(mask.sum())
        stats = {"n_patients":n,"pct_patients":round(100*n/len(df),1),"policy":_INTERVAL_POLICY[stratum]}
        if TARGET_COLUMN in df.columns:
            stats["mean_biopsy_rate"] = round(float(df[TARGET_COLUMN].iloc[mask].mean()),4)
        stratum_stats[stratum] = stats
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler":scaler,"kmeans":kmeans,"cluster_to_stratum":cluster_to_stratum,"feature_cols":feature_cols}, CLUSTER_MODEL_PATH)
    summary = {"k":k,"n_patients":len(df),"stratum_distribution":stratum_stats,
        "silhouette_score":round(float(silhouette_score(X_scaled,cluster_ids,sample_size=min(1000,len(X_scaled)))),4)}
    with open(CLUSTER_SUMMARY_PATH,"w") as f: json.dump(summary, f, indent=2)
    logger.info("Stratifier fitted: k=%d, silhouette=%.3f", k, summary["silhouette_score"])
    return summary

def assign_stratum(df):
    artefact = _load_artefact()
    scaler, kmeans = artefact["scaler"], artefact["kmeans"]
    cluster_to_stratum = artefact["cluster_to_stratum"]
    feature_cols       = artefact["feature_cols"]
    missing = set(feature_cols) - set(df.columns)
    for col in missing: df[col] = 0.0
    X        = df[feature_cols].astype(float).values
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)
    results = []
    for c in clusters:
        stratum = cluster_to_stratum.get(int(c),"moderate")
        results.append({"stratum":stratum, **_INTERVAL_POLICY[stratum]})
    return results

def get_stratum_summary():
    if not CLUSTER_SUMMARY_PATH.exists():
        return {"message":"Stratifier not fitted yet. POST /pipeline/trigger to start."}
    with open(CLUSTER_SUMMARY_PATH) as f: return json.load(f)

def _load_artefact():
    if not CLUSTER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Risk stratifier not found at {CLUSTER_MODEL_PATH}.")
    return joblib.load(CLUSTER_MODEL_PATH)
