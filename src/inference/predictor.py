import logging
from pathlib import Path
import pandas as pd
import shap
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)
_model = None; _explainer = None; _feature_names = None

def _load_artifacts():
    global _model, _explainer, _feature_names
    if _model is not None: return
    from src.training.trainer import load_model, load_feature_names
    _model = load_model()
    _feature_names = load_feature_names()
    _explainer = shap.TreeExplainer(_model)
    logger.info("Model and SHAP explainer loaded.")

def predict(df):
    _load_artifacts()
    missing = set(_feature_names) - set(df.columns)
    if missing:
        for col in missing: df[col] = 0.0
    X = df[_feature_names].astype(float)
    predictions  = _model.predict(X)
    probabilities = _model.predict_proba(X)[:,1]
    shap_values  = _explainer.shap_values(X)
    results = []
    for i in range(len(X)):
        prob = float(probabilities[i])
        risk_level = "high" if prob >= 0.6 else "moderate" if prob >= 0.3 else "low"
        top_features = sorted(
            [{"feature":k,"shap_value":round(float(v),5)} for k,v in zip(_feature_names,shap_values[i])],
            key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
        results.append({"prediction":int(predictions[i]),"probability":round(prob,4),
                        "risk_level":risk_level,"top_features":top_features})
    return results

def predict_single(features):
    return predict(pd.DataFrame([features]))[0]
