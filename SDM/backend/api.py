from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
import joblib
import time
import shap
import os
import gdown

from backend.preprocess import preprocess

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Habitat Suitability Prediction API")

# =====================================================
# MODEL CONFIG (KHÔNG ĐỂ .PKL TRONG GITHUB)
# =====================================================
MODEL_DIR = "backend/model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = {
    "xgboost": {
        "name": "XGBoost",
        "feature_path": "backend/model/feature_columns.pkl",
        "gdrive_id": "1dvTfFmUqTRD_NrLOs35psw_q7Ziw6PAb",
        "filename": "xgb_presence_model.pkl"
    },
    "randomforest": {
        "name": "Random Forest",
        "feature_path": "backend/model/feature_columns.pkl",
        "gdrive_id": "1s91CIaYbtkaJpyaWYTGJ_NfdvHcTVXXl",
        "filename": "rf_presence_model.pkl"
    },
    "logistic": {
        "name": "Logistic Regression",
        "feature_path": "backend/model/feature_columns.pkl",
        "gdrive_id": "1A8bXC8je_y7B4SMMeeXkO_7XSBjKDdea",
        "filename": "lr_presence_model.pkl"
    },
    "lgbm": {
        "name": "LightGBM",
        "feature_path": "backend/model/feature_columns.pkl",
        "gdrive_id": "1yq5mcl49NoVaqHvZg07Kz35Cyjy0OE0x",
        "filename": "lgbm_presence_model.pkl"
    },
    "stacking": {
        "name": "Stacking",
        "feature_path": "backend/model/feature_columns.pkl",
        "gdrive_id": "1v1fwXCqiSHvuCOhyVddtFNxo0Au-VOW6",
        "filename": "stacking_presence_model.pkl"
    }
}

DATA_PATH = "model_data/full_data.csv"

# =====================================================
# HELPER: DOWNLOAD MODEL IF NOT EXISTS
# =====================================================
def download_model_if_needed(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        print(f"⬇ Downloading model: {output_path}")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# =====================================================
# LOAD DATA
# =====================================================
full_data = pd.read_csv(DATA_PATH)

# =====================================================
# LOAD MODELS
# =====================================================
loaded_models = {}
loaded_features = {}
explainers = {}

for key, cfg in MODELS.items():
    try:
        model_path = os.path.join(MODEL_DIR, cfg["filename"])

        download_model_if_needed(cfg["gdrive_id"], model_path)

        model = joblib.load(model_path)
        features = joblib.load(cfg["feature_path"])

        loaded_models[key] = model
        loaded_features[key] = features

        if key in ["xgboost", "randomforest", "lgbm"]:
            explainers[key] = shap.TreeExplainer(model)
        else:
            explainers[key] = None

        print(f"✅ Loaded model: {key}")

    except Exception as e:
        print(f"[WARN] Cannot load {key}: {e}")

# =====================================================
# CURRENT MODEL
# =====================================================
current_model_key = "xgboost"

def get_current_model():
    return loaded_models[current_model_key]

def get_current_features():
    return loaded_features[current_model_key]

def get_current_explainer():
    return explainers[current_model_key]

# =====================================================
# SERVE FRONTEND
# =====================================================
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("frontend/index.html", encoding="utf-8") as f:
        return f.read()

# =====================================================
# API: HEALTH
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok", "model": MODELS[current_model_key]["name"]}

# =====================================================
# API: MODEL LIST
# =====================================================
@app.get("/models")
def get_models():
    return {
        "current": current_model_key,
        "models": [
            {"key": k, "name": v["name"]}
            for k, v in MODELS.items()
            if k in loaded_models
        ]
    }

@app.post("/set_model")
def set_model(payload: dict):
    global current_model_key
    key = payload.get("model_key")
    if key in loaded_models:
        current_model_key = key
        return {"success": True, "model": MODELS[key]["name"]}
    return {"success": False, "error": "Model not found"}

# =====================================================
# API: GLOBAL FEATURE IMPORTANCE
# =====================================================
@app.get("/feature_importance")
def feature_importance_api():
    model = get_current_model()
    features = get_current_features()

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        ranked = sorted(zip(features, imp), key=lambda x: x[1], reverse=True)[:5]
        return [{"feature": f, "importance": round(float(w), 4)} for f, w in ranked]

    return [{"feature": f, "importance": 0.0} for f in features[:5]]

# =====================================================
# API: MAP CLICK → PREDICT
# =====================================================
@app.post("/predict_by_location")
def predict_by_location(payload: dict):
    lat, lon = payload["lat"], payload["lon"]
    threshold = float(payload.get("threshold", 0.5))

    model = get_current_model()
    features = get_current_features()

    dists = (full_data["decimalLatitude"] - lat) ** 2 + \
            (full_data["decimalLongitude"] - lon) ** 2
    row = full_data.loc[dists.idxmin()]

    df = preprocess(pd.DataFrame([row]))[features]

    t0 = time.time()
    prob = model.predict_proba(df)[0][1]
    t1 = time.time()

    return {
        "label": "Present" if prob >= threshold else "Absent",
        "probability": round(float(prob), 4),
        "inference_time_ms": round((t1 - t0) * 1000, 2),
        "model": MODELS[current_model_key]["name"]
    }

# =====================================================
# API: CSV BATCH PREDICT
# =====================================================
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    model = get_current_model()
    features = get_current_features()

    df = pd.read_csv(file.file).head(800)
    df_proc = preprocess(df.copy())[features]

    t0 = time.time()
    probs = model.predict_proba(df_proc)[:, 1]
    t1 = time.time()

    df["probability"] = probs

    return {
        "rows": df.to_dict(orient="records"),
        "total_rows": len(df),
        "inference_time_ms": round((t1 - t0) * 1000, 2),
        "model": MODELS[current_model_key]["name"]
    }

# =====================================================
# API: LOCAL SHAP EXPLANATION
# =====================================================
@app.post("/local_explain")
def local_explain(payload: dict):
    lat, lon = payload["lat"], payload["lon"]

    explainer = get_current_explainer()
    features = get_current_features()

    if explainer is None:
        return {
            "local_features": [
                {"feature": f, "shap_value": 0.0}
                for f in features[:5]
            ],
            "note": "SHAP not supported for this model"
        }

    dists = (full_data["decimalLatitude"] - lat) ** 2 + \
            (full_data["decimalLongitude"] - lon) ** 2
    row = full_data.loc[dists.idxmin()]

    df = preprocess(pd.DataFrame([row]))[features]
    shap_vals = explainer.shap_values(df)[0]

    ranked = sorted(
        zip(features, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return {
        "local_features": [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in ranked
        ]
    }
