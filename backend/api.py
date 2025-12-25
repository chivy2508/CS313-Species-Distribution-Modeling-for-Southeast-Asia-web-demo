from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
import time
import shap
import rasterio
from pathlib import Path
from fastapi.responses import FileResponse

from backend.preprocess import preprocess

app = FastAPI(title="Habitat Suitability Prediction API")

# =====================================================
# MODEL CONFIG
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
# CLIMATE SCENARIOS CONFIG
# =====================================================
CLIMATE_SCENARIOS = {
    "2041-2060_ssp245": {
        "path": "backend/data/SEA_wc2.1_30s_bioc_HadGEM3-GC31-LL_ssp245_2041-2060.tif",
        "gdrive_id": "111dGgeREwYJ6ZiA1TKr79e73V8HOhQ9S"
    },
    "2041-2060_ssp585": {
        "path": "backend/data/SEA_wc2.1_30s_bioc_HadGEM3-GC31-LL_ssp585_2041-2060.tif",
        "gdrive_id": "1XX-GJ5KmnPBgJ6sV7hTz9YmBLBLWsazA"
    },
    "2061-2080_ssp245": {
        "path": "backend/data/SEA_wc2.1_30s_bioc_HadGEM3-GC31-LL_ssp245_2061-2080.tif",
        "gdrive_id": "1dFOriXu3PTLYGNvS-uffEO7G2JNIL33T"
    },
    "2061-2080_ssp585": {
        "path": "backend/data/SEA_wc2.1_30s_bioc_HadGEM3-GC31-LL_ssp585_2061-2080.tif",
        "gdrive_id": "10wykayenfpsAiH8-85WBnGETgZicXUOn"
    }
}


# Load climate data
climate_data = {}
for key, path in CLIMATE_SCENARIOS.items():
    try:
        if Path(path).exists():
            climate_data[key] = rasterio.open(path)
            print(f"✓ Loaded climate data: {key} ({climate_data[key].count} bands)")
        else:
            print(f"✗ File not found: {path}")
    except Exception as e:
        print(f"✗ Cannot load climate data {key}: {e}")

# =====================================================
# LOAD MODELS
# =====================================================
loaded_models = {}
loaded_features = {}
explainers = {}

for key, cfg in MODELS.items():
    try:
        model = joblib.load(cfg["model_path"])
        feats = joblib.load(cfg["feature_path"])
        loaded_models[key] = model
        loaded_features[key] = feats

        # SHAP only for tree-based models
        if key in ["xgboost", "randomforest", "lgbm", "stacking"]:
            explainers[key] = shap.TreeExplainer(model)
        else:
            explainers[key] = None

        print(f"✓ Loaded model: {key}")
    except Exception as e:
        print(f"✗ Cannot load {key}: {e}")

full_data = pd.read_csv(DATA_PATH)
print(f"✓ Loaded training data: {len(full_data)} rows")

current_model_key = "xgboost"

def get_current_model():
    return loaded_models[current_model_key]

def get_current_features():
    return loaded_features[current_model_key]

def get_current_explainer():
    return explainers[current_model_key]

# =====================================================
# CLIMATE DATA EXTRACTION
# =====================================================
def extract_bioclim_from_tif(lat, lon, scenario_key):
    if scenario_key not in climate_data:
        print(f"⚠️  Scenario '{scenario_key}' not loaded in climate_data")
        return None
    
    try:
        dataset = climate_data[scenario_key]
        row, col = dataset.index(lon, lat)

        if row < 0 or row >= dataset.height or col < 0 or col >= dataset.width:
            print(f"⚠️  Coordinates ({lat}, {lon}) are out of bounds")
            return None
        
        bio_values = {}
        for i in range(1, 20):
            value = dataset.read(
                i,
                window=((row, row + 1), (col, col + 1))
            )[0, 0]

            if dataset.nodata is not None and value == dataset.nodata:
                print(f"⚠️  NoData value at ({lat}, {lon}) for bio{i}")
                return None

            bio_values[f"bio{i}"] = float(value)
        
        print(f"✓ Extracted {len(bio_values)} bioclim vars for ({lat:.4f}, {lon:.4f}) from {scenario_key}")
        return bio_values

    except Exception as e:
        print(f"❌ Error extracting bioclim data: {e}")
        return None

# =====================================================
# FRONTEND
# =====================================================
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("frontend/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico")
def favicon():
    try:
        return FileResponse("frontend/favicon.ico")
    except:
        return {"status": "no favicon"}

# =====================================================
# MODEL APIs
# =====================================================
@app.get("/models")
def get_models():
    return {
        "models": [
            {"key": k, "name": MODELS[k]["name"]}
            for k in loaded_models
        ],
        "current": current_model_key
    }

@app.post("/set_model")
def set_model(payload: dict):
    global current_model_key
    key = payload.get("model_key")
    if key in loaded_models:
        current_model_key = key
        return {"success": True, "model": MODELS[key]["name"]}
    return {"success": False}

# =====================================================
# CLIMATE SCENARIOS API
# =====================================================
@app.get("/climate_scenarios")
def get_climate_scenarios():
    return {
        "scenarios": [
            {
                "key": k, 
                "label": k.replace("_", " ").upper(),
                "loaded": k in climate_data
            }
            for k in CLIMATE_SCENARIOS.keys()
        ]
    }

# =====================================================
# PRESENCE DATA API
# =====================================================
@app.get("/presence_points")
def get_presence_points():
    """Get actual presence points from training data"""
    try:
        presence_data = full_data[full_data['presence'] == 1]
        
        points = []
        for _, row in presence_data.iterrows():
            points.append({
                "lat": float(row['decimalLatitude']),
                "lon": float(row['decimalLongitude'])
            })
        
        return {
            "points": points,
            "total": len(points)
        }
    except Exception as e:
        print(f"Error getting presence points: {e}")
        return {"points": [], "total": 0}

# =====================================================
# FEATURE IMPORTANCE (GLOBAL)
# =====================================================
@app.get("/feature_importance")
def feature_importance_api():
    model = get_current_model()
    features = get_current_features()

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        data = sorted(zip(features, imp), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 4)} for f, v in data[:5]]
    else:
        return [{"feature": f, "importance": 0.0} for f in features[:5]]

# =====================================================
# PREDICT BY LOCATION (FIXED WITH BIO MAPPING)
# =====================================================
@app.post("/predict_by_location")
def predict_by_location(payload: dict):
    lat, lon = payload["lat"], payload["lon"]
    threshold = payload.get("threshold", 0.5)
    use_future = payload.get("use_future", False)
    scenario = payload.get("scenario", "2041-2060_ssp245")

    print(f"\n{'='*60}")
    print(f"PREDICT REQUEST: lat={lat:.4f}, lon={lon:.4f}")
    print(f"Mode: {'FUTURE' if use_future else 'CURRENT'}")
    if use_future:
        print(f"Scenario: {scenario}")

    model = get_current_model()
    features = get_current_features()

    if use_future:
        # Use future climate data from .tif
        print(f"Extracting future climate data from {scenario}...")
        bio_values = extract_bioclim_from_tif(lat, lon, scenario)
        
        if bio_values is None:
            error_msg = f"Cannot extract climate data for location ({lat:.4f}, {lon:.4f}) in scenario {scenario}"
            print(f"⚠️  {error_msg}")
            return {"error": error_msg}
        
        # Find nearest row for other features (non-climate)
        dists = (full_data["decimalLatitude"] - lat) ** 2 + \
                (full_data["decimalLongitude"] - lon) ** 2
        row = full_data.loc[dists.idxmin()].to_dict()
        
        # Check if bio columns use underscore format (bio_1) or not (bio1)
        sample_keys = list(row.keys())
        uses_underscore = any('bio_' in str(k) for k in sample_keys)
        
        print(f"🔍 Data format: {'bio_X (with underscore)' if uses_underscore else 'bioX (no underscore)'}")
        
        # Store original bio values (handle both formats)
        original_bio_values = {}
        for i in range(1, 20):
            bio_key_underscore = f"bio_{i}"
            bio_key_no_underscore = f"bio{i}"
            
            if bio_key_underscore in row:
                original_bio_values[f"bio{i}"] = row[bio_key_underscore]
            elif bio_key_no_underscore in row:
                original_bio_values[f"bio{i}"] = row[bio_key_no_underscore]
            else:
                original_bio_values[f"bio{i}"] = None
        
        # Replace bio variables with future values
        # Map correctly based on data format
        for bio_key, bio_val in bio_values.items():
            if uses_underscore:
                # Convert bio1 -> bio_1
                bio_num = bio_key.replace('bio', '')
                mapped_key = f"bio_{bio_num}"
                row[mapped_key] = bio_val
            else:
                # Use as is: bio1 -> bio1
                row[bio_key] = bio_val
        
        # Print detailed comparison
        print("\n" + "🌡️  BIO VALUES COMPARISON ".center(60, "="))
        changes = []
        for i in range(1, 20):
            bio_key = f"bio{i}"
            original = original_bio_values.get(bio_key)
            future = bio_values.get(bio_key)
            
            if original is not None and future is not None:
                change = future - original
                change_pct = (change / original * 100) if original != 0 else 0
                symbol = "🔺" if change > 0 else "🔻" if change < 0 else "➡️"
                print(f"{symbol} {bio_key:6} | Original: {original:9.2f} → Future: {future:9.2f} | Δ {change:+9.2f} ({change_pct:+6.1f}%)")
                changes.append(abs(change_pct))
        
        if changes:
            avg_change = sum(changes) / len(changes)
            print(f"\n📊 Average absolute change: {avg_change:.1f}%")
        else:
            print("⚠️  No valid comparisons found")
        
        print("="*60 + "\n")
        
        df = preprocess(pd.DataFrame([row]))[features]
    else:
        # Use current data
        print("Using current climate data from training set...")
        dists = (full_data["decimalLatitude"] - lat) ** 2 + \
                (full_data["decimalLongitude"] - lon) ** 2
        row = full_data.loc[dists.idxmin()]
        df = preprocess(pd.DataFrame([row]))[features]

    t0 = time.time()
    prob = model.predict_proba(df)[0][1]
    t1 = time.time()

    print(f"Prediction: {prob:.4f} ({prob*100:.2f}%)")
    print(f"{'='*60}\n")

    response = {
        "label": "Present" if prob >= threshold else "Absent",
        "probability": round(float(prob), 4),
        "inference_time_ms": round((t1 - t0) * 1000, 2),
        "model": MODELS[current_model_key]["name"]
    }
    
    if use_future:
        response["scenario"] = scenario

    return response

@app.post("/compare_scenarios")
def compare_scenarios(payload: dict):
    lat, lon = payload["lat"], payload["lon"]

    model = get_current_model()
    features = get_current_features()

    results = []

    # ===== CURRENT =====
    dists = (full_data["decimalLatitude"] - lat) ** 2 + \
            (full_data["decimalLongitude"] - lon) ** 2
    base_row = full_data.loc[dists.idxmin()].to_dict()

    df_cur = preprocess(pd.DataFrame([base_row]))[features]
    prob_cur = model.predict_proba(df_cur)[0][1]

    results.append({
        "scenario": "Current",
        "probability": round(float(prob_cur), 4)
    })

    # ===== FUTURE SCENARIOS =====
    for scenario in CLIMATE_SCENARIOS.keys():
        bio_values = extract_bioclim_from_tif(lat, lon, scenario)
        if bio_values is None:
            continue

        row = base_row.copy()
        for k, v in bio_values.items():
            row[k] = v

        df = preprocess(pd.DataFrame([row]))[features]
        prob = model.predict_proba(df)[0][1]

        results.append({
            "scenario": scenario,
            "probability": round(float(prob), 4)
        })

    return {
        "lat": lat,
        "lon": lon,
        "results": results
    }

# =====================================================
# CSV PREDICT (FIXED WITH BIO MAPPING)
# =====================================================
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), use_future: bool = False, scenario: str = "2041-2060_ssp245"):
    model = get_current_model()
    features = get_current_features()

    df = pd.read_csv(file.file).head(800)
    
    print(f"\n{'='*60}")
    print(f"CSV PREDICT: {len(df)} locations")
    print(f"Mode: {'FUTURE' if use_future else 'CURRENT'}")
    if use_future:
        print(f"Scenario: {scenario}")
    
    if use_future and scenario in climate_data:
        # Check data format
        uses_underscore = any('bio_' in str(col) for col in df.columns)
        print(f"🔍 CSV data format: {'bio_X' if uses_underscore else 'bioX'}")
        
        # Extract future climate data for each location
        success_count = 0
        for idx, row in df.iterrows():
            lat, lon = row['decimalLatitude'], row['decimalLongitude']
            bio_values = extract_bioclim_from_tif(lat, lon, scenario)
            
            if bio_values:
                # Map correctly based on data format
                for bio_key, bio_val in bio_values.items():
                    if uses_underscore:
                        # Convert bio1 -> bio_1
                        bio_num = bio_key.replace('bio', '')
                        mapped_key = f"bio_{bio_num}"
                        if mapped_key in df.columns:
                            df.at[idx, mapped_key] = bio_val
                    else:
                        # Use as is
                        if bio_key in df.columns:
                            df.at[idx, bio_key] = bio_val
                success_count += 1
        
        print(f"Successfully extracted climate data for {success_count}/{len(df)} locations")
    
    df_proc = preprocess(df.copy())[features]

    t0 = time.time()
    probs = model.predict_proba(df_proc)[:, 1]
    t1 = time.time()

    df["probability"] = probs
    
    print(f"Avg probability: {np.mean(probs):.4f}")
    print(f"{'='*60}\n")

    response = {
        "rows": df.to_dict(orient="records"),
        "total_rows": len(df),
        "inference_time_ms": round((t1 - t0) * 1000, 2),
        "model": MODELS[current_model_key]["name"]
    }
    
    if use_future:
        response["scenario"] = scenario

    return response

# =====================================================
# LOCAL SHAP (FIXED WITH BIO MAPPING)
# =====================================================
@app.post("/local_explain")
def local_explain(payload: dict):
    lat, lon = payload["lat"], payload["lon"]
    use_future = payload.get("use_future", False)
    scenario = payload.get("scenario", "2041-2060_ssp245")

    explainer = get_current_explainer()
    features = get_current_features()

    if use_future and scenario in climate_data:
        bio_values = extract_bioclim_from_tif(lat, lon, scenario)
        if bio_values is None:
            return {"error": "Cannot extract climate data for SHAP explanation"}
        
        dists = (full_data["decimalLatitude"] - lat) ** 2 + \
                (full_data["decimalLongitude"] - lon) ** 2
        row = full_data.loc[dists.idxmin()].to_dict()
        
        # Check data format and map correctly
        sample_keys = list(row.keys())
        uses_underscore = any('bio_' in str(k) for k in sample_keys)
        
        for bio_key, bio_val in bio_values.items():
            if uses_underscore:
                bio_num = bio_key.replace('bio', '')
                mapped_key = f"bio_{bio_num}"
                row[mapped_key] = bio_val
            else:
                row[bio_key] = bio_val
        
        df = preprocess(pd.DataFrame([row]))[features]
    else:
        dists = (full_data["decimalLatitude"] - lat) ** 2 + \
                (full_data["decimalLongitude"] - lon) ** 2
        row = full_data.loc[dists.idxmin()]
        df = preprocess(pd.DataFrame([row]))[features]

    if explainer is None:
        return {
            "local_features": [
                {"feature": f, "shap_value": 0.0} for f in features[:5]
            ],
            "note": "SHAP not supported for this model"
        }

    try:
        shap_vals = explainer.shap_values(df)[0]
        top = sorted(zip(features, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]

        return {
            "local_features": [
                {"feature": f, "shap_value": round(float(v), 4)} for f, v in top
            ]
        }
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return {
            "local_features": [],
            "note": f"Error: {str(e)}"

        }
