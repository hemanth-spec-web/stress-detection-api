from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pickle
import os

# ── Load model & scaler once at startup ──────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "rf_smote_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# ── Label mapping (WESAD protocol) ───────────────────────────────────────────
LABEL_MAP = {
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stress Detection API",
    description=(
        "Predicts stress state from 20 physiological features extracted from "
        "WESAD-style signals (EDA, ECG, HRV, RESP, EMG, TEMP, BVP, ACC). "
        "Model: RandomForestClassifier (200 trees) trained with SMOTE balancing."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class FeaturesInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=20,
        max_length=20,
        example=[0.0] * 20,
        description="Exactly 20 physiological features in the same order used during training.",
    )

class PredictionResponse(BaseModel):
    predicted_class: int
    label: str
    probabilities: dict[str, float]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "Stress Detection API is running."}

@app.get("/info", summary="Model info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "n_features": model.n_features_in_,
        "classes": [LABEL_MAP.get(c, str(c)) for c in model.classes_.tolist()],
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict stress state")
def predict(body: FeaturesInput):
    try:
        x = np.array(body.features, dtype=np.float64).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred_class = int(model.predict(x_scaled)[0])
        proba = model.predict_proba(x_scaled)[0]

        proba_dict = {
            LABEL_MAP.get(int(cls), str(cls)): round(float(p), 4)
            for cls, p in zip(model.classes_, proba)
        }

        return PredictionResponse(
            predicted_class=pred_class,
            label=LABEL_MAP.get(pred_class, str(pred_class)),
            probabilities=proba_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
