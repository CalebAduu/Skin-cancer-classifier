"""
FastAPI serving layer.

    uvicorn app.main:app --reload

POST /predict  (multipart/form-data)
    image        : file (required)
    age          : float (optional)
    sex          : str   (optional: male/female/unknown)
    localization : str   (optional: e.g. back, face, upper extremity)
    explain      : bool  (optional: returns a Score-CAM overlay PNG, base64)

This is a decision-support research prototype, NOT a diagnostic device. Do not
use for clinical decisions.
"""
from __future__ import annotations

import io
import os
import time

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import Predictor

BUNDLE_PATH = os.environ.get("BUNDLE_PATH", "saved_models/inference_bundle.pt")
# Comma-separated list of allowed browser origins for the frontend, e.g.
# "https://dermserve.vercel.app,http://localhost:3000"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Dermoscopy Multimodal Classifier", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
_predictor: Predictor | None = None


@app.on_event("startup")
def _load_model():
    global _predictor
    _predictor = Predictor(BUNDLE_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: float | None = Form(None),
    sex: str = Form("unknown"),
    localization: str = Form("unknown"),
    explain: bool = Form(False),
):
    if _predictor is None:
        raise HTTPException(503, "Model not loaded")
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception:
        raise HTTPException(400, "Could not read image file")

    t0 = time.perf_counter()
    result = _predictor.predict(img, age=age, sex=sex, localization=localization, explain=explain)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["disclaimer"] = "Research prototype for decision support only; not a diagnostic device."
    return result
