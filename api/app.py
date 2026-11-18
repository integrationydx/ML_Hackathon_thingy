"""FastAPI service for PathSnap candidate scoring and route display."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL", ROOT / "artifacts/model_final.joblib"))
METADATA_PATH = Path(os.getenv("PATHSNAP_METADATA", ROOT / "artifacts/model_metadata.json"))
MODEL_LOG_PATH = ROOT / "reports/member2/model_accuracy_log.json"

DEFAULT_FEATURES = [
    "distance_m", "heading_delta_deg", "projection_ratio",
    "distance_rank", "segment_length_m", "speed_mps", "dt_s",
    "heading_valid", "gap_before", "zero_length_arc",
]


class Candidate(BaseModel):
    arc_id: int
    features: dict[str, float | int | bool]
    geometry: list[list[float]] = Field(default_factory=list, description="[[lat, lon], ...]")

    @model_validator(mode="after")
    def validate_geometry(self) -> "Candidate":
        for point in self.geometry:
            if len(point) != 2 or not all(np.isfinite(point)):
                raise ValueError("geometry points must be finite [latitude, longitude] pairs")
        return self


class Observation(BaseModel):
    obs_idx: int
    candidates: list[Candidate] = Field(min_length=1)
    gps: list[float] | None = Field(default=None, description="[latitude, longitude]")

    @model_validator(mode="after")
    def validate_gps(self) -> "Observation":
        if self.gps is not None and (len(self.gps) != 2 or not all(np.isfinite(self.gps))):
            raise ValueError("gps must be a finite [latitude, longitude] pair")
        return self


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1)
    reference_route: list[int] = Field(default_factory=list)


class Prediction(BaseModel):
    obs_idx: int
    arc_id: int
    probability: float
    geometry: list[list[float]]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    predicted_route: list[int]
    gps: list[list[float]]
    reference_route: list[int]
    reference_geometry: list[list[list[float]]]
    model: dict[str, object]


def _load_model() -> object:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _load_metadata() -> dict[str, object]:
    if not METADATA_PATH.is_file():
        return {"features": DEFAULT_FEATURES, "model_path": str(MODEL_PATH)}
    return json.loads(METADATA_PATH.read_text())


model = _load_model()
metadata = _load_metadata()
feature_columns = list(metadata.get("features", DEFAULT_FEATURES))

app = FastAPI(title="PathSnap Evaluator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def evaluator_ui() -> FileResponse:
    return FileResponse(ROOT / "ui/index.html", media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": str(metadata.get("champion_model", "unknown"))}


@app.get("/metadata")
def get_metadata() -> dict[str, object]:
    return {"features": feature_columns, **metadata}


@app.get("/model-comparison")
def get_model_comparison() -> dict[str, object]:
    if not MODEL_LOG_PATH.is_file():
        raise HTTPException(status_code=404, detail="Model accuracy log not found")
    return json.loads(MODEL_LOG_PATH.read_text())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    rows = [candidate for observation in request.observations for candidate in observation.candidates]
    missing = sorted({name for row in rows for name in feature_columns if name not in row.features})
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model features: {', '.join(missing)}")

    try:
        matrix = np.asarray(
            [[float(row.features[name]) for name in feature_columns] for row in rows],
            dtype=float,
        )
        if not np.isfinite(matrix).all():
            raise ValueError("feature values must be finite")
        probabilities = model.predict_proba(matrix)[:, 1]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    predictions: list[Prediction] = []
    geometry_by_arc = {
        candidate.arc_id: candidate.geometry
        for observation in request.observations
        for candidate in observation.candidates
        if candidate.geometry
    }
    offset = 0
    for observation in request.observations:
        count = len(observation.candidates)
        scores = probabilities[offset:offset + count]
        selected = int(np.argmax(scores))
        candidate = observation.candidates[selected]
        predictions.append(Prediction(
            obs_idx=observation.obs_idx,
            arc_id=candidate.arc_id,
            probability=float(scores[selected]),
            geometry=candidate.geometry,
        ))
        offset += count

    return PredictResponse(
        predictions=predictions,
        predicted_route=[item.arc_id for item in predictions],
        gps=[item.gps for item in request.observations if item.gps is not None],
        reference_route=request.reference_route,
        reference_geometry=[
            geometry_by_arc[arc_id]
            for arc_id in request.reference_route
            if arc_id in geometry_by_arc
        ],
        model=metadata,
    )
