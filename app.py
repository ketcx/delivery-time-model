"""FastAPI app for delivery time predictions - Optimized for minimal Azure costs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.model import create_delivery_model

# Initialize FastAPI app
app = FastAPI(
    title="Delivery Time Prediction API",
    description="Predict delivery times based on distance, time, and day of week",
    version="1.0.0",
)

# Global variables for model and normalization stats
MODEL = None
DISTANCE_MEAN = None
DISTANCE_STD = None
TIME_MEAN = None
TIME_STD = None


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    distance_miles: float = Field(
        ..., ge=0, le=20, description="Distance in miles (0-20)"
    )
    time_of_day_hours: float = Field(
        ..., ge=8.0, le=20.0, description="Time in 24h format (8.0-20.0)"
    )
    is_weekend: int = Field(..., ge=0, le=1, description="0=weekday, 1=weekend")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "distance_miles": 5.5,
                "time_of_day_hours": 17.5,
                "is_weekend": 0,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    predicted_delivery_time_minutes: float
    distance_miles: float
    time_of_day_hours: float
    is_weekend: int
    is_rush_hour: bool


def _rush_hour_feature(time_of_day: float, is_weekend: int) -> float:
    """Compute rush hour feature."""
    is_morning_rush = 8.0 <= time_of_day < 10.0
    is_evening_rush = 16.0 <= time_of_day < 19.0
    is_weekday = is_weekend == 0
    return float(is_weekday and (is_morning_rush or is_evening_rush))


def _normalize(value: float, mean: float, std: float) -> float:
    """Normalize a value with mean and std."""
    if std == 0:
        return 0.0
    return (value - mean) / std


@app.on_event("startup")
async def load_model():
    """Load model and normalization stats on startup."""
    global MODEL, DISTANCE_MEAN, DISTANCE_STD, TIME_MEAN, TIME_STD

    # Paths
    model_path_env = os.getenv("MODEL_PATH")
    if model_path_env is None:
        raise RuntimeError(
            "MODEL_PATH environment variable is not set. "
            "Please set MODEL_PATH to a trained checkpoint, e.g. "
            "`runs/<run_name>/checkpoints/best.pt`."
        )
    model_path = Path(model_path_env)
    data_path = Path(os.getenv("DATA_PATH", "data/data_with_features.csv"))

    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    MODEL = create_delivery_model()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()

    # Load normalization stats
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    DISTANCE_MEAN = float(df["distance_miles"].mean())
    DISTANCE_STD = float(df["distance_miles"].std())
    TIME_MEAN = float(df["time_of_day_hours"].mean())
    TIME_STD = float(df["time_of_day_hours"].std())

    print(f"✅ Model loaded from {model_path}")
    print(f"✅ Normalization stats loaded from {data_path}")


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Delivery Time Prediction API is running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health() -> Dict[str, str | bool]:
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "stats_loaded": all(
            x is not None for x in [DISTANCE_MEAN, DISTANCE_STD, TIME_MEAN, TIME_STD]
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict delivery time based on input features."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Compute rush hour feature
    rush_hour = _rush_hour_feature(request.time_of_day_hours, request.is_weekend)

    # Normalize features
    distance_norm = _normalize(request.distance_miles, DISTANCE_MEAN, DISTANCE_STD)
    time_norm = _normalize(request.time_of_day_hours, TIME_MEAN, TIME_STD)

    # Create input tensor
    features = torch.tensor(
        [[distance_norm, time_norm, float(request.is_weekend), rush_hour]],
        dtype=torch.float32,
    )

    # Make prediction
    with torch.no_grad():
        prediction = MODEL(features)
        predicted_time = float(prediction.item())

    return PredictionResponse(
        predicted_delivery_time_minutes=round(predicted_time, 2),
        distance_miles=request.distance_miles,
        time_of_day_hours=request.time_of_day_hours,
        is_weekend=request.is_weekend,
        is_rush_hour=bool(rush_hour),
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    """Batch prediction endpoint for multiple deliveries."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(requests) > 100:
        raise HTTPException(
            status_code=400, detail="Maximum 100 predictions per batch request"
        )

    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)

    return results
