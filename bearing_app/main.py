"""FastAPI service for bearing Remaining Useful Life prediction."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from .prognosticator import BearingPrognosticator


PACKAGE_DIR = Path(__file__).resolve().parent
ASSET_DIR = PACKAGE_DIR / "Assets"
MODEL_PATH = ASSET_DIR / "model.json"
CONFIG_PATH = ASSET_DIR / "config.json"


class PredictionRequest(BaseModel):
    """Raw chronological vibration snapshots expected by the model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "signals": [
                    [0.10, -0.20, 0.30, 0.05],
                    [0.11, -0.18, 0.28, 0.07],
                ]
            }
        }
    )

    signals: list[list[float]]


def load_predictor() -> BearingPrognosticator:
    """Load model configuration and the trained XGBoost artifact."""
    if not CONFIG_PATH.is_file():
        raise RuntimeError(f"Missing model configuration: {CONFIG_PATH}")
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Missing trained model: {MODEL_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    return BearingPrognosticator(MODEL_PATH, config)


def create_app(predictor: BearingPrognosticator | None = None) -> FastAPI:
    """Create the API app. A predictor can be injected for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "predictor", None) is None:
            app.state.predictor = load_predictor()
        yield

    app = FastAPI(
        title="Bearing RUL Prediction API",
        description="Predict Remaining Useful Life from sequential vibration snapshots.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.predictor = predictor

    @app.get("/")
    def root() -> dict[str, str]:
        return {"message": "Bearing RUL Prediction API", "docs": "/docs"}

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        status = "ready" if request.app.state.predictor is not None else "loading"
        return {"status": status}

    @app.post("/predict")
    def predict_rul(request: Request, payload: PredictionRequest) -> dict[str, float | str]:
        try:
            prediction = request.app.state.predictor.predict_rul(payload.signals)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed due to an internal server error.",
            ) from exc

        return {"predicted_rul": prediction, "status": "success"}

    return app


app = create_app()
