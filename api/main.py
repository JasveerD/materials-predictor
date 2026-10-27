"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import init_routes, router
from model.evaluate import load_model

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"
TRAIN_DATA_PATH = Path(__file__).resolve().parent.parent / "processed" / "train.pt"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CHECKPOINT_PATH.exists():
        model = load_model(CHECKPOINT_PATH, device)
        ckpt_meta = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    else:
        model = None  # type: ignore[assignment]
        ckpt_meta = {}
        print(f"Warning: no checkpoint at {CHECKPOINT_PATH}, /predict will return 503")

    train_data = None
    if TRAIN_DATA_PATH.exists():
        train_data = torch.load(TRAIN_DATA_PATH, weights_only=False)

    init_routes(model, device, ckpt_meta, train_data)
    yield


app = FastAPI(
    title="Materials Band Gap Predictor",
    description="Predict band gaps of crystal structures using a graph neural network",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)

if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
