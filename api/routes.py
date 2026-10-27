"""API route definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pymatgen.core import Structure

from api.schemas import (
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    SimilarMaterial,
)
from data.process import structure_to_graph
from model.predict import predict_from_cif, predict_structure

if TYPE_CHECKING:
    from model.gnn import CrystalGNN

router = APIRouter()

_model: CrystalGNN | None = None
_device: torch.device | None = None
_checkpoint_meta: dict = {}
_train_data: list | None = None


def init_routes(
    model: CrystalGNN,
    device: torch.device,
    checkpoint_meta: dict,
    train_data: list | None = None,
) -> None:
    """Inject the loaded model and metadata into the router's module state."""
    global _model, _device, _checkpoint_meta, _train_data
    _model = model
    _device = device
    _checkpoint_meta = checkpoint_meta
    _train_data = train_data


def _find_similar(
    structure: Structure,
    k: int = 5,
) -> list[SimilarMaterial]:
    """Find k nearest neighbors from training set by graph-level embedding distance."""
    if _train_data is None or _model is None or _device is None:
        return []

    query_graph = structure_to_graph(structure, band_gap=0.0)
    query_graph.batch = torch.zeros(query_graph.x.shape[0], dtype=torch.long)
    query_graph = query_graph.to(_device)

    _model.eval()
    with torch.no_grad():
        qx = _model.node_embed(query_graph.x)
        for conv, bn in zip(_model.convs, _model.bns):
            residual = qx
            qx = conv(qx, query_graph.edge_index, query_graph.edge_attr)
            qx = bn(qx)
            qx = torch.relu(qx + residual)
        from torch_geometric.nn import global_mean_pool
        q_embed = global_mean_pool(qx, query_graph.batch)

    distances: list[tuple[float, int]] = []
    from torch_geometric.loader import DataLoader

    loader = DataLoader(_train_data, batch_size=128)
    idx_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(_device)
            bx = _model.node_embed(batch.x)
            for conv, bn in zip(_model.convs, _model.bns):
                residual = bx
                bx = conv(bx, batch.edge_index, batch.edge_attr)
                bx = bn(bx)
                bx = torch.relu(bx + residual)
            t_embed = global_mean_pool(bx, batch.batch)

            dists = torch.norm(t_embed - q_embed, dim=1)
            for i, d in enumerate(dists.cpu().tolist()):
                distances.append((d, idx_offset + i))
            idx_offset += batch.num_graphs

    distances.sort(key=lambda x: x[0])
    results: list[SimilarMaterial] = []
    for _, idx in distances[:k]:
        item = _train_data[idx]
        results.append(SimilarMaterial(
            material_id=getattr(item, "material_id", f"train-{idx}"),
            formula=getattr(item, "formula", "unknown"),
            band_gap=round(float(item.y.item()), 4),
        ))
    return results


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    config = _checkpoint_meta.get("config", {})
    return ModelInfo(
        val_mae=_checkpoint_meta.get("val_mae", -1.0),
        node_dim=config.get("node_dim", 0),
        edge_dim=config.get("edge_dim", 0),
        hidden_dim=config.get("hidden_dim", 0),
        num_conv_layers=config.get("num_conv_layers", 0),
        num_training_materials=len(_train_data) if _train_data else 0,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if _model is None or _device is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.cif and not req.material_id:
        raise HTTPException(status_code=422, detail="Provide either 'cif' or 'material_id'")

    try:
        if req.cif:
            result = predict_from_cif(_model, req.cif, _device)
            structure = Structure.from_str(req.cif, fmt="cif")
        else:
            import os
            from mp_api.client import MPRester

            api_key = os.environ.get("MP_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=503,
                    detail="MP_API_KEY not set; cannot fetch by material_id",
                )
            with MPRester(api_key) as mpr:
                doc = mpr.materials.summary.get_data_by_id(req.material_id)
            structure = doc.structure
            result = predict_structure(_model, structure, _device)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    similar = _find_similar(structure, k=5)

    return PredictResponse(
        predicted_band_gap=result.band_gap,
        uncertainty=result.uncertainty,
        classification=result.classification,
        similar_materials=similar,
    )
