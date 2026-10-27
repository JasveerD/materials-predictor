"""Pydantic models for API request/response shapes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    cif: str | None = Field(None, description="Raw CIF file content")
    material_id: str | None = Field(None, description="Materials Project ID (e.g. mp-149)")


class SimilarMaterial(BaseModel):
    material_id: str
    formula: str
    band_gap: float


class PredictResponse(BaseModel):
    predicted_band_gap: float = Field(description="Predicted band gap in eV")
    uncertainty: float = Field(description="Uncertainty estimate (std of MC dropout)")
    classification: str = Field(description="metal / semiconductor / insulator")
    similar_materials: list[SimilarMaterial] = Field(default_factory=list)


class ModelInfo(BaseModel):
    model_name: str = "CrystalGNN"
    val_mae: float
    node_dim: int
    edge_dim: int
    hidden_dim: int
    num_conv_layers: int
    num_training_materials: int


class HealthResponse(BaseModel):
    status: str = "ok"
