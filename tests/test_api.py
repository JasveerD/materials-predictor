"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import init_routes
from model.gnn import CrystalGNN
from model.predict import Prediction

import torch


@pytest.fixture()
def client():
    model = CrystalGNN(node_dim=102, edge_dim=4, hidden_dim=32, num_conv_layers=2)
    device = torch.device("cpu")
    init_routes(model, device, {"config": {"node_dim": 102, "edge_dim": 4, "hidden_dim": 32, "num_conv_layers": 2}, "val_mae": 0.35}, train_data=None)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client: TestClient):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "CrystalGNN"
    assert data["hidden_dim"] == 32


def test_predict_requires_input(client: TestClient):
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_with_cif(client: TestClient):
    mock_result = Prediction(band_gap=1.12, uncertainty=0.05, classification="semiconductor")
    with patch("api.routes.predict_from_cif", return_value=mock_result):
        with patch("api.routes._find_similar", return_value=[]):
            r = client.post("/predict", json={"cif": "data_fake\n_cell_length_a 5.43\n"})

    assert r.status_code == 200
    data = r.json()
    assert data["predicted_band_gap"] == 1.12
    assert data["classification"] == "semiconductor"
