"""Tests for the GNN model forward pass."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from model.gnn import CrystalGNN


def _make_dummy_batch(n_graphs: int = 4, n_nodes: int = 3) -> Batch:
    """Create a small batch of dummy graphs."""
    graphs = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, 102)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr = torch.randn(4, 4)
        y = torch.tensor([1.5])
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(graphs)


def test_forward_pass_shape():
    model = CrystalGNN(node_dim=102, edge_dim=4, hidden_dim=32, num_conv_layers=2)
    batch = _make_dummy_batch()
    out = model(batch)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_forward_pass_no_nan():
    model = CrystalGNN(node_dim=102, edge_dim=4, hidden_dim=32, num_conv_layers=2)
    batch = _make_dummy_batch()
    out = model(batch)
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_model_gradients():
    model = CrystalGNN(node_dim=102, edge_dim=4, hidden_dim=32, num_conv_layers=2)
    batch = _make_dummy_batch()
    out = model(batch)
    loss = (out - batch.y).pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
