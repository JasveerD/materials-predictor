"""CGCNN-inspired graph neural network for band gap prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import CGConv, global_max_pool, global_mean_pool


class CrystalGNN(nn.Module):
    """Graph neural network for predicting material properties from crystal graphs.

    Architecture: stacked CGConv layers with batch norm and residual connections,
    global mean+max pooling, and a 2-layer MLP regression head.
    """

    def __init__(
        self,
        node_dim: int = 102,  # 100 one-hot atomic number + 2 continuous features
        edge_dim: int = 4,    # distance + 3D unit vector
        hidden_dim: int = 128,
        num_conv_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(CGConv(hidden_dim, dim=edge_dim, batch_norm=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = self.node_embed(x)

        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x + residual)

        pooled = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=1)

        return self.head(pooled).squeeze(-1)
