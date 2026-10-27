"""Convert crystal structures to PyTorch Geometric graph objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


RADIUS_CUTOFF = 8.0

ATOM_FEATURES = {
    "electronegativity": {
        1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
        9: 3.98, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58,
        17: 3.16, 19: 0.82, 20: 1.00, 22: 1.54, 26: 1.83, 29: 1.90, 30: 1.65,
    },
    "atomic_radius": {
        1: 0.25, 3: 1.45, 4: 1.05, 5: 0.85, 6: 0.70, 7: 0.65, 8: 0.60,
        9: 0.50, 11: 1.80, 12: 1.50, 13: 1.25, 14: 1.10, 15: 1.00, 16: 1.00,
        17: 1.00, 19: 2.20, 20: 1.80, 22: 1.40, 26: 1.40, 29: 1.35, 30: 1.35,
    },
}

MAX_ATOMIC_NUM = 100
NUM_GROUPS = 18
NUM_PERIODS = 7


def _get_element_features(atomic_num: int) -> torch.Tensor:
    """Build feature vector for a single atom: one-hot Z + continuous props."""
    one_hot = torch.zeros(MAX_ATOMIC_NUM)
    one_hot[min(atomic_num, MAX_ATOMIC_NUM) - 1] = 1.0

    en = ATOM_FEATURES["electronegativity"].get(atomic_num, 1.5)
    radius = ATOM_FEATURES["atomic_radius"].get(atomic_num, 1.2)

    continuous = torch.tensor([en, radius], dtype=torch.float32)
    return torch.cat([one_hot, continuous])


def structure_to_graph(structure: Structure, band_gap: float) -> Data:
    """Convert a pymatgen Structure to a PyG Data object."""
    node_features = []
    for site in structure:
        z = site.specie.Z
        node_features.append(_get_element_features(z))
    x = torch.stack(node_features)

    neighbors = structure.get_all_neighbors(RADIUS_CUTOFF, include_index=True)

    src_indices: list[int] = []
    dst_indices: list[int] = []
    edge_attrs: list[list[float]] = []

    for i, site_neighbors in enumerate(neighbors):
        for neighbor in site_neighbors:
            j = neighbor[2]  # index
            dist = neighbor[1]  # distance
            diff = neighbor[0].coords - structure[i].coords
            unit_vec = diff / (np.linalg.norm(diff) + 1e-8)

            src_indices.append(i)
            dst_indices.append(j)
            edge_attrs.append([dist, *unit_vec.tolist()])

    if not src_indices:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    y = torch.tensor([band_gap], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def process_dataset(
    raw_path: Path,
    output_dir: Path | None = None,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> dict[str, Path]:
    """Load raw JSON, convert to graphs, split, and save."""
    if output_dir is None:
        output_dir = raw_path.parent.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(raw_path.read_text())

    graphs: list[Data] = []
    labels_for_split: list[int] = []

    for rec in records:
        structure = Structure.from_dict(rec["structure"])
        graph = structure_to_graph(structure, rec["band_gap"])
        graph.material_id = rec["material_id"]
        graph.formula = rec["formula"]
        graphs.append(graph)

        bg = rec["band_gap"]
        if bg == 0:
            labels_for_split.append(0)  # metal
        elif bg < 2.0:
            labels_for_split.append(1)  # semiconductor
        else:
            labels_for_split.append(2)  # insulator

    idx = list(range(len(graphs)))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, labels_for_split, test_size=test_size + val_size,
        stratify=labels_for_split, random_state=42,
    )
    relative_val = val_size / (test_size + val_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=1 - relative_val,
        stratify=y_temp, random_state=42,
    )

    splits: dict[str, Path] = {}
    for name, indices in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        split_data = [graphs[i] for i in indices]
        path = output_dir / f"{name}.pt"
        torch.save(split_data, path)
        splits[name] = path
        print(f"{name}: {len(split_data)} graphs -> {path}")

    return splits


if __name__ == "__main__":
    raw = Path(__file__).resolve().parent / "raw" / "materials.json"
    process_dataset(raw)
