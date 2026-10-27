"""Tests for data processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure

from data.process import structure_to_graph


def _make_si_structure() -> Structure:
    lattice = Lattice.cubic(5.43)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def test_structure_to_graph_shape():
    structure = _make_si_structure()
    graph = structure_to_graph(structure, band_gap=1.12)

    assert graph.x.shape[0] == 2  # 2 atoms
    assert graph.x.shape[1] == 102  # 100 one-hot + 2 continuous
    assert graph.y.item() == 1.12
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 4  # distance + 3D unit vector


def test_structure_to_graph_has_edges():
    structure = _make_si_structure()
    graph = structure_to_graph(structure, band_gap=0.0)
    assert graph.edge_index.shape[1] > 0, "Si diamond should have neighbors within 8A"


def test_sample_dataset_loads():
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample" / "materials.json"
    if not sample_path.exists():
        from data.generate_sample import generate_sample
        data = generate_sample(n=10)
    else:
        data = json.loads(sample_path.read_text())[:10]

    assert len(data) >= 10
    for rec in data[:3]:
        assert "material_id" in rec
        assert "structure" in rec
        assert "band_gap" in rec
        structure = Structure.from_dict(rec["structure"])
        graph = structure_to_graph(structure, rec["band_gap"])
        assert graph.x.shape[0] == len(structure)
