"""Inference: predict band gap for a new crystal structure with uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from pymatgen.core import Structure

from data.process import structure_to_graph
from model.evaluate import load_model
from model.gnn import CrystalGNN


@dataclass
class Prediction:
    band_gap: float
    uncertainty: float
    classification: str


def _enable_dropout(model: CrystalGNN) -> None:
    """Enable dropout layers during inference for MC dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def predict_structure(
    model: CrystalGNN,
    structure: Structure,
    device: torch.device | None = None,
    mc_samples: int = 5,
) -> Prediction:
    """Predict band gap for a single crystal structure.

    Uses MC dropout for uncertainty estimation: runs multiple forward passes
    with dropout enabled and reports mean ± std.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = structure_to_graph(structure, band_gap=0.0)
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    graph = graph.to(device)

    model.eval()
    _enable_dropout(model)

    preds: list[float] = []
    with torch.no_grad():
        for _ in range(mc_samples):
            pred = model(graph)
            preds.append(pred.item())

    mean_pred = sum(preds) / len(preds)
    std_pred = (sum((p - mean_pred) ** 2 for p in preds) / len(preds)) ** 0.5

    band_gap = max(0.0, mean_pred)

    if band_gap < 0.01:
        classification = "metal"
    elif band_gap < 2.0:
        classification = "semiconductor"
    else:
        classification = "insulator"

    return Prediction(
        band_gap=round(band_gap, 4),
        uncertainty=round(std_pred, 4),
        classification=classification,
    )


def predict_from_cif(
    model: CrystalGNN,
    cif_string: str,
    device: torch.device | None = None,
    mc_samples: int = 5,
) -> Prediction:
    """Predict band gap from a CIF file string."""
    structure = Structure.from_str(cif_string, fmt="cif")
    return predict_structure(model, structure, device, mc_samples)


if __name__ == "__main__":
    ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"
    model = load_model(ckpt)

    structure = Structure.from_spacegroup(
        "Fd-3m", lattice={"a": 5.43}, species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    result = predict_structure(model, structure)
    print(f"Predicted band gap: {result.band_gap:.4f} ± {result.uncertainty:.4f} eV")
    print(f"Classification: {result.classification}")
