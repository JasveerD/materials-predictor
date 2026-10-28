"""Evaluate trained model: compute metrics and generate parity plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.loader import DataLoader

from model.gnn import CrystalGNN


def load_model(ckpt_path: Path, device: torch.device | None = None) -> CrystalGNN:
    """Load a trained model from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = CrystalGNN(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_metrics(
    model: CrystalGNN,
    data: list,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> dict[str, float]:
    """Compute MAE, RMSE, and R² on a dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(data, batch_size=batch_size)
    all_pred: list[float] = []
    all_true: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(batch.y.cpu().tolist())

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)

    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    ss_res = float(np.sum((pred_arr - true_arr) ** 2))
    ss_tot = float(np.sum((true_arr - np.mean(true_arr)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2}


def parity_plot(
    model: CrystalGNN,
    data: list,
    output_path: Path,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> Path:
    """Generate an interactive parity plot (predicted vs actual) and save as HTML."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(data, batch_size=batch_size)
    all_pred: list[float] = []
    all_true: list[float] = []
    formulas: list[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(batch.y.cpu().tolist())
            if hasattr(batch, "formula"):
                formulas.extend(batch.formula)

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)
    metrics = compute_metrics(model, data, device, batch_size)

    max_val = max(true_arr.max(), pred_arr.max()) * 1.1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=true_arr,
        y=pred_arr,
        mode="markers",
        marker=dict(size=5, opacity=0.5, color="steelblue"),
        text=formulas if formulas else None,
        hovertemplate="Actual: %{x:.2f} eV<br>Predicted: %{y:.2f} eV<extra>%{text}</extra>",
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Ideal",
    ))

    fig.update_layout(
        title=f"Band Gap Parity Plot (MAE={metrics['mae']:.3f} eV, R²={metrics['r2']:.3f})",
        xaxis_title="DFT Band Gap (eV)",
        yaxis_title="Predicted Band Gap (eV)",
        template="plotly_white",
        width=700,
        height=600,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Parity plot saved to {output_path}")
    return output_path


if __name__ == "__main__":
    ckpt = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"
    test_data = torch.load(
        Path(__file__).resolve().parent.parent / "data" / "processed" / "test.pt",
        weights_only=False,
    )

    model = load_model(ckpt)
    metrics = compute_metrics(model, test_data)
    print(f"Test MAE:  {metrics['mae']:.4f} eV")
    print(f"Test RMSE: {metrics['rmse']:.4f} eV")
    print(f"Test R²:   {metrics['r2']:.4f}")

    parity_plot(
        model, test_data,
        Path(__file__).resolve().parent.parent / "results" / "parity_plot.html",
    )
