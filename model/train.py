"""Chunked training loop for the crystal GNN.

Trains on ~500 data points at a time, saves after each chunk so you can
pause and resume whenever. Run `python -m model.train` repeatedly to
advance through chunks one at a time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from model.gnn import CrystalGNN


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _validate(model: CrystalGNN, val_loader: DataLoader, device: torch.device, n_val: int) -> float:
    model.eval()
    val_mae = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            val_mae += (pred - batch.y).abs().sum().item()
    return val_mae / n_val


def train_chunk(
    data_dir: Path = DEFAULT_DATA_DIR,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    chunk_size: int = 500,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    hidden_dim: int = 128,
    num_conv_layers: int = 4,
    dropout: float = 0.2,
) -> Path:
    """Train on the next chunk of data, then save and stop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if train_data[0].edge_attr.numel() else 4

    progress_path = ckpt_dir / "progress.json"
    resume_ckpt = ckpt_dir / "latest.pt"

    # Figure out where we left off
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        current_chunk = progress["next_chunk"]
        total_epochs = progress["total_epochs"]
        best_val_mae = progress["best_val_mae"]
        chunk_order = progress["chunk_order"]
    else:
        current_chunk = 0
        total_epochs = 0
        best_val_mae = float("inf")
        chunk_order = list(range(len(train_data)))
        random.seed(42)
        random.shuffle(chunk_order)

    n_total = len(train_data)
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    if current_chunk >= n_chunks:
        print(f"All {n_chunks} chunks done ({n_total} materials). Training complete!")
        print(f"Best val MAE: {best_val_mae:.4f} eV")
        return ckpt_dir / "best_model.pt"

    # Slice the current chunk
    start = current_chunk * chunk_size
    end = min(start + chunk_size, n_total)
    chunk_indices = chunk_order[start:end]
    chunk_data = [train_data[i] for i in chunk_indices]
    chunk_loader = DataLoader(chunk_data, batch_size=batch_size, shuffle=True)

    print(f"=== Chunk {current_chunk + 1}/{n_chunks} ({len(chunk_data)} materials, "
          f"{start}-{end} of {n_total}) ===")

    # Build or resume model
    model = CrystalGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_conv_layers=num_conv_layers,
        dropout=dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    if resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Resumed from checkpoint (prev val MAE: {ckpt.get('val_mae', '?')})")

    criterion = nn.HuberLoss(delta=1.0)
    writer = SummaryWriter(log_dir=ckpt_dir.parent / "runs")

    chunk_best_mae = float("inf")
    epochs_without_improvement = 0
    best_ckpt = ckpt_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in chunk_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(chunk_data)
        scheduler.step()

        val_mae = _validate(model, val_loader, device, len(val_data))
        global_epoch = total_epochs + epoch

        writer.add_scalar("Loss/train", train_loss, global_epoch)
        writer.add_scalar("MAE/val", val_mae, global_epoch)
        writer.add_scalar("Chunk", current_chunk + 1, global_epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} eV")

        if val_mae < chunk_best_mae:
            chunk_best_mae = val_mae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Stabilized at epoch {epoch} (val MAE: {chunk_best_mae:.4f} eV)")
                break

    writer.close()
    total_epochs += epoch

    # Save best model if this chunk improved overall
    if chunk_best_mae < best_val_mae:
        best_val_mae = chunk_best_mae
        torch.save({
            "model_state_dict": model.state_dict(),
            "val_mae": best_val_mae,
            "chunk": current_chunk + 1,
            "total_epochs": total_epochs,
            "config": {
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "hidden_dim": hidden_dim,
                "num_conv_layers": num_conv_layers,
                "dropout": dropout,
            },
        }, best_ckpt)
        print(f"  New best model! Val MAE: {best_val_mae:.4f} eV -> {best_ckpt.name}")

    # Always save latest for resume
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_mae": chunk_best_mae,
        "chunk": current_chunk + 1,
        "total_epochs": total_epochs,
        "config": {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "hidden_dim": hidden_dim,
            "num_conv_layers": num_conv_layers,
            "dropout": dropout,
        },
    }, resume_ckpt)

    # Save progress
    progress = {
        "next_chunk": current_chunk + 1,
        "total_epochs": total_epochs,
        "best_val_mae": best_val_mae,
        "chunk_order": chunk_order,
        "n_total": n_total,
        "chunk_size": chunk_size,
    }
    progress_path.write_text(json.dumps(progress, indent=2))

    remaining = n_chunks - (current_chunk + 1)
    print(f"\nChunk {current_chunk + 1}/{n_chunks} done. "
          f"Best MAE so far: {best_val_mae:.4f} eV")
    if remaining > 0:
        print(f"{remaining} chunks remaining. Run `python -m model.train` again for the next one.")
    else:
        print("All chunks done! Training complete.")

    return best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crystal GNN (one chunk at a time)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--reset", action="store_true", help="Start over from chunk 1")
    args = parser.parse_args()

    if args.reset:
        p = args.ckpt_dir / "progress.json"
        if p.exists():
            p.unlink()
            print("Progress reset. Starting from chunk 1.")

    train_chunk(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
