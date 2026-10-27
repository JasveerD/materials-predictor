"""Training loop for the crystal GNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from model.gnn import CrystalGNN


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "processed"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    hidden_dim: int = 128,
    num_conv_layers: int = 4,
    dropout: float = 0.2,
) -> Path:
    """Train the GNN and return the path to the best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if train_data[0].edge_attr.numel() else 4

    model = CrystalGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_conv_layers=num_conv_layers,
        dropout=dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=1.0)

    writer = SummaryWriter(log_dir=ckpt_dir.parent / "runs")

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    best_ckpt = ckpt_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_data)
        scheduler.step()

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_mae += (pred - batch.y).abs().sum().item()
        val_mae /= len(val_data)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} eV")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
                "config": {
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "hidden_dim": hidden_dim,
                    "num_conv_layers": num_conv_layers,
                    "dropout": dropout,
                },
            }, best_ckpt)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (best val MAE: {best_val_mae:.4f} eV)")
                break

    writer.close()
    print(f"Best checkpoint saved to {best_ckpt} (MAE: {best_val_mae:.4f} eV)")
    return best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crystal GNN")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
