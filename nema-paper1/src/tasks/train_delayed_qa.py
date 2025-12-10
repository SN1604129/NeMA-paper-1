# src/train_delayed_qa.py

from __future__ import annotations
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer_wrapper import SimpleTransformerWithMemory
from tasks.synthetic_delayed_qa import DelayedQADataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # Reset memory every batch for now (simple)
        model.reset_memory()
        logits = model(x, use_memory=True)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_memory: bool = True,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    for x, y in tqdm(loader, desc=f"Eval (mem={use_memory})", leave=False):
        x = x.to(device)
        y = y.to(device)

        model.reset_memory()
        logits = model(x, use_memory=use_memory)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Hyperparameters
    vocab_size = 10
    seq_len = 64
    batch_size = 64
    num_epochs = 5
    lr = 1e-3

    # Data
    train_ds = DelayedQADataset(num_samples=5000, seq_len=seq_len, vocab_size=vocab_size, seed=123)
    val_ds = DelayedQADataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size, seed=456)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = SimpleTransformerWithMemory(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        max_seq_len=seq_len,
        num_classes=vocab_size,
        mem_max_entries=256,
        write_threshold=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        val_loss_mem, val_acc_mem = eval_epoch(model, val_loader, device, use_memory=True)
        val_loss_nomem, val_acc_nomem = eval_epoch(model, val_loader, device, use_memory=False)

        print(f"Val (with mem):   loss={val_loss_mem:.4f}, acc={val_acc_mem:.4f}")
        print(f"Val (no memory):  loss={val_loss_nomem:.4f}, acc={val_acc_nomem:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/simple_transformer_memory.pt")
    print("Model saved to checkpoints/simple_transformer_memory.pt")


if __name__ == "__main__":
    main()
