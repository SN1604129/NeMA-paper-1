# src/train_delayed_qa.py

from __future__ import annotations
import os
import csv
import argparse
from datetime import datetime
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
    mem_lambda: float,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        avg_loss: float       – mean training loss over dataset
        avg_write_ratio: float – total_writes / total_tokens
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    # track write statistics
    total_writes = 0
    total_tokens = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # Reset memory every batch for now (simple)
        model.reset_memory()
        logits = model(x, use_memory=True)

        # task loss
        loss = criterion(logits, y)

        # memory usage penalty: encourage smaller write probabilities
        if getattr(model, "last_write_prob_mean", None) is not None:
            loss = loss + mem_lambda * model.last_write_prob_mean

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        # accumulate write stats from this batch
        if (
            getattr(model, "last_write_count", None) is not None
            and getattr(model, "last_token_count", None) is not None
        ):
            total_writes += model.last_write_count
            total_tokens += model.last_token_count

    avg_write_ratio = total_writes / max(total_tokens, 1)
    avg_loss = total_loss / len(loader.dataset)

    print(f"  Avg write ratio this epoch: {avg_write_ratio:.4f}")

    return avg_loss, avg_write_ratio


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


def parse_args():
    parser = argparse.ArgumentParser()

    # training hyperparams
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    # memory-related
    parser.add_argument("--mem_lambda", type=float, default=0.1)
    parser.add_argument("--write_threshold", type=float, default=0.5)
    parser.add_argument("--mem_max_entries", type=int, default=256)

    # logging
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="results")

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Set up run id & logging path
    if args.run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"run_{timestamp}"

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{args.run_id}.csv")

    # Data
    train_ds = DelayedQADataset(
        num_samples=5000,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=123,
    )
    val_ds = DelayedQADataset(
        num_samples=1000,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=456,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = SimpleTransformerWithMemory(
        vocab_size=args.vocab_size,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        max_seq_len=args.seq_len,
        num_classes=args.vocab_size,
        mem_max_entries=args.mem_max_entries,
        write_threshold=args.write_threshold,  # now configurable
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Prepare CSV logging
    header = [
        "run_id",
        "epoch",
        "mem_lambda",
        "write_threshold",
        "avg_write_ratio",
        "train_loss",
        "val_loss_mem",
        "val_acc_mem",
        "val_loss_nomem",
        "val_acc_nomem",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        train_loss, avg_write_ratio = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mem_lambda=args.mem_lambda,
        )
        print(f"Train loss: {train_loss:.4f}")

        val_loss_mem, val_acc_mem = eval_epoch(
            model, val_loader, device, use_memory=True
        )
        val_loss_nomem, val_acc_nomem = eval_epoch(
            model, val_loader, device, use_memory=False
        )

        print(f"Val (with mem):   loss={val_loss_mem:.4f}, acc={val_acc_mem:.4f}")
        print(f"Val (no memory):  loss={val_loss_nomem:.4f}, acc={val_acc_nomem:.4f}")

        # Append row to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    args.run_id,
                    epoch,
                    args.mem_lambda,
                    args.write_threshold,
                    avg_write_ratio,
                    train_loss,
                    val_loss_mem,
                    val_acc_mem,
                    val_loss_nomem,
                    val_acc_nomem,
                ]
            )

    # Save model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{args.run_id}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
