# src/models/memory_store.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    """
    A single memory entry.
    key:   vector used for retrieval (e.g., hidden state)
    value: stored content (can be same as key for now)
    age:   how many steps since written
    usage: how many times retrieved
    """
    key: torch.Tensor
    value: torch.Tensor
    age: int = 0
    usage: int = 0


class MemoryStore(nn.Module):
    """
    Simple episodic memory store implemented as a fixed-size buffer.
    - Stores MemoryEntry objects
    - Supports add() and retrieve_topk()
    """

    def __init__(self, max_entries: int = 256):
        super().__init__()
        self.max_entries = max_entries
        # We keep entries in a Python list; actual tensors live on device.
        self.entries: list[MemoryEntry] = []

    def reset(self) -> None:
        """Clear all memory."""
        self.entries = []

    @torch.no_grad()
    def step_ages(self) -> None:
        """Increase age of all entries by 1."""
        for entry in self.entries:
            entry.age += 1

    def add(self, key: torch.Tensor, value: Optional[torch.Tensor] = None) -> None:
        """
        Add a new memory entry.
        key:   (d_model,)
        value: (d_model,) or None (defaults to key)
        """
        if value is None:
            value = key

        if key.dim() != 1:
            raise ValueError(f"key must be 1D (d_model,), got shape {key.shape}")
        if value.dim() != 1:
            raise ValueError(f"value must be 1D (d_model,), got shape {value.shape}")

        if len(self.entries) >= self.max_entries:
            # Simple FIFO eviction: pop the oldest (index 0).
            self.entries.pop(0)

        self.entries.append(MemoryEntry(key=key.detach(), value=value.detach()))

    def retrieve_topk(
        self,
        query: torch.Tensor,
        k: int = 4,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k entries by cosine similarity with query.

        query: (d_model,)
        Returns:
          values: (k, d_model) aggregated memory values
          sims:   (k,) cosine similarities
        If memory is empty, returns zeros.
        """
        if device is None:
            device = query.device

        if len(self.entries) == 0:
            return (
                torch.zeros(k, query.size(0), device=device),
                torch.zeros(k, device=device),
            )

        keys = torch.stack([e.key.to(device) for e in self.entries], dim=0)  # (N, d)
        values = torch.stack([e.value.to(device) for e in self.entries], dim=0)  # (N, d)

        # Cosine similarity between query and all keys
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)  # (1, d)
        key_norm = F.normalize(keys, dim=-1)  # (N, d)
        sims = torch.sum(query_norm * key_norm, dim=-1)  # (N,)

        k = min(k, sims.size(0))
        topk_sims, topk_idx = torch.topk(sims, k=k, dim=-1)

        topk_values = values[topk_idx]  # (k, d)

        # Update usage counts
        for idx in topk_idx.tolist():
            self.entries[idx].usage += 1

        return topk_values, topk_sims
