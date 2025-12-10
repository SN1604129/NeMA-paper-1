# src/tasks/synthetic_delayed_qa.py

from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import Dataset


class DelayedQADataset(Dataset):
    """
    Synthetic "delayed QA" task:
    - Sequence of random digits [0..9]
    - Choose one index i in the first half of the sequence
    - We treat token at pos 0 as CLS
    - The target is the digit at index i

    For now, input is just the raw sequence of digits (no explicit "question" token),
    but the task still requires long-range memory.
    """

    def __init__(
        self,
        num_samples: int = 10_000,
        seq_len: int = 64,
        vocab_size: int = 10,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        g = torch.Generator()
        g.manual_seed(seed)

        # Pre-generate the data for simplicity
        self.inputs = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_len),
            generator=g,
        )
        # Use index 0 as special CLS token always = 0
        self.inputs[:, 0] = 0

        # Random positions to query (avoid index 0)
        idx_low = 1
        idx_high = seq_len // 2
        self.query_positions = torch.randint(
            low=idx_low,
            high=idx_high,
            size=(num_samples,),
            generator=g,
        )

        # Label is the digit at that position
        self.labels = self.inputs[torch.arange(num_samples), self.query_positions]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]           # (seq_len,)
        y = self.labels[idx]           # scalar
        return x.long(), y.long()
