# src/models/write_gate.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class WriteGate(nn.Module):
    """
    A lightweight differentiable write gate.
    Given a hidden state h_t (batch, d_model) or (d_model,),
    outputs a probability g_t in [0, 1] for writing to memory.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (..., d_model)
        returns: (...,) write probabilities in [0, 1]
        """
        x = F.relu(self.fc1(h))
        logits = self.fc2(x).squeeze(-1)  # (...,)
        g = torch.sigmoid(logits)
        return g
