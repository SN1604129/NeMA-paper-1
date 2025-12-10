# src/models/transformer_wrapper.py

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_store import MemoryStore
from .write_gate import WriteGate


class SimpleTransformerWithMemory(nn.Module):
    """
    A small Transformer encoder with:
      - token embeddings
      - positional embeddings
      - external MemoryStore
      - WriteGate controlling what to store
      - retrieval at the end to augment CLS representation

    This is intentionally small and simple for Paper 1 experiments.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_seq_len: int = 512,
        num_classes: int = 10,
        mem_max_entries: int = 256,
        write_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.write_threshold = write_threshold

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.write_gate = WriteGate(d_model=d_model, hidden_dim=d_model)
        self.memory = MemoryStore(max_entries=mem_max_entries)

        # Classifier on top of [CLS] with optional memory augmentation
        self.classifier = nn.Linear(d_model, num_classes)

    def reset_memory(self) -> None:
        self.memory.reset()

    def forward(
        self,
        input_ids: torch.Tensor,
        use_memory: bool = True,
    ) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        Returns logits: (batch, num_classes)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} > max_seq_len {self.max_seq_len}")

        # Positional indices
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)  # (batch, seq_len, d_model)
        h = self.transformer(x)  # (batch, seq_len, d_model)

        # We'll take CLS = position 0
        cls_state = h[:, 0, :]  # (batch, d_model)

        if use_memory:
            # 1) WRITE: for each time-step, decide which positions to store
            # We'll flatten across batch for gating, then iterate per item.
            flat_h = h.reshape(-1, self.d_model)  # (batch * seq_len, d_model)
            write_probs = self.write_gate(flat_h)  # (batch * seq_len,)
            write_probs = write_probs.reshape(batch_size, seq_len)  # (batch, seq_len)

            # Use a simple hard threshold. For more advanced training,
            # you can use straight-through estimators or relaxations.
            with torch.no_grad():
                write_mask = write_probs > self.write_threshold  # (batch, seq_len)

            # For now, we share a single MemoryStore across batch (simple).
            # More advanced: per-instance memory.
            for b in range(batch_size):
                for t in range(seq_len):
                    if write_mask[b, t]:
                        self.memory.add(
                            key=h[b, t].detach().cpu(),  # store on CPU to avoid GPU memory blow-up
                            value=h[b, t].detach().cpu(),
                        )

            # 2) READ: retrieve from memory using CLS as query
            # For now: we retrieve per sample, then average augmented state
            augmented_cls = []
            for b in range(batch_size):
                query = cls_state[b].detach().cpu()
                values, sims = self.memory.retrieve_topk(query=query, k=4)
                # Move back to device
                values = values.to(device)
                sims = sims.to(device)

                if sims.sum().abs() < 1e-6:
                    mem_vec = torch.zeros(self.d_model, device=device)
                else:
                    attn = F.softmax(sims, dim=-1)  # (k,)
                    mem_vec = torch.sum(attn.unsqueeze(-1) * values, dim=0)  # (d_model,)

                augmented_cls.append(cls_state[b] + mem_vec)

            cls_state = torch.stack(augmented_cls, dim=0)  # (batch, d_model)

        logits = self.classifier(cls_state)  # (batch, num_classes)
        return logits
