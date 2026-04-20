import math

import torch
from e3nn.util.jit import compile_mode


@compile_mode("script")
class CategoricalPositionalEmbedding(torch.nn.Module):
    """Fixed sinusoidal embeddings indexed by categorical ids."""

    num_types: int
    embedding_dim: int
    base: float

    def __init__(self, num_types: int, embedding_dim: int, base: float = 10000.0):
        super().__init__()
        if num_types <= 0:
            raise ValueError(f"num_types must be > 0, got {num_types}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if base <= 0.0:
            raise ValueError(f"base must be > 0, got {base}")

        self.num_types = int(num_types)
        self.embedding_dim = int(embedding_dim)
        self.base = float(base)

        # Build a deterministic sinusoidal table with support for odd dimensions.
        positions = torch.arange(self.num_types, dtype=torch.float32).unsqueeze(1)  # [num_types, 1]
        freq_count = (self.embedding_dim + 1) // 2
        freqs = torch.exp(
            -math.log(self.base) * torch.arange(freq_count, dtype=torch.float32) / max(freq_count, 1)
        )  # [freq_count]
        angles = positions * freqs.unsqueeze(0)  # [num_types, freq_count]
        table = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)[:, : self.embedding_dim]
        self.register_buffer("table", table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.table[x.to(dtype=torch.long)]
