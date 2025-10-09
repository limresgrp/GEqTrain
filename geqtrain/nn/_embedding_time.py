import math
from typing import Optional
import torch
from torch import nn
from e3nn.util.jit import compile_mode


@compile_mode("script")
class PositionalEmbedding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create a long enough positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [1] of integer positions
            num_nodes: If provided, expand the embedding to this size.
        """
        embedding = self.pe[x.view(-1)]
        if num_nodes is not None:
            # Expand to (num_nodes, d_model)
            return embedding.expand(num_nodes, -1)
        return embedding