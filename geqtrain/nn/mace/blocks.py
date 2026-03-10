###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional

import torch.nn.functional
from e3nn import o3
from e3nn.util.jit import compile_mode
from .symmetric_contraction import SymmetricContraction


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        sc: bool = False,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sc = sc

        # The symmetric contraction requires scalar features. If none are provided,
        # this block cannot produce any output, so we should handle this case.
        if node_feats_irreps.dim > 0 and any(ir.l == 0 for _, ir in node_feats_irreps):
            self.symmetric_contractions = SymmetricContraction(
                irreps_in=node_feats_irreps,
                irreps_out=target_irreps,
                correlation=correlation,
                num_elements=num_elements,
            )
        else:
            self.symmetric_contractions = None

        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        sc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.symmetric_contractions is None:
            # If there are no scalar features to contract, the result of the contraction is zero.
            node_feats = torch.zeros(node_feats.shape[0], self.linear.irreps_in.dim, device=node_feats.device, dtype=node_feats.dtype)
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)

        if self.sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)