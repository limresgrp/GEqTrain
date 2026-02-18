from typing import Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn._embedding import BaseEmbedding
from geqtrain.nn.so3 import SO3_Linear


@compile_mode("script")
class BaseNodeEmbedding(BaseEmbedding):
    def __init__(
        self,
        include_node_field: bool = True,
        out_irreps: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_node_field = bool(include_node_field)
        self.has_node_attr = False
        self._use_node_field = False
        self.projection = None

        node_dim = 0
        if (
            self.include_node_field
            and self.node_field is not None
            and self.node_field in self.irreps_in
        ):
            self._use_node_field = True
            self.has_node_attr = True
            node_dim += self.irreps_in[self.node_field].dim

        if not self.has_node_attr:
            self.out_irreps = None
            return

        target_irreps = (
            o3.Irreps(out_irreps) if out_irreps is not None else o3.Irreps(f"{node_dim}x0e")
        )
        if any(ir.l != 0 for _, ir in target_irreps):
            raise ValueError(
                f"BaseNodeEmbedding only supports invariant outputs, got out_irreps={target_irreps}"
            )
        if target_irreps.dim != node_dim:
            self.projection = torch.nn.Linear(node_dim, target_irreps.dim, bias=True)
            torch.nn.init.xavier_uniform_(self.projection.weight)
        self.out_irreps = target_irreps

    def forward(self, data: AtomicDataDict.Type) -> Optional[torch.Tensor]:
        if not self.has_node_attr:
            return None
        features = []
        if self._use_node_field:
            node_field = torch.jit._unwrap_optional(self.node_field)
            features.append(data[node_field])
        if len(features) == 1:
            out = features[0]
        else:
            out = torch.cat(features, dim=-1)
        if self.projection is not None:
            out = self.projection(out)
        return out


@compile_mode("script")
class BaseNodeEqEmbedding(BaseEmbedding):
    def __init__(
        self,
        include_node_eq_field: bool = True,
        out_irreps: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_node_eq_field = bool(include_node_eq_field)
        self.has_node_eq_attr = False
        self._use_node_eq_field = False
        self.projection = None

        node_eq_irreps = None
        if (
            self.include_node_eq_field
            and self.node_eq_field is not None
            and self.node_eq_field in self.irreps_in
        ):
            self._use_node_eq_field = True
            self.has_node_eq_attr = True
            node_eq_irreps = self.irreps_in[self.node_eq_field]

        if not self.has_node_eq_attr:
            self.out_irreps = None
            return

        assert node_eq_irreps is not None
        target_irreps = o3.Irreps(out_irreps) if out_irreps is not None else node_eq_irreps
        if target_irreps != node_eq_irreps:
            self.projection = SO3_Linear(
                in_irreps=node_eq_irreps,
                out_irreps=target_irreps,
                internal_weights=True,
                shared_weights=True,
                pad_to_alignment=1,
            )
        self.out_irreps = target_irreps

    def forward(self, data: AtomicDataDict.Type) -> Optional[torch.Tensor]:
        if not self.has_node_eq_attr:
            return None
        node_eq_field = torch.jit._unwrap_optional(self.node_eq_field)
        node_eq = data[node_eq_field]
        if self.projection is not None:
            node_eq = self.projection(node_eq)
        return node_eq
