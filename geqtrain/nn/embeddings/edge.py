from typing import Optional, Union

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn._embedding import BaseEmbedding
from geqtrain.nn.so3 import SO3_Linear
from geqtrain.utils._model_utils import build_concatenation_permutation


@compile_mode("script")
class BaseEdgeEmbedding(BaseEmbedding):
    def __init__(
        self,
        include_edge_field: bool = True,
        include_edge_radial: bool = False,
        out_irreps: Optional[Union[str, o3.Irreps]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_edge_field = bool(include_edge_field)
        self.include_edge_radial = bool(include_edge_radial)
        self.has_edge_attr = False
        self._use_edge_field = False
        self._use_edge_radial = False
        self.projection = None

        edge_dim = 0
        if (
            self.include_edge_field
            and self.edge_field is not None
            and self.edge_field in self.irreps_in
        ):
            self._use_edge_field = True
            self.has_edge_attr = True
            edge_dim += self.irreps_in[self.edge_field].dim

        if self.include_edge_radial and AtomicDataDict.EDGE_RADIAL_EMB_KEY in self.irreps_in:
            self._use_edge_radial = True
            self.has_edge_attr = True
            edge_dim += self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY].dim

        if not self.has_edge_attr:
            self.out_irreps = None
            return

        target_irreps = (
            o3.Irreps(out_irreps) if out_irreps is not None else o3.Irreps(f"{edge_dim}x0e")
        )
        if any(ir.l != 0 for _, ir in target_irreps):
            raise ValueError(
                f"BaseEdgeEmbedding only supports invariant outputs, got out_irreps={target_irreps}"
            )
        if target_irreps.dim != edge_dim:
            self.projection = torch.nn.Linear(edge_dim, target_irreps.dim, bias=True)
            torch.nn.init.xavier_uniform_(self.projection.weight)
        self.out_irreps = target_irreps

    def forward(self, data: AtomicDataDict.Type) -> Optional[torch.Tensor]:
        if not self.has_edge_attr:
            return None
        features = []
        if self._use_edge_field:
            edge_field = torch.jit._unwrap_optional(self.edge_field)
            features.append(data[edge_field])
        if self._use_edge_radial:
            features.append(data[AtomicDataDict.EDGE_RADIAL_EMB_KEY])

        if len(features) == 1:
            out = features[0]
        else:
            out = torch.cat(features, dim=-1)
        if self.projection is not None:
            out = self.projection(out)
        return out


@compile_mode("script")
class BaseEdgeEqEmbedding(BaseEmbedding):
    def __init__(
        self,
        include_edge_eq_field: bool = True,
        include_edge_spharm: bool = False,
        include_node_eq_center: bool = False,
        include_node_eq_neighbor: bool = False,
        out_irreps: Optional[Union[str, o3.Irreps]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_edge_eq_field = bool(include_edge_eq_field)
        self.include_edge_spharm = bool(include_edge_spharm)
        self.include_node_eq_center = bool(include_node_eq_center)
        self.include_node_eq_neighbor = bool(include_node_eq_neighbor)
        self.has_edge_eq_attr = False
        self._use_edge_eq_field = False
        self._use_edge_spharm = False
        self._use_node_eq_center = False
        self._use_node_eq_neighbor = False
        self.projection = None

        irreps_to_concat = []
        if (
            self.include_edge_eq_field
            and self.edge_eq_field is not None
            and self.edge_eq_field in self.irreps_in
        ):
            self._use_edge_eq_field = True
            self.has_edge_eq_attr = True
            irreps_to_concat.append(self.irreps_in[self.edge_eq_field])

        if self.include_edge_spharm and AtomicDataDict.EDGE_SPHARMS_EMB_KEY in self.irreps_in:
            self._use_edge_spharm = True
            self.has_edge_eq_attr = True
            irreps_to_concat.append(self.irreps_in[AtomicDataDict.EDGE_SPHARMS_EMB_KEY])

        if (
            self.include_node_eq_center
            and self.node_eq_field is not None
            and self.node_eq_field in self.irreps_in
        ):
            self._use_node_eq_center = True
            self.has_edge_eq_attr = True
            irreps_to_concat.append(self.irreps_in[self.node_eq_field])

        if (
            self.include_node_eq_neighbor
            and self.node_eq_field is not None
            and self.node_eq_field in self.irreps_in
        ):
            self._use_node_eq_neighbor = True
            self.has_edge_eq_attr = True
            irreps_to_concat.append(self.irreps_in[self.node_eq_field])

        if self.has_edge_eq_attr:
            permutation, sorted_irreps = build_concatenation_permutation(irreps_to_concat)
            if permutation is not None:
                self.register_buffer("concatenation_permutation", permutation)
            else:
                self.concatenation_permutation = None
            target_irreps = o3.Irreps(out_irreps) if out_irreps is not None else sorted_irreps
            if target_irreps.dim != sorted_irreps.dim or target_irreps != sorted_irreps:
                self.projection = SO3_Linear(
                    in_irreps=sorted_irreps,
                    out_irreps=target_irreps,
                    internal_weights=True,
                    shared_weights=True,
                    pad_to_alignment=1,
                )
            edge_eq_irreps = target_irreps
        else:
            self.concatenation_permutation = None
            edge_eq_irreps = None
        self.out_irreps = edge_eq_irreps

    def forward(self, data: AtomicDataDict.Type) -> Optional[torch.Tensor]:
        if not self.has_edge_eq_attr:
            return None
        to_concat = []

        if self._use_edge_eq_field:
            edge_eq_field = torch.jit._unwrap_optional(self.edge_eq_field)
            to_concat.append(data[edge_eq_field])
        if self._use_edge_spharm:
            to_concat.append(data[AtomicDataDict.EDGE_SPHARMS_EMB_KEY])
        if self._use_node_eq_center or self._use_node_eq_neighbor:
            node_eq_field = torch.jit._unwrap_optional(self.node_eq_field)
            node_eq = data[node_eq_field]
            edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
            edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
            if self._use_node_eq_center:
                to_concat.append(node_eq[edge_center])
            if self._use_node_eq_neighbor:
                to_concat.append(node_eq[edge_neighbor])

        if len(to_concat) == 1:
            edge_eq_attr = to_concat[0]
        else:
            edge_eq_attr = torch.cat(to_concat, dim=-1)
        if self.concatenation_permutation is not None:
            edge_eq_attr = edge_eq_attr[:, self.concatenation_permutation]
        if self.projection is not None:
            edge_eq_attr = self.projection(edge_eq_attr)
        return edge_eq_attr
