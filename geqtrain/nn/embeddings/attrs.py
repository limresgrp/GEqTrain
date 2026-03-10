from typing import Optional

import torch
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn._graph_mixin import GraphModuleMixin
from geqtrain.nn.embeddings.edge import BaseEdgeEmbedding, BaseEdgeEqEmbedding
from geqtrain.nn.embeddings.node import BaseNodeEmbedding, BaseNodeEqEmbedding


@compile_mode("script")
class EmbeddingAttrs(GraphModuleMixin, torch.nn.Module):
    num_types: int

    def __init__(
        self,
        node_emb=BaseNodeEmbedding,
        node_emb_kwargs=None,
        node_eq_emb=BaseNodeEqEmbedding,
        node_eq_emb_kwargs=None,
        edge_emb=BaseEdgeEmbedding,
        edge_emb_kwargs=None,
        edge_eq_emb=BaseEdgeEqEmbedding,
        edge_eq_emb_kwargs=None,
        node_out_irreps: Optional[str] = None,
        node_eq_out_irreps: Optional[str] = None,
        edge_out_irreps: Optional[str] = None,
        edge_eq_out_irreps: Optional[str] = None,
        irreps_in=None,
    ):
        super().__init__()
        node_emb_kwargs = dict(node_emb_kwargs or {})
        node_eq_emb_kwargs = dict(node_eq_emb_kwargs or {})
        edge_emb_kwargs = dict(edge_emb_kwargs or {})
        edge_eq_emb_kwargs = dict(edge_eq_emb_kwargs or {})

        if node_out_irreps is not None:
            node_out_irreps = str(Irreps(node_out_irreps))
        if node_eq_out_irreps is not None:
            node_eq_out_irreps = str(Irreps(node_eq_out_irreps))
        if edge_out_irreps is not None:
            edge_out_irreps = str(Irreps(edge_out_irreps))
        if edge_eq_out_irreps is not None:
            edge_eq_out_irreps = str(Irreps(edge_eq_out_irreps))

        if node_emb is None and node_out_irreps is not None:
            node_emb = BaseNodeEmbedding
        if node_eq_emb is None and node_eq_out_irreps is not None:
            node_eq_emb = BaseNodeEqEmbedding
        if edge_emb is None and edge_out_irreps is not None:
            edge_emb = BaseEdgeEmbedding
        if edge_eq_emb is None and edge_eq_out_irreps is not None:
            edge_eq_emb = BaseEdgeEqEmbedding

        if node_emb is BaseNodeEmbedding:
            node_emb_kwargs.setdefault("out_irreps", node_out_irreps)
        if node_eq_emb is BaseNodeEqEmbedding:
            node_eq_emb_kwargs.setdefault("out_irreps", node_eq_out_irreps)
        if edge_emb is BaseEdgeEmbedding:
            edge_emb_kwargs.setdefault("out_irreps", edge_out_irreps)
        if edge_eq_emb is BaseEdgeEqEmbedding:
            edge_eq_emb_kwargs.setdefault("out_irreps", edge_eq_out_irreps)

        self.node_field = AtomicDataDict.NODE_INPUT_ATTRS_KEY
        self.node_eq_field = AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY
        self.edge_field = AtomicDataDict.EDGE_INPUT_ATTRS_KEY
        self.edge_eq_field = AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY
        self.node_out_field = AtomicDataDict.NODE_ATTRS_KEY
        self.node_eq_out_field = AtomicDataDict.NODE_EQ_ATTRS_KEY
        self.edge_out_field = AtomicDataDict.EDGE_ATTRS_KEY
        self.edge_eq_out_field = AtomicDataDict.EDGE_EQ_ATTRS_KEY

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_RADIAL_EMB_KEY,
                AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
            ],
        )
        assert self.node_field in self.irreps_in

        irreps_out = self.irreps_in.copy()

        self.node_emb = (
            node_emb(
                node_field=self.node_field,
                node_eq_field=self.node_eq_field,
                edge_field=self.edge_field,
                edge_eq_field=self.edge_eq_field,
                irreps_in=irreps_out,
                **node_emb_kwargs,
            )
            if node_emb is not None
            else None
        )
        if self.node_emb is None:
            node_irreps_out = self.irreps_in[self.node_field]
        else:
            node_irreps_out = self.node_emb.out_irreps
        if node_irreps_out is None:
            raise ValueError("Node embedding produced no node scalar attributes.")
        irreps_out[self.node_out_field] = node_irreps_out

        self.node_eq_emb = (
            node_eq_emb(
                node_field=self.node_out_field,
                node_eq_field=self.node_eq_field,
                edge_field=self.edge_field,
                edge_eq_field=self.edge_eq_field,
                irreps_in=irreps_out,
                **node_eq_emb_kwargs,
            )
            if node_eq_emb is not None
            else None
        )
        if self.node_eq_emb is None:
            if self.node_eq_field in self.irreps_in:
                irreps_out[self.node_eq_out_field] = self.irreps_in[self.node_eq_field]
        elif self.node_eq_emb.out_irreps is not None:
            irreps_out[self.node_eq_out_field] = self.node_eq_emb.out_irreps

        self.edge_emb = (
            edge_emb(
                node_field=self.node_out_field,
                node_eq_field=self.node_eq_out_field,
                edge_field=self.edge_field,
                edge_eq_field=self.edge_eq_field,
                irreps_in=irreps_out,
                **edge_emb_kwargs,
            )
            if edge_emb is not None
            else None
        )
        if self.edge_emb is None:
            edge_irreps_out = self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        else:
            edge_irreps_out = self.edge_emb.out_irreps
        if edge_irreps_out is not None:
            irreps_out[self.edge_out_field] = edge_irreps_out

        self.edge_eq_emb = (
            edge_eq_emb(
                node_field=self.node_out_field,
                node_eq_field=self.node_eq_out_field,
                edge_field=self.edge_out_field,
                edge_eq_field=self.edge_eq_field,
                irreps_in=irreps_out,
                **edge_eq_emb_kwargs,
            )
            if edge_eq_emb is not None
            else None
        )
        if self.edge_eq_emb is not None and self.edge_eq_emb.out_irreps is not None:
            irreps_out[self.edge_eq_out_field] = self.edge_eq_emb.out_irreps
        self.irreps_out.update(irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.node_emb is None:
            node_attr: torch.Tensor = data[self.node_field]
        else:
            node_attr = self.node_emb(data)
            if node_attr is None:
                node_attr = data[self.node_field]
        data[self.node_out_field] = node_attr

        if self.node_eq_emb is not None:
            node_eq_attr = self.node_eq_emb(data)
            if node_eq_attr is not None:
                data[self.node_eq_out_field] = node_eq_attr

        if self.edge_emb is not None:
            edge_attr = self.edge_emb(data)
            if edge_attr is not None:
                data[self.edge_out_field] = edge_attr

        if self.edge_eq_emb is not None:
            edge_eq_attr = self.edge_eq_emb(data)
            if edge_eq_attr is not None:
                data[self.edge_eq_out_field] = edge_eq_attr

        return data
