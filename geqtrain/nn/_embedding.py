import torch
import torch.nn
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from typing import Optional


@compile_mode("script")
class BaseEmbedding(GraphModuleMixin, torch.nn.Module):
    node_field         : Optional[str]    = None
    node_eq_field      : Optional[str]    = None
    edge_field         : Optional[str]    = None
    edge_eq_field      : Optional[str]    = None

    def __init__(
        self,
        node_field   : Optional[str] = None,
        node_eq_field: Optional[str] = None,
        edge_field   : Optional[str] = None,
        edge_eq_field: Optional[str] = None,
        # other
        irreps_in = None,
    ):
        super().__init__()
        self.node_field    = node_field
        self.node_eq_field = node_eq_field
        self.edge_field    = edge_field
        self.edge_eq_field = edge_eq_field

        self._init_irreps(irreps_in=irreps_in)

    def forward(
        self,
        data: AtomicDataDict.Type,
    ) -> torch.Tensor:
        raise NotImplementedError()