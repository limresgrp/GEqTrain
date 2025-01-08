from typing import Dict
import torch
import torch.nn
import math
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin


@compile_mode("script")
class EmbeddingGraphAttrs(GraphModuleMixin, torch.nn.Module):
    """Select the graph embedding based on graph input features.

    Args:
        embedding_dim (int): Dimension of the node attribute embedding tensor.
    """

    def __init__(
        self,
        graph_attributes: Dict[str, Dict] = {},
        irreps_in=None,
    ):
        super().__init__()

        irreps_out = {}
        attr_modules = torch.nn.ModuleDict() # k: str field name, v: nn.Embedding layer
        output_embedding_dim = 0
        for field, values in graph_attributes.items():
            if 'embedding_dimensionality' not in values: # this means the attr is not used as embedding
                continue
            n_types = values.get('actual_num_types')
            embedding_dim = values['embedding_dimensionality']
            emb_module = torch.nn.Embedding(n_types, embedding_dim)
            torch.nn.init.normal_(emb_module.weight, mean=0, std=math.isqrt(embedding_dim))

            attr_modules[field] = emb_module
            output_embedding_dim += embedding_dim

        self.attr_modules = attr_modules
        if output_embedding_dim:
            irreps_out = {AtomicDataDict.GRAPH_ATTRS_KEY: Irreps([(output_embedding_dim, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not self.attr_modules: return data
        out = []
        for attribute_name, emb_layer in self.attr_modules.items():
            x = data[attribute_name].squeeze(-1)
            x = emb_layer(x)
            out.append(x)

        data[AtomicDataDict.GRAPH_ATTRS_KEY] = torch.cat(out, dim=-1)
        return data