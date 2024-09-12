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
        attributes_to_embed = {} # k: str field name, v: nn.Embedding layer name
        output_embedding_dim = 0
        for field, values in graph_attributes.items():
            if 'embedding_dimensionality' not in values: # this means the attr is not used as embedding
                continue
            emb_layer_name = f"{field}_embedding"
            attributes_to_embed[field] = emb_layer_name
            n_types = values.get('num_types') + 1
            embedding_dim = values['embedding_dimensionality']
            setattr(self, emb_layer_name, torch.nn.Embedding(n_types, embedding_dim))
            torch.nn.init.normal_(getattr(self, emb_layer_name).weight, mean=0, std=math.isqrt(embedding_dim))
            output_embedding_dim += embedding_dim

        self.attributes_to_embed = attributes_to_embed
        if output_embedding_dim:
            irreps_out = {AtomicDataDict.GRAPH_ATTRS_KEY: Irreps([(output_embedding_dim, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not self.attributes_to_embed: return data
        out = []
        for attribute_name, emb_layer_name in self.attributes_to_embed.items():
            x = data[attribute_name].squeeze()
            x = getattr(self, emb_layer_name)(x)
            out.append(x)

        data[AtomicDataDict.GRAPH_ATTRS_KEY] = torch.cat(out, dim=-1)
        return data