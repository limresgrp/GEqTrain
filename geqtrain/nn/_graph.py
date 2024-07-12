from collections import OrderedDict
from typing import List
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

    graph_input_num_types: int

    def __init__(
        self,
        graph_input_num_types: List[int],
        embedding_dim: int = 8,
        irreps_in=None,
    ):
        super().__init__()
        self.graph_inputs: int = len(graph_input_num_types)
        output_embedding_dim = 0
        emb_dict = OrderedDict()
        for idx, num in enumerate(graph_input_num_types):
            embedding = torch.nn.Embedding(num, embedding_dim)
            torch.nn.init.normal_(embedding.weight, mean=0, std=1/math.sqrt(embedding_dim))
            emb_dict[f'{idx}_emb'] = embedding
            output_embedding_dim += embedding_dim
        self.embeddings = torch.nn.Sequential(emb_dict)

        irreps_out = {AtomicDataDict.GRAPH_ATTRS_KEY: Irreps([(output_embedding_dim, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if data.get(AtomicDataDict.GRAPH_ATTRS_KEY, None) is None:
            graph_inputs = data.get(AtomicDataDict.GRAPH_INPUT_TYPE_KEY).reshape(-1, self.graph_inputs).T
            graph_attrs = []
            for embedding, graph_input in zip(self.embeddings, graph_inputs):
                graph_attrs.append(embedding(graph_input))
            data[AtomicDataDict.GRAPH_ATTRS_KEY] = torch.cat(graph_attrs, dim=1)
        return data