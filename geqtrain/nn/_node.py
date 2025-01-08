import torch
import torch.nn
import math
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin

from typing import Dict, Optional

@compile_mode("script")
class EmbeddingNodeAttrs(GraphModuleMixin, torch.nn.Module):
    """Select the node embedding based on node type.

    Args:
    """

    def __init__(
        self,
        node_attributes: Dict[str, Dict] = {},
        num_types: Optional[int] = None,
        irreps_in=None,
    ):
        super().__init__()

        attr_modules = torch.nn.ModuleDict() # k: str field name, v: nn.Embedding layer
        output_embedding_dim = 0
        for field, values in node_attributes.items():
            if 'embedding_dimensionality' not in values: # this means the attr is not used as embedding
                continue
            n_types = values.get('actual_num_types', num_types)
            embedding_dim = values['embedding_dimensionality']
            emb_module = torch.nn.Embedding(n_types, embedding_dim)
            torch.nn.init.normal_(emb_module.weight, mean=0, std=1) # std 1 or math.isqrt(embedding_dim), 1 could be better

            attr_modules[field] = emb_module
            output_embedding_dim += embedding_dim

        self.attr_modules = attr_modules
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(output_embedding_dim, (0, 1))])} # output_embedding_dim scalars (l=0) with even parity
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        out = []
        for attribute_name, emb_layer in self.attr_modules.items():
            x = data[attribute_name].squeeze()
            x = emb_layer(x)
            out.append(x)

        data[AtomicDataDict.NODE_ATTRS_KEY] = torch.cat(out, dim=-1)
        return data


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features

        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[AtomicDataDict.NODE_ATTRS_KEY]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out) # my guess: None -> NatomsTypes of l = 0, defines inpt/outpt shapes

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if data.get(AtomicDataDict.NODE_ATTRS_KEY, None) is None:
            type_numbers = data.get(AtomicDataDict.NODE_TYPE_KEY).squeeze(-1)
            one_hot = torch.nn.functional.one_hot(
                type_numbers, num_classes=self.num_types
            ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype) # my guess: [bs, n, NatomsTypes]

            data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
            if self.set_features:
                data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data