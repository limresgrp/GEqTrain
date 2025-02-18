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
        stable_embedding:bool=False,
    ):
        super().__init__()

        numerical_attrs = []
        categorical_attr_modules = torch.nn.ModuleDict() # k: str field name, v: nn.Embedding layer
        output_embedding_dim = 0
        for field, values in node_attributes.items():

            if 'embedding_dimensionality' not in values: # this means the attr is not used as embedding
                continue

            embedding_dim = values['embedding_dimensionality']
            if values.get('attribute_type', 'categorical') == 'numerical':
                numerical_attrs.append(field)
            else:
                n_types = values.get('actual_num_types', num_types)
                embedding_dim = values['embedding_dimensionality']
                emb_module = torch.nn.Embedding(n_types, embedding_dim)
                # torch.nn.init.xavier_uniform_(emb_module.weight) # with option 3 below?
                torch.nn.init.normal_(emb_module.weight, mean=0, std=1.0) # options: 1) std=1 2) math.isqrt(embedding_dim) 3) 0.3333*math.isqrt(embedding_dim) as in https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md
                categorical_attr_modules[field] = emb_module

            if stable_embedding:
                categorical_attr_modules[field] = torch.nn.Sequential(emb_module, torch.nn.LayerNorm(embedding_dim)) # as in: https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/embeddings#bitsandbytes.nn.StableEmbedding

            output_embedding_dim += embedding_dim

        self.numerical_attrs = numerical_attrs
        self.categorical_attr_modules = categorical_attr_modules
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(output_embedding_dim, (0, 1))])} # output_embedding_dim scalars (l=0) with even parity
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        with torch.cuda.amp.autocast(enabled=False): # choice: embeddings are always kept to high precision, regardless of amp
            out = []
            for attribute_name, emb_layer in self.categorical_attr_modules.items():
                x = data[attribute_name].squeeze()
                x = emb_layer(x)
                out.append(x)

        for attribute_name in self.numerical_attrs:
            x = data[attribute_name]
            out.append(x)

        data[AtomicDataDict.NODE_ATTRS_KEY] = torch.cat(out, dim=-1).float()
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