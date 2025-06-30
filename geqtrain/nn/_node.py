import logging
import torch
import torch.nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
import math
# from torch_scatter import scatter
from geqtrain.utils.pytorch_scatter import scatter_sum
from typing import Dict, Optional, List


def apply_masking(x: torch.Tensor, mask_token_index: int) -> torch.Tensor:
    """
    Applies masking to categorical attributes within an input tensor.

    This function randomly masks elements of the input tensor `x` and replaces them
    either with a mask token or a random token, based on a certain probability.
    This is often used as a data augmentation technique in training models dealing
    with categorical data, such as training language models.

    Args:
        x (torch.Tensor): The input tensor containing categorical attributes.
            It is assumed that the tensor contains integer indices representing
            different categories.
        mask_token_index (int): The index of the mask token. This is used to replace
            some of the input tokens with a mask token. It is also assumed that the
            number of categories is `mask_token_index - 1`.

    Returns:
        torch.Tensor: A new tensor with the same shape as `x`, where some elements
                      have been replaced with either a mask token or a random token.

    """
    # step 1: define which elements of input are going to be masked
    random_mask = torch.rand(x.shape, device=x.device)
    mask = (random_mask > 0.8) # 20% masking
    x = x.clone()  # Avoid modifying the original tensor

    # step 2: determine whether to use mask token or random token
    random_choice = torch.rand(x.shape, device=x.device)
    mask_or_random = random_choice > 0.5  # 50% chance of random token

    # step 3a: apply mask token
    mask_indices = mask & ~mask_or_random # Apply mask where mask is True AND mask_or_random is False
    x[mask_indices] = mask_token_index

    # step3b: apply random token
    random_indices = mask & mask_or_random # Apply random where mask is True AND mask_or_random is True
    num_categories = mask_token_index - 1 # Exclude mask token from random sampling
    random_tokens = torch.randint(0, num_categories, random_indices.sum().item(), device=x.device, dtype=x.dtype)
    x[random_indices] = random_tokens

    return x


@compile_mode("script")
class EmbeddingAttrs(GraphModuleMixin, torch.nn.Module):
    
    fields_to_mask: List[str]

    def __init__(
        self,
        out_field: str,
        attributes: Dict[str, Dict] = {}, # key to parse from yaml
        eq_out_field: Optional[str] = None,
        eq_attributes: Optional[Dict[str, Dict]] = None,
        num_types: Optional[int] = None,
        use_masking: bool = True,
        fields_to_mask: List[str] = [],
        
        irreps_in=None,
    ):
        super().__init__()
        self.out_field     = out_field
        self.eq_out_field  = eq_out_field
        self._categorical_attrs_modules = torch.nn.ModuleDict() # k: str field name, v: nn.Embedding layer
        self._numerical_attrs    = []
        self._eq_numerical_attrs = []

        self.use_masking = use_masking
        self.fields_to_mask = fields_to_mask
        
        output_embedding_dim = 0
        for field, values in attributes.items():
            if 'embedding_dimensionality' not in values: # if attr is not used as embedding
                logging.warning(f"Field {field} is missing 'embedding_dimensionality'. Not using as invariant input")
                continue
            embedding_dim = values['embedding_dimensionality']
            output_embedding_dim += embedding_dim
            if values.get('attribute_type', 'categorical') == 'numerical':
                self._numerical_attrs.append(field)
            else:
                n_types = values.get('actual_num_types', num_types) # ! IMPO should be + 1 for masking category but handled via cfg.yaml s.t. round emb_module.weight.shape[0] to be set to the closest power of 2 possible
                emb_module = torch.nn.Embedding(n_types, embedding_dim)
                # TODO explore these different options
                # torch.nn.init.normal_(emb_module.weight, mean=0, std=1.0) # options: 1) std=1 2) math.isqrt(embedding_dim) 3) 0.3333*math.isqrt(embedding_dim) as in https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md
                torch.nn.init.xavier_uniform_(emb_module.weight)
                emb_module.weight.data *= 0.3333 * math.isqrt(embedding_dim)
                self._categorical_attrs_modules[field] = emb_module
        
        eq_output_embedding_dim = 0
        eq_irreps = None
        if eq_attributes is None: eq_attributes = {}
        for field, values in eq_attributes.items():
            if 'embedding_dimensionality' not in values: # if attr is not used as embedding
                logging.warning(f"Field {field} is missing 'embedding_dimensionality'. Not using as equivariant input")
                continue
            embedding_dim = values['embedding_dimensionality']
            field_irreps  = Irreps(values['irreps'])
            eq_output_embedding_dim += embedding_dim
            if eq_irreps is None:
                eq_irreps = field_irreps
            else:
                eq_irreps += field_irreps
            self._eq_numerical_attrs.append(field)

        self._has_categoricals = len(self._categorical_attrs_modules) > 0
        self._has_numericals   = len(self._numerical_attrs) > 0
        self._has_equivariants = len(self._eq_numerical_attrs) > 0

        irreps_out = {}
        if self._has_categoricals or self._has_numericals:
            irreps_out[self.out_field] = Irreps([(output_embedding_dim, (0, 1))])
        if self._has_equivariants:
            assert self.eq_out_field is not None
            irreps_out[self.eq_out_field] = eq_irreps
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    #! ----------- COMMENT TO JIT COMPILE --------------- #
    @torch.amp.autocast('cuda', enabled=False) # embeddings always kept to high precision, regardless of AMP
    # --------------------------------------------------- #
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        out = []
        for attribute_name, emb_layer in self._categorical_attrs_modules.items():
            x = data[attribute_name].squeeze(-1)
            if self.use_masking and attribute_name in self.fields_to_mask:
                x = apply_masking(x, mask_token_index=emb_layer.weight.shape[0] - 1) # last index is reserved for masking, make sure to match this in yaml
            x_emb = emb_layer(x)
            out.append(x_emb)
        
        if self._has_numericals:
            assert hasattr(self, '_numerical_attrs') # Needed to jit compile
            for attribute_name in self._numerical_attrs:
                x = data[attribute_name]
                out.append(x)
        
        data[self.out_field] = torch.cat(out, dim=-1).to(torch.get_default_dtype())
        
        if self._has_equivariants:
            eq_out = []
            assert hasattr(self, '_eq_numerical_attrs') # Needed to jit compile
            for eq_attribute_name in self._eq_numerical_attrs:
                x = data[eq_attribute_name]
                eq_out.append(x)
            data[self.eq_out_field] = torch.cat(eq_out, dim=-1).to(torch.get_default_dtype())    
        
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


@compile_mode("script")
class GotenNetEmbedding(GraphModuleMixin, torch.nn.Module):
    
    fields_to_mask: List[str]

    def __init__(
        self,
        field: str = AtomicDataDict.NODE_ATTRS_KEY,
        out_field: str = AtomicDataDict.NODE_ATTRS_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self._init_irreps(irreps_in=irreps_in)

        node_attr_dim = self.irreps_in[self.field].dim
        self.W_node = torch.nn.Parameter(torch.randn(node_attr_dim, node_attr_dim))
        self.W_center_node = torch.nn.Parameter(torch.randn(node_attr_dim, node_attr_dim))
        self.W_concat_node = torch.nn.Parameter(torch.randn(2 * node_attr_dim, node_attr_dim))
        edge_attr_dim = self.irreps_in[AtomicDataDict.EDGE_RADIAL_ATTRS_KEY].dim
        self.W_edge = torch.nn.Parameter(torch.randn(edge_attr_dim, node_attr_dim))
        self.norm = torch.nn.LayerNorm(node_attr_dim)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center, edge_neigh = data[AtomicDataDict.EDGE_INDEX_KEY]
        node_attr = data[self.field]
        edge_attr = data[AtomicDataDict.EDGE_RADIAL_ATTRS_KEY]
        
        proj_node = torch.einsum('nd,dd -> nd', node_attr, self.W_node)
        proj_edge = proj_node[edge_neigh]
        proj_radial = torch.einsum('ej,jd -> ed', edge_attr, self.W_edge)

        num_nodes = len(node_attr)
        m_node = scatter_sum(proj_edge * proj_radial, edge_center, dim=0, dim_size=num_nodes)

        proj_center_node = torch.einsum('nd,dd -> nd', node_attr, self.W_center_node)
        h_node = self.norm(torch.einsum('nk,kd -> nd', torch.cat(proj_center_node, m_node, dim=-1), self.W_concat_node))

        data[self.out_field] = h_node
        return data