import logging
import torch
import torch.nn
from e3nn.o3 import Irreps
from geqtrain.data import AtomicDataDict
from geqtrain.nn._edge import BaseEdgeEmbedding, BaseEdgeEqEmbedding
from ._graph_mixin import GraphModuleMixin
from typing import Dict, Optional, List
from e3nn.util.jit import compile_mode


@torch.jit.script
def apply_masking(x: torch.Tensor, mask_token_index: int) -> torch.Tensor:
    """
    Applies masking to categorical attributes within an input tensor in a JIT-compatible way.

    This function randomly masks elements of the input tensor `x` and replaces them
    either with a mask token or a random token, based on a certain probability.
    This is often used as a data augmentation technique in training models dealing
    with categorical data.

    Args:
        x (torch.Tensor): The input tensor containing categorical attributes.
            It is assumed that the tensor contains integer indices representing
            different categories.
        mask_token_index (int): The index of the mask token. This is used to replace
            some of the input tokens with a mask token. It is also assumed that the
            number of categories for random sampling is `mask_token_index`.

    Returns:
        torch.Tensor: A new tensor with the same shape as `x`, where some elements
                      have been replaced with either a mask token or a random token.
    """
    # step 1: define which elements of input are going to be masked
    # Create a clone to avoid modifying the original tensor in place
    output = x.clone()
    random_mask = torch.rand(x.shape, device=x.device)
    mask = (random_mask > 0.8)  # 20% masking

    # step 2: determine whether to use mask token or random token
    random_choice = torch.rand(x.shape, device=x.device)
    use_random_token = random_choice > 0.5  # 50% chance of random token

    # step 3: prepare replacements
    # The number of actual categories (e.g., atom types) to sample from is
    # all indices up to, but not including, the mask_token_index.
    # torch.randint's second argument is the exclusive upper bound.
    # So, we should sample from [0, mask_token_index).
    num_random_categories = mask_token_index

    # 3a: prepare random tokens for all positions
    random_tokens = torch.randint(
        0, num_random_categories, x.shape, device=x.device, dtype=x.dtype
    )

    # 3b: prepare mask tokens for all positions
    mask_tokens = torch.full_like(x, fill_value=mask_token_index)

    # step 4: apply replacements using torch.where
    # First, apply random tokens where `mask` is true AND `use_random_token` is true
    condition_random = mask & use_random_token
    output = torch.where(condition_random, random_tokens, output)

    # Then, apply mask tokens where `mask` is true AND `use_random_token` is false
    condition_mask = mask & ~use_random_token
    output = torch.where(condition_mask, mask_tokens, output)

    return output


@compile_mode("script")
class EmbeddingInputAttrs(GraphModuleMixin, torch.nn.Module):

    fields_to_mask: List[str]

    def __init__(
        self,
        out_field:      Optional[str]             = None,
        eq_out_field:   Optional[str]             = None,
        attributes:     Dict[str, Dict]           = {}, # key to parse from yaml
        eq_attributes:  Optional[Dict[str, Dict]] = None,
        use_masking:    bool                      = True,
        fields_to_mask: List[str]                 = [],
        irreps_in                                 = None,
    ):
        super().__init__()
        self.out_field     = out_field
        self.eq_out_field  = eq_out_field
        self._categorical_attrs_modules = torch.nn.ModuleDict() # k: str field name, v: nn.Embedding layer or nn.OneHot if one-hot
        self._numerical_attrs    = []
        self._eq_numerical_attrs = []

        self.use_masking = use_masking
        self.fields_to_mask = fields_to_mask
        self.n_types = 0

        output_embedding_dim = 0
        for field, values in attributes.items():
            embedding_mode = values.get('embedding_mode', 'embedding')
            if embedding_mode == 'embedding' and 'embedding_dimensionality' not in values: # if attr is not used as embedding
                logging.warning(f"Field {field} is missing 'embedding_dimensionality'. Not using as invariant input")
                continue
            embedding_dim: int = values.get('embedding_dimensionality', 0)
            if values.get('attribute_type', 'categorical') == 'numerical':
                output_embedding_dim += embedding_dim
                self._numerical_attrs.append(field)
            else:
                self.n_types: int = values['actual_num_types']
                if embedding_mode == 'one_hot':
                    # Use one-hot encoding
                    emb_module = OneHotEncoding(self.n_types)
                    self._categorical_attrs_modules[field] = emb_module
                    output_embedding_dim_incr = self.n_types
                else:
                    # Use Embedding
                    assert embedding_mode == 'embedding'
                    emb_module = torch.nn.Embedding(self.n_types, embedding_dim)
                    torch.nn.init.normal_(emb_module.weight, mean=0.0, std=1.0) # Large Initial Magnitudes -> Large Initial Outputs -> Large Initial Gradients
                    output_embedding_dim_incr = embedding_dim
                output_embedding_dim += output_embedding_dim_incr
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
    #@torch.amp.autocast('cuda', enabled=False) # embeddings always kept to high precision, regardless of AMP
    # --------------------------------------------------- #
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        out = []
        for attribute_name, emb_layer in self._categorical_attrs_modules.items():
            x = data[attribute_name].squeeze(-1)
            if self.use_masking and attribute_name in self.fields_to_mask and self.n_types > 0:
                x = apply_masking(x, mask_token_index=self.n_types - 1) # last index is reserved for masking, make sure to match this in yaml
            x_emb = emb_layer(x)
            out.append(x_emb)

        if self._has_numericals:
            assert hasattr(self, '_numerical_attrs') # Needed to jit compile
            for attribute_name in self._numerical_attrs:
                x = data[attribute_name]
                out.append(x)

        dtype = data[AtomicDataDict.POSITIONS_KEY].dtype
        data[self.out_field] = torch.cat(out, dim=-1).to(dtype)

        if self._has_equivariants:
            eq_out = []
            assert hasattr(self, '_eq_numerical_attrs') # Needed to jit compile
            for eq_attribute_name in self._eq_numerical_attrs:
                x = data[eq_attribute_name]
                eq_out.append(x)
            data[self.eq_out_field] = torch.cat(eq_out, dim=-1).to()

        return data


@compile_mode("script")
class OneHotEncoding(torch.nn.Module):
    num_types: int

    def __init__(self, num_types: int):
        super().__init__()
        self.num_types = num_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x, num_classes=self.num_types)


@compile_mode("script")
class EmbeddingAttrs(GraphModuleMixin, torch.nn.Module):
    num_types: int

    def __init__(
        self,
        # architectural choices
        node_emb           = None, # in yaml file, add param (e.g. "node_emb: geqtrain.nn.goten.GotenNodeEmbedding")
        node_emb_kwargs    = {},   # in yaml file, add param (e.g. "node_emb_my_param: 42")
        node_eq_emb        = None, # in yaml file, add param (not yet implemented any)
        node_eq_emb_kwargs = {},   # in yaml file, add param (not yet implemented any)
        edge_emb           = BaseEdgeEmbedding, # in yaml file, add param (e.g. "edge_emb: geqtrain.nn.allegro.AllegroEdgeEmbedding")
        edge_emb_kwargs    = {},   # in yaml file, add param (e.g. "edge_emb_my_param: True")
        edge_eq_emb        = BaseEdgeEqEmbedding, # in yaml file, add param (default is the only implemented)
        edge_eq_emb_kwargs = {},   # in yaml file, add param (e.g. "edge_emb_my_param: 1")
        # other
        irreps_in          = None,
    ):
        super().__init__()
        self.node_field        = AtomicDataDict.NODE_INPUT_ATTRS_KEY
        self.node_eq_field     = AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY
        self.edge_field        = AtomicDataDict.EDGE_INPUT_ATTRS_KEY
        self.edge_eq_field     = AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY
        self.node_out_field    = AtomicDataDict.NODE_ATTRS_KEY
        self.node_eq_out_field = AtomicDataDict.NODE_EQ_ATTRS_KEY
        self.edge_out_field    = AtomicDataDict.EDGE_ATTRS_KEY
        self.edge_eq_out_field = AtomicDataDict.EDGE_EQ_ATTRS_KEY
        self.has_node_eq_attr  = False
        # irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.EDGE_RADIAL_EMB_KEY, AtomicDataDict.EDGE_SPHARMS_EMB_KEY]
        )
        assert self.node_field in self.irreps_in

        ### architectural choices
        irreps_out = self.irreps_in.copy()

        # node scalar
        self.node_emb = node_emb(
            node_field    = self.node_field,
            node_eq_field = self.node_eq_field,
            edge_field    = self.edge_field,
            edge_eq_field = self.edge_eq_field,
            irreps_in     = irreps_out,
            **node_emb_kwargs
        ) if node_emb is not None else None
        if self.node_emb is None:
            node_irreps_out = self.irreps_in[self.node_field]
        else:
            node_irreps_out = self.node_emb.out_irreps
        irreps_out[self.node_out_field] = node_irreps_out

        # node equivariant
        self.node_eq_emb = node_eq_emb(
            node_field    = self.node_out_field,
            node_eq_field = self.node_eq_field,
            edge_field    = self.edge_field,
            edge_eq_field = self.edge_eq_field,
            irreps_in     = irreps_out,
            **node_eq_emb_kwargs
        ) if node_eq_emb is not None else None
        if self.node_eq_emb is None:
            if self.node_eq_field in self.irreps_in:
                irreps_out[self.node_eq_out_field] = self.irreps_in[self.node_eq_field]
        else:
            irreps_out[self.node_eq_out_field] = self.node_eq_emb.out_irreps

        # edge scalar
        self.edge_emb = edge_emb(
            node_field    = self.node_out_field,
            node_eq_field = self.node_eq_out_field,
            edge_field    = self.edge_field,
            edge_eq_field = self.edge_eq_field,
            irreps_in     = irreps_out,
            **edge_emb_kwargs
        ) if edge_emb is not None else None
        if self.edge_emb is None:
            edge_irreps_out = self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        else:
            edge_irreps_out = self.edge_emb.out_irreps
        irreps_out[self.edge_out_field] = edge_irreps_out

        # edge equivariant
        self.edge_eq_emb = edge_eq_emb(
            node_field    = self.node_out_field,
            node_eq_field = self.node_eq_out_field,
            edge_field    = self.edge_out_field,
            edge_eq_field = self.edge_eq_field,
            irreps_in     = irreps_out,
            **edge_eq_emb_kwargs
        ) if edge_eq_emb is not None else None
        if self.edge_eq_emb is None:
            edge_eq_irreps_out = self.irreps_in[AtomicDataDict.EDGE_SPHARMS_EMB_KEY]
        else:
            edge_eq_irreps_out = self.edge_eq_emb.out_irreps
        irreps_out[self.edge_eq_out_field] = edge_eq_irreps_out

        self.irreps_out.update(irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # node scalar
        if self.node_emb is None:
            node_attr: torch.Tensor = data.pop(self.node_field) # default embedding
        else:
            node_attr: torch.Tensor = self.node_emb(data)
        data[self.node_out_field] = node_attr

        # node equivariant (optional)
        node_eq_attr: Optional[torch.Tensor] = None
        if self.node_eq_emb is None:
            if self.node_eq_field in data:
                node_eq_attr = data.get(self.node_eq_field) # default embedding
        else:
            node_eq_attr = self.node_emb(data)
        if node_eq_attr is not None:
            data[self.node_eq_out_field] = node_eq_attr

        # edge scalar
        if self.edge_emb is None:
            edge_attr = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY] # default embedding
        else:
            edge_attr = self.edge_emb(data)
        data[self.edge_out_field] = edge_attr

        # edge equivariant
        if self.edge_emb is None:
            edge_eq_attr = data[AtomicDataDict.EDGE_SPHARMS_EMB_KEY] # default embedding
        else:
            edge_eq_attr = self.edge_eq_emb(data)
        data[self.edge_eq_out_field] = edge_eq_attr

        return data