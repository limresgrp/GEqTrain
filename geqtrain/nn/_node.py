import torch
import torch.nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
import math
from typing import Dict, Optional, List
import pickle
import os

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
    random_tokens = torch.randint(0, num_categories, (random_indices.sum(),), device=x.device, dtype=x.dtype)
    x[random_indices] = random_tokens

    return x


@compile_mode("script")
class EmbeddingAttrs(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        out_field: str,
        attributes: Dict[str, Dict] = {}, # key to parse from yaml
        num_types: Optional[int] = None,
        use_masking: bool = True,
        fields_to_mask: List[str] = [],
        irreps_in=None,
        use_kano_embeddings: bool = True,
    ):
        super().__init__()
        self.out_field = out_field
        self.use_masking = use_masking
        self.fields_to_mask = fields_to_mask
        self._numerical_attrs = []
        self.attr_modules = torch.nn.ModuleDict() # k: str field name, v: nn.Embedding layer
        output_embedding_dim = 0
        self.use_kano_embeddings = use_kano_embeddings

        # node_types
        kano_dims = 0
        if self.use_kano_embeddings and out_field == AtomicDataDict.NODE_ATTRS_KEY:
            # assert presence of atom number eg 6 for carbon
            kano_emb_path = os.path.join(os.path.dirname(__file__), "kano_embeddings", "ele2emb.pkl")
            with open(kano_emb_path, 'rb') as f:
                self.kano_embeddings = pickle.load(f) # dict of atom_type: np.array
                kano_dims = self.kano_embeddings[0].shape[0] # check if all embeddings have the same size
                self.kano_embeddings = torch.nn.ParameterDict({str(k): torch.nn.Parameter(torch.tensor(v, dtype=torch.float32)) for k, v in self.kano_embeddings.items()})
                self.kano_embeddings.requires_grad = True # can freeze the embeddings here if needed
        else:
            self.use_kano_embeddings = False

        output_embedding_dim += kano_dims
        for field, values in attributes.items():

            if 'embedding_dimensionality' not in values: # if attr is not used as embedding
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

                self.attr_modules[field] = emb_module

        irreps_out = {self.out_field: Irreps([(output_embedding_dim, (0, 1))])} # output_embedding_dim scalars (l=0) with even parity
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
        self._has_numericals = len(self._numerical_attrs) > 0

    @torch.amp.autocast('cuda', enabled=False) # embeddings always kept to high precision, regardless of AMP
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        out = []
        for attribute_name, emb_layer in self.attr_modules.items():
            x = data[attribute_name].squeeze(-1)

            if self.use_masking and attribute_name in self.fields_to_mask:
                x = apply_masking(x, mask_token_index=emb_layer.weight.shape[0] -1) # last index is reserved for masking, make sure to match this in yaml

            x_emb = emb_layer(x)
            out.append(x_emb)

        if self._has_numericals:
            assert hasattr(self, '_numerical_attrs') # Needed to jit compile
            for attribute_name in self._numerical_attrs:
                x = data[attribute_name]
                out.append(x)

        if self.use_kano_embeddings:
            # get kano embeddings for each atom type
            atom_types = data[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
            kano_emb = torch.stack([self.kano_embeddings[str(atom_type.item())] for atom_type in atom_types], dim=0)
            out.append(kano_emb)

        data[self.out_field] = torch.cat(out, dim=-1).float()
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