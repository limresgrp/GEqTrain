import logging
from typing import Dict, List, Optional

import torch
import torch.nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn._graph_mixin import GraphModuleMixin


@torch.jit.script
def apply_masking(x: torch.Tensor, mask_token_index: int) -> torch.Tensor:
    output = x.clone()
    random_mask = torch.rand(x.shape, device=x.device)
    mask = random_mask > 0.8

    random_choice = torch.rand(x.shape, device=x.device)
    use_random_token = random_choice > 0.5

    num_random_categories = mask_token_index
    random_tokens = torch.randint(
        0,
        num_random_categories,
        x.shape,
        device=x.device,
        dtype=x.dtype,
    )
    mask_tokens = torch.full_like(x, fill_value=mask_token_index)

    condition_random = mask & use_random_token
    output = torch.where(condition_random, random_tokens, output)

    condition_mask = mask & ~use_random_token
    output = torch.where(condition_mask, mask_tokens, output)

    return output


@compile_mode("script")
class EmbeddingInputAttrs(GraphModuleMixin, torch.nn.Module):
    fields_to_mask: List[str]

    def __init__(
        self,
        attributes: Optional[Dict[str, Dict]] = None,
        eq_attributes: Optional[Dict[str, Dict]] = None,
        out_field: Optional[str] = None,
        eq_out_field: Optional[str] = None,
        use_masking: bool = True,
        fields_to_mask: Optional[List[str]] = None,
        irreps_in=None,
    ):
        super().__init__()
        self.out_field = out_field
        self.eq_out_field = eq_out_field
        self._categorical_attrs_modules = torch.nn.ModuleDict()
        self._categorical_num_types = {}
        self._numerical_attrs = []
        self._eq_numerical_attrs = []

        self.use_masking = use_masking
        self.fields_to_mask = fields_to_mask if fields_to_mask is not None else []

        attributes = attributes or {}

        output_embedding_dim = 0
        my_irreps_in = {}
        for field, values in attributes.items():
            attribute_type = values.get("attribute_type", "categorical")
            embedding_mode = values.get("embedding_mode", "embedding")
            is_binned_numerical = attribute_type == "numerical" and _uses_binning(values)

            if attribute_type == "numerical" and not is_binned_numerical:
                if "embedding_dimensionality" not in values:
                    logging.warning(
                        f"Field {field} is missing 'embedding_dimensionality'. Not using as numerical input"
                    )
                    continue
                embedding_dim = int(values["embedding_dimensionality"])
                if embedding_dim <= 0:
                    logging.warning(
                        f"Field {field} has non-positive embedding_dimensionality. Skipping numerical input."
                    )
                    continue
                output_embedding_dim += embedding_dim
                self._numerical_attrs.append(field)
                my_irreps_in[field] = Irreps(f"{embedding_dim}x0e")
            else:
                if embedding_mode == "embedding" and "embedding_dimensionality" not in values:
                    logging.warning(
                        f"Field {field} is missing 'embedding_dimensionality'. Not using as invariant input"
                    )
                    continue
                n_types = values.get("actual_num_types", values.get("num_types"))
                if n_types is None or int(n_types) <= 0:
                    logging.warning(
                        f"Field {field} is missing a valid 'num_types'. Skipping categorical input."
                    )
                    continue
                n_types = int(n_types)
                self._categorical_num_types[field] = n_types
                if embedding_mode == "one_hot":
                    emb_module = OneHotEncoding(n_types)
                    output_embedding_dim_incr = n_types
                else:
                    assert embedding_mode == "embedding"
                    embedding_dim = int(values["embedding_dimensionality"])
                    if embedding_dim <= 0:
                        logging.warning(
                            f"Field {field} has non-positive embedding_dimensionality. Skipping categorical embedding."
                        )
                        continue
                    emb_module = torch.nn.Embedding(n_types, embedding_dim)
                    torch.nn.init.xavier_uniform_(emb_module.weight)
                    output_embedding_dim_incr = embedding_dim
                output_embedding_dim += output_embedding_dim_incr
                self._categorical_attrs_modules[field] = emb_module
                my_irreps_in[field] = None

        eq_irreps = None
        if eq_attributes is None:
            eq_attributes = {}
        for field, values in eq_attributes.items():
            if values.get("attribute_type", "numerical") != "numerical":
                raise ValueError(f"Equivariant attribute '{field}' must be numerical.")
            if "irreps" not in values:
                raise ValueError(f"Equivariant attribute '{field}' is missing required 'irreps'.")
            field_irreps = Irreps(values["irreps"])
            if "embedding_dimensionality" in values:
                embedding_dim = int(values["embedding_dimensionality"])
                if embedding_dim != field_irreps.dim:
                    raise ValueError(
                        f"Equivariant attribute '{field}' embedding_dimensionality={embedding_dim} "
                        f"does not match irreps dim={field_irreps.dim}."
                    )
            if eq_irreps is None:
                eq_irreps = field_irreps
            else:
                eq_irreps += field_irreps
            self._eq_numerical_attrs.append(field)
            my_irreps_in[field] = field_irreps

        self._has_categoricals = len(self._categorical_attrs_modules) > 0
        self._has_numericals = len(self._numerical_attrs) > 0
        self._has_equivariants = len(self._eq_numerical_attrs) > 0

        irreps_out = {}
        if self._has_categoricals or self._has_numericals:
            irreps_out[self.out_field] = Irreps([(output_embedding_dim, (0, 1))])
        if self._has_equivariants:
            assert self.eq_out_field is not None
            irreps_out[self.eq_out_field] = eq_irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in=my_irreps_in,
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not torch.jit.is_scripting():
            with torch.amp.autocast("cuda", enabled=False):
                return self._forward_impl(data)
        return self._forward_impl(data)

    def _forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        dtype = data[AtomicDataDict.POSITIONS_KEY].dtype
        if self._has_categoricals or self._has_numericals:
            out = []
            for attribute_name, emb_layer in self._categorical_attrs_modules.items():
                x = data[attribute_name].squeeze(-1)
                if self.use_masking and attribute_name in self.fields_to_mask:
                    num_types = self._categorical_num_types.get(attribute_name, 0)
                    if num_types > 0:
                        x = apply_masking(x, mask_token_index=num_types - 1)
                x_emb = emb_layer(x)
                out.append(x_emb)

            if self._has_numericals:
                assert hasattr(self, "_numerical_attrs")
                for attribute_name in self._numerical_attrs:
                    out.append(data[attribute_name])
            if len(out) > 0:
                data[self.out_field] = torch.cat(out, dim=-1).to(dtype)

        if self._has_equivariants:
            eq_out = []
            assert hasattr(self, "_eq_numerical_attrs")
            for eq_attribute_name in self._eq_numerical_attrs:
                eq_out.append(data[eq_attribute_name])
            data[self.eq_out_field] = torch.cat(eq_out, dim=-1).to(dtype)

        return data


def _uses_binning(values: Dict) -> bool:
    return any(key in values for key in ("min_value", "max_value", "bin_edges", "bins"))


@compile_mode("script")
class OneHotEncoding(torch.nn.Module):
    num_types: int

    def __init__(self, num_types: int):
        super().__init__()
        self.num_types = num_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x, num_classes=self.num_types)
