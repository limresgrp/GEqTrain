from typing import List, Optional, Union

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

# Assuming these are the correct import paths from your project structure
from geqtrain.data import (
    AtomicDataDict,
    _NODE_FIELDS,
    _GRAPH_FIELDS,
)
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.nn._heads import L0IndexedAttention
from geqtrain.nn._equivariant_scalar_mlp import EquivariantScalarMLP


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, nn.Module):
    conditioning_fields: List[str]
    """
    This module takes a feature tensor (`field`) and computes an output tensor
    (`out_field`). It can optionally condition its internal transformations on a
    second tensor (`conditioning_field`).
    """
    def __init__(
        self,
        field: str,
        irreps_in,
        out_field: Optional[str] = None, # The key where the output will be stored
        conditioning_fields: Optional[List[str]] = None, # List of keys to use for conditioning
        out_irreps: Union[o3.Irreps, str, None] = None,
        strict_irreps: bool = True,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        resnet: bool = False,
        ignore_amp: bool = False,
        bias: Optional[Union[float, List]] = None,
    ):
        super().__init__()

        self.field = field
        self.out_field = out_field or self.field
        self.conditioning_fields = conditioning_fields if conditioning_fields is not None else []
        self.ignore_amp = ignore_amp
        self.resnet = resnet

        # --- Irreps Initialization ---
        if out_irreps is None:
            if self.out_field in irreps_in:
                out_irreps = irreps_in[self.out_field]
            else:
                raise ValueError(
                    f"out_irreps is None, but out_field '{self.out_field}' is not in irreps_in. "
                    "Please provide out_irreps explicitly."
                )

        required_irreps = [field]
        required_irreps.extend(self.conditioning_fields)

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=required_irreps,
            irreps_out={self.out_field: out_irreps},
        )
        in_irreps: o3.Irreps = self.irreps_in[field]
        out_irreps: o3.Irreps = self.irreps_out[self.out_field]

        # --- Resnet ---
        self._resnet_update_coeff: Optional[nn.Parameter] = None
        if self.resnet:
            if self.out_field not in self.irreps_in:
                 raise ValueError(f"For resnet=True, out_field='{self.out_field}' must be in `irreps_in`")
            if self.irreps_in[self.out_field] != out_irreps:
                 raise ValueError("For resnet=True, output irreps must match input irreps for the out_field.")
            self._resnet_update_coeff = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        
        # --- Conditioning Fields Validation and Dimension Calculation ---
        self.total_conditioning_dim = 0
        for cond_field in self.conditioning_fields:
            if cond_field not in self.irreps_in:
                raise ValueError(f"Conditioning field '{cond_field}' not found in irreps_in.")
            cond_irreps = self.irreps_in[cond_field]
            if not all(ir.l == 0 for _, ir in cond_irreps):
                raise ValueError(f"Conditioning field '{cond_field}' must have scalar (0e) irreps, but got {cond_irreps}.")
            self.total_conditioning_dim += cond_irreps.dim

        # --- Core Processing Module ---
        self.processor = EquivariantScalarMLP(
            in_irreps=in_irreps,
            out_irreps=out_irreps,
            conditioning_dim=self.total_conditioning_dim,
            latent_module=readout_latent,
            latent_kwargs=readout_latent_kwargs,
            strict_irreps=strict_irreps,
        )
        self.n_scalars_out = sum(mul for mul, ir in out_irreps if ir.l == 0)

        # --- Bias ---
        self.bias = None
        if self.n_scalars_out > 0:
            if bias is not None:
                if isinstance(bias, float):
                    bias_init = torch.full((self.n_scalars_out,), bias, dtype=torch.float32)
                elif isinstance(bias, list):
                    bias_init = torch.tensor(bias, dtype=torch.float32)
                    assert bias_init.shape[0] == self.n_scalars_out, "Length of bias list must match number of scalar outputs."
                else: 
                    raise ValueError("Bias must be a float or a list of floats.")
                self.bias = nn.Parameter(bias_init)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not torch.jit.is_scripting() and self.ignore_amp:
            with torch.amp.autocast('cuda', enabled=False):
                return self._forward_impl(data)
        return self._forward_impl(data)

    def _forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        conditioning_tensor: Optional[torch.Tensor] = None
        if len(self.conditioning_fields) > 0:
            conditioning_tensor_list = [data[f] for f in self.conditioning_fields]
            conditioning_tensor = torch.cat(conditioning_tensor_list, dim=-1)

        out_features = self.processor(features, conditioning_tensor)

        if self.resnet:
            old_features = data[self.out_field]
            assert self._resnet_update_coeff is not None
            coeff = self._resnet_update_coeff.sigmoid()
            coefficient_old = torch.rsqrt(coeff.square() + 1)
            coefficient_new = coeff * coefficient_old
            out_features = coefficient_old * old_features + coefficient_new * out_features

        if self.bias is not None:
            out_scalars = out_features[..., :self.n_scalars_out]
            out_equiv = out_features[..., self.n_scalars_out:]
            biased_scalars = out_scalars + self.bias
            if out_equiv.shape[-1] > 0:
                out_features = torch.cat([biased_scalars, out_equiv], dim=-1)
            else:
                out_features = biased_scalars

        data[self.out_field] = out_features
        return data


@compile_mode("script")
class AttentionReadoutModule(ReadoutModule):
    """
    Extends ReadoutModule to include a non-local attention mechanism.

    Attention is applied to the scalar part of the input features *before*
    the standard readout process. This breaks strict locality, allowing
    information to be aggregated across nodes or graphs.
    """
    def __init__(
        self,
        num_heads: int = 8,
        dataset_mode: str = 'single', # single|ensemble
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.n_scalars_in == 0:
            raise ValueError("AttentionReadoutModule requires scalar input features.")
        
        self.scalar_attnt_enabled = True
        idx_key = ""
        
        # self.split_index is initialized in the parent `processor`
        # but JIT needs a type hint.
        self.split_index: int = self.processor.split_index
        if self.field in _NODE_FIELDS:
            idx_key = AtomicDataDict.BATCH_KEY
        elif self.field in _GRAPH_FIELDS and dataset_mode == 'ensemble':
            idx_key = AtomicDataDict.ENSEMBLE_INDEX_KEY
        else:
            self.scalar_attnt_enabled = False
            
        if self.scalar_attnt_enabled:
            self.ensemble_attnt1 = L0IndexedAttention(
                irreps_in=self.irreps_in, field=self.field, out_field=self.field, 
                num_heads=num_heads, idx_key=idx_key, update_mlp=True
            )
            self.ensemble_attnt2 = L0IndexedAttention(
                irreps_in=self.irreps_in, field=self.field, out_field=self.field, 
                num_heads=num_heads, idx_key=idx_key
            )
        else:
            self.ensemble_attnt1 = None
            self.ensemble_attnt2 = None

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Applies attention to scalar features, then passes to the base ReadoutModule.
        """
        if self.scalar_attnt_enabled and self.ensemble_attnt1 is not None:
            # This is a pre-processing step for the main readout forward pass.
            # We modify the feature tensor in `data` before calling super().forward().
            features = data[self.field]
            scalars, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)
            
            # The original L0IndexedAttention implementation seems to operate on a tensor of scalars.
            scalars = self.ensemble_attnt1(scalars, data)
            scalars = self.ensemble_attnt2(scalars, data)
            
            data[self.field] = torch.cat((scalars, equiv), dim=-1)

        return super().forward(data)
