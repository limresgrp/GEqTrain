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
from geqtrain.nn._film import FiLMFunction
from geqtrain.nn.allegro import Linear
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.nn._heads import L0IndexedAttention
from geqtrain.utils.tp_utils import PSEUDO_SCALAR, SCALAR


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, nn.Module):
    """
    This module takes a feature tensor (`field`) and computes an output tensor
    (`out_field`). It can optionally condition its internal transformations on a
    second tensor (`conditioning_field`).
    """
    def __init__(
        self,
        field: str,
        irreps_in,
        out_field: Optional[str] = None,
        conditioning_field: Optional[str] = None,
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
        self.conditioning_field = conditioning_field
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
        if self.conditioning_field is not None:
            required_irreps.append(self.conditioning_field)
            
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=required_irreps,
            irreps_out={self.out_field: out_irreps},
        )
        in_irreps: o3.Irreps = self.irreps_in[field]
        out_irreps: o3.Irreps = self.irreps_out[self.out_field]
        self.out_irreps_dim = out_irreps.dim

        # --- Resnet ---
        self._resnet_update_coeff: Optional[nn.Parameter] = None
        if self.resnet:
            if self.out_field not in self.irreps_in:
                 raise ValueError(f"For resnet=True, out_field='{self.out_field}' must be in `irreps_in`")
            if self.irreps_in[self.out_field] != out_irreps:
                 raise ValueError("For resnet=True, output irreps must match input irreps for the out_field.")
            self._resnet_update_coeff = nn.Parameter(torch.tensor([0.0]))

        # --- Feature Properties ---
        self.n_scalars_in = sum(mul for mul, ir in in_irreps if ir.l == 0)
        self.n_scalars_out = sum(mul for mul, ir in out_irreps if ir.l == 0)
        self.has_invariant_output = self.n_scalars_out > 0
        self.has_equivariant_output = out_irreps.dim > self.n_scalars_out
        self.split_index = sum(mul for mul, ir in in_irreps if ir in [SCALAR, PSEUDO_SCALAR])

        # --- Conditioning Layers ---
        self.conditioner = None
        if self.conditioning_field is not None and self.n_scalars_in > 0:
            conditioning_dim = self.irreps_in[self.conditioning_field].dim
            conditioner_modules = {
                "film1": FiLMFunction(conditioning_dim, [], self.n_scalars_in, mlp_nonlinearity=None),
                "fc1": readout_latent(mlp_input_dimension=self.n_scalars_in, mlp_output_dimension=self.n_scalars_in, **readout_latent_kwargs),
                "film2": FiLMFunction(conditioning_dim, [], self.n_scalars_in, mlp_nonlinearity=None),
            }
            if self.has_invariant_output:
                conditioner_modules["film_scalar"] = FiLMFunction(conditioning_dim, [], self.n_scalars_out, mlp_nonlinearity=None)
            self.conditioner = nn.ModuleDict(conditioner_modules)

        # --- Invariant (Scalar) Readout ---
        self.inv_readout = None
        if self.has_invariant_output:
            if self.n_scalars_in == 0:
                raise ValueError("Cannot produce scalar output with no scalar input features.")
            self.inv_readout = readout_latent(
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **readout_latent_kwargs,
            )

        # --- Equivariant (Vectorial) Readout ---
        self.reshape_in: Optional[reshape_irreps] = None
        self.eq_readout = None
        self.weights_emb = None
        self.reshape_back_features = None
        self.use_internal_weights = self.n_scalars_in == 0

        if self.has_equivariant_output:
            eq_in_irreps = o3.Irreps([(mul, ir) for mul, ir in in_irreps if ir.l > 0])
            eq_out_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l > 0])
            self.reshape_in = reshape_irreps(eq_in_irreps)

            self.eq_readout = Linear(eq_in_irreps, eq_out_irreps, internal_weights=self.use_internal_weights, pad_to_alignment=1)
            
            if not self.use_internal_weights:
                self.weights_emb = readout_latent(self.n_scalars_in, self.eq_readout.weight_numel, **readout_latent_kwargs)
                if self.conditioner is not None:
                     self.conditioner["film_vectorial"] = FiLMFunction(self.irreps_in[self.conditioning_field].dim, [], self.eq_readout.weight_numel, mlp_nonlinearity=None)

            self.reshape_back_features = inverse_reshape_irreps(eq_out_irreps)
        elif strict_irreps and in_irreps.dim > self.n_scalars_in:
            raise ValueError(
                f"Input for field '{self.field}' contains non-scalar irreps ({in_irreps}), "
                f"but output is all scalars ({out_irreps}). Non-scalar features would be unused. "
                "To allow this, set 'strict_irreps=False'."
            )

        # --- Bias ---
        self.bias = None
        if self.has_invariant_output:
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
        if self.ignore_amp:
            with torch.amp.autocast('cuda', enabled=False):
                return self._forward_impl(data)
        return self._forward_impl(data)

    def _forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        
        scalars, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)

        if self.conditioner is not None and self.conditioning_field is not None:
            conditioning_tensor = data[self.conditioning_field]
            scalars = self.conditioner["film1"](scalars, conditioning_tensor)
            scalars = self.conditioner["fc1"](scalars)
            scalars = self.conditioner["film2"](scalars, conditioning_tensor)

        out_scalars_list = []
        if self.has_invariant_output:
            current_out_scalars = self.inv_readout(scalars)
            if self.conditioner is not None and "film_scalar" in self.conditioner:
                current_out_scalars = self.conditioner["film_scalar"](current_out_scalars, data[self.conditioning_field])
            out_scalars_list.append(current_out_scalars)

        out_equiv_list = []
        if self.has_equivariant_output:
            eq_features_in = self.reshape_in(equiv)
            
            if self.use_internal_weights:
                eq_features_out = self.eq_readout(eq_features_in)
            else:
                weights = self.weights_emb(scalars)
                if self.conditioner is not None and "film_vectorial" in self.conditioner:
                    weights = self.conditioner["film_vectorial"](weights, data[self.conditioning_field])
                eq_features_out = self.eq_readout(eq_features_in, weights)
            
            out_equiv_list.append(self.reshape_back_features(eq_features_out))

        output_components = out_scalars_list + out_equiv_list
        if not output_components:
            raise ValueError("ReadoutModule produced no output features.")
        out_features = torch.cat(output_components, dim=-1)
        
        if self.bias is not None:
            out_scalars = out_features[..., :self.n_scalars_out]
            out_equiv = out_features[..., self.n_scalars_out:]
            biased_scalars = out_scalars + self.bias
            if out_equiv.shape[-1] > 0:
                out_features = torch.cat([biased_scalars, out_equiv], dim=-1)
            else:
                out_features = biased_scalars

        if self.resnet:
            old_features = data[self.out_field]
            coeff = self._resnet_update_coeff.sigmoid()
            coefficient_old = torch.rsqrt(coeff.square() + 1)
            coefficient_new = coeff * coefficient_old
            out_features = coefficient_old * old_features + coefficient_new * out_features

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
        
        if self.field in _NODE_FIELDS:
            idx_key = 'batch'
        elif self.field in _GRAPH_FIELDS and dataset_mode == 'ensemble':
            idx_key = 'ensemble_index'
        else:
            self.scalar_attnt_enabled = False
            
        if self.scalar_attnt_enabled:
            in_irreps: o3.Irreps = self.irreps_in[self.field]
            self.ensemble_attnt1 = L0IndexedAttention(
                irreps_in=in_irreps, field=self.field, out_field=self.field, 
                num_heads=num_heads, idx_key=idx_key, update_mlp=True
            )
            self.ensemble_attnt2 = L0IndexedAttention(
                irreps_in=in_irreps, field=self.field, out_field=self.field, 
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
