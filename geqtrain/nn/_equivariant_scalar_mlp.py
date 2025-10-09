from typing import List, Optional, Union

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.nn import ScalarMLPFunction
from geqtrain.nn._film import FiLMFunction
from geqtrain.nn.allegro import Linear
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.utils.tp_utils import PSEUDO_SCALAR, SCALAR


@compile_mode("script")
class EquivariantScalarMLP(nn.Module):
    """
    A module that processes a feature tensor containing both scalar and equivariant parts.
    It applies an MLP to the scalar part and a weighted linear transformation to the
    equivariant part, with optional conditioning.
    """
    def __init__(
        self,
        in_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        conditioning_dim: int = 0,
        latent_module=ScalarMLPFunction,
        latent_kwargs={},
        strict_irreps: bool = True,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.conditioning_dim = conditioning_dim

        # --- Feature Properties ---
        self.n_scalars_in = sum(mul for mul, ir in in_irreps if ir.l == 0)
        self.n_scalars_out = sum(mul for mul, ir in out_irreps if ir.l == 0)
        self.has_invariant_output = self.n_scalars_out > 0
        self.has_equivariant_output = out_irreps.dim > self.n_scalars_out
        self.split_index = sum(mul for mul, ir in in_irreps if ir in [SCALAR, PSEUDO_SCALAR])

        has_conditioning = self.conditioning_dim > 0

        # --- Conditioning Layers ---
        self.conditioner = nn.ModuleDict()
        if has_conditioning and self.n_scalars_in > 0:
            self.conditioner = nn.ModuleDict({
                "film1": FiLMFunction(self.conditioning_dim, [], self.n_scalars_in, mlp_nonlinearity=None),
                "fc1": latent_module(mlp_input_dimension=self.n_scalars_in, mlp_output_dimension=self.n_scalars_in, **latent_kwargs),
                "film2": FiLMFunction(self.conditioning_dim, [], self.n_scalars_in, mlp_nonlinearity=None),
            })

        # --- Invariant (Scalar) Readout ---
        self.inv_readout = None
        if self.has_invariant_output:
            if self.n_scalars_in == 0:
                raise ValueError("Cannot produce scalar output with no scalar input features.")
            self.inv_readout = latent_module(
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **latent_kwargs,
            )

        # --- Equivariant (Vectorial) Readout ---
        self.reshape_in: Optional[reshape_irreps] = None
        self.eq_readout_internal = None
        self.eq_readout = None
        self.weights_emb = None
        self.reshape_back_features = None
        self.use_internal_weights = self.n_scalars_in == 0

        if self.has_equivariant_output:
            eq_in_irreps = o3.Irreps([(mul, ir) for mul, ir in in_irreps if ir.l > 0])
            assert len(eq_in_irreps) > 0, "No equivariant (l > 0) input irreps found. Cannot perform equivariant readout."
            eq_out_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l > 0])
            self.reshape_in = reshape_irreps(eq_in_irreps)

            if self.use_internal_weights:
                self.eq_readout_internal = Linear(eq_in_irreps, eq_out_irreps, internal_weights=True, pad_to_alignment=1)
            else:
                self.eq_readout = Linear(eq_in_irreps, eq_out_irreps, internal_weights=False, pad_to_alignment=1)
                self.weights_emb = latent_module(mlp_input_dimension=self.n_scalars_in, mlp_output_dimension=self.eq_readout.weight_numel, **latent_kwargs)

            self.reshape_back_features = inverse_reshape_irreps(eq_out_irreps)
        elif strict_irreps and in_irreps.dim > self.n_scalars_in:
            raise ValueError(
                f"Input contains non-scalar irreps ({in_irreps}), "
                f"but output is all scalars ({out_irreps}). Non-scalar features would be unused. "
                "To allow this, set 'strict_irreps=False'."
            )

    def forward(self, features: torch.Tensor, conditioning_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        scalars, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)

        if self.conditioner and conditioning_tensor is not None:
            scalars = self.conditioner["film1"](scalars, conditioning_tensor)
            scalars = self.conditioner["fc1"](scalars)
            scalars = self.conditioner["film2"](scalars, conditioning_tensor)

        out_scalars_list = []
        if self.has_invariant_output:
            assert self.inv_readout is not None
            current_out_scalars = self.inv_readout(scalars)
            out_scalars_list.append(current_out_scalars)

        out_equiv_list = []
        if self.has_equivariant_output:
            assert self.reshape_in is not None
            eq_features_in = self.reshape_in(equiv)
            
            if self.use_internal_weights:
                assert self.eq_readout_internal is not None
                eq_features_out = self.eq_readout_internal(eq_features_in)
            else:
                assert self.eq_readout is not None and self.weights_emb is not None
                weights = self.weights_emb(scalars)
                eq_features_out = self.eq_readout(eq_features_in, weights)
            
            assert self.reshape_back_features is not None
            out_equiv_list.append(self.reshape_back_features(eq_features_out))

        output_components = out_scalars_list + out_equiv_list
        if not output_components:
            raise ValueError("EquivariantScalarMLP produced no output features.")
        
        return torch.cat(output_components, dim=-1)