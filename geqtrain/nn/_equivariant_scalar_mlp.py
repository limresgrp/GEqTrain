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
        in_irreps: o3.Irreps = None,
        out_irreps: o3.Irreps = None,
        in_dim: int = None,
        out_dim: int = None,
        conditioning_dim: int = 0,
        latent_module=ScalarMLPFunction,
        latent_kwargs={},
        strict_irreps: bool = True,
        reshape_in: bool = True,
        reshape_back: bool = True,
    ):
        super().__init__()
        if in_irreps is None and in_dim is not None:
            in_irreps = o3.Irreps(f'{in_dim}x0e')
        if out_irreps is None and out_dim is not None:
            out_irreps = o3.Irreps(f'{out_dim}x0e')
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
            if reshape_in:
                self.reshape_in = reshape_irreps(eq_in_irreps)

            if self.use_internal_weights:
                self.eq_readout_internal = Linear(eq_in_irreps, eq_out_irreps, internal_weights=True, shared_weights=True, pad_to_alignment=1)
            else:
                self.eq_readout = Linear(eq_in_irreps, eq_out_irreps, internal_weights=False, pad_to_alignment=1)
                self.weights_emb = latent_module(mlp_input_dimension=self.n_scalars_in, mlp_output_dimension=self.eq_readout.weight_numel, **latent_kwargs)

            if reshape_back:
                self.reshape_back_features = inverse_reshape_irreps(eq_out_irreps)
        elif strict_irreps and in_irreps.dim > self.n_scalars_in:
            raise ValueError(
                f"Input contains non-scalar irreps ({in_irreps}), "
                f"but output is all scalars ({out_irreps}). Non-scalar features would be unused. "
                "To allow this, set 'strict_irreps=False'."
            )

    def forward(self, features: torch.Tensor, conditioning_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.reshape_in is None and self.has_equivariant_output:
            # Input is channel-wise: [N, C, D_channel]
            # We need to extract scalars and reshape them to be flat for the MLP.
            # The number of scalar features per channel is self.n_scalars_in // multiplicity
            n_scalar_channels = self.n_scalars_in // self.in_irreps[0].mul
            scalars_channel, equiv_channel = torch.split(features, [n_scalar_channels, features.shape[-1] - n_scalar_channels], dim=-1)
            scalars = scalars_channel.reshape(features.shape[0], -1) # [N, C, D_scalar_channel] -> [N, C * D_scalar_channel]
            equiv = equiv_channel # This part remains channel-wise for the equivariant readout
        else:
            # Input is flat: [N, D_flat]
            scalars, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)

        if self.conditioner and conditioning_tensor is not None:
            scalars = self.conditioner["film1"](scalars, conditioning_tensor)
            scalars = self.conditioner["fc1"](scalars)
            scalars = self.conditioner["film2"](scalars, conditioning_tensor)

        out_scalars_list = []
        if self.has_invariant_output:
            assert self.inv_readout is not None
            current_out_scalars = self.inv_readout(scalars) # [N, D_scalar]
            
            # If we have an un-flattened equivariant output, we need to reshape the scalars to match.
            # The equivariant output will be [N, C, D_equiv], so scalars should become [N, C, D_scalar].
            if self.has_equivariant_output and self.reshape_back_features is None:
                # Get multiplicity `C` from the output irreps
                multiplicity = self.out_irreps[0].mul 
                # Reshape scalars from [N, D_scalar_total] to [N, C, D_scalar_per_channel]
                num_scalar_channels = self.n_scalars_out // multiplicity
                if self.n_scalars_out % multiplicity != 0:
                    raise ValueError("When reshape_back=False, the number of scalar outputs must be divisible by the output multiplicity.")
                current_out_scalars = current_out_scalars.view(current_out_scalars.shape[0], multiplicity, num_scalar_channels)

            out_scalars_list.append(current_out_scalars)

        out_equiv_list = []
        if self.has_equivariant_output:
            if self.reshape_in is not None:
                eq_features_in = self.reshape_in(equiv) # [N, D_equiv_flat] -> [N, C, D_equiv_channel]
            else:
                eq_features_in = equiv # Already in [N, C, D_equiv_channel]
            
            if self.use_internal_weights:
                assert self.eq_readout_internal is not None
                eq_features_out = self.eq_readout_internal(eq_features_in)
            else:
                assert self.eq_readout is not None and self.weights_emb is not None
                weights = self.weights_emb(scalars)
                eq_features_out = self.eq_readout(eq_features_in, weights)
            
            if self.reshape_back_features is not None:
                out_equiv_list.append(self.reshape_back_features(eq_features_out))
            else:
                out_equiv_list.append(eq_features_out)

        output_components = out_scalars_list + out_equiv_list
        if not output_components:
            raise ValueError("EquivariantScalarMLP produced no output features.")
        
        return torch.cat(output_components, dim=-1)