import math
from typing import List, Optional, Union, Tuple
 
import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.nn import ScalarMLPFunction, SO3_Linear
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

    It can take a single feature tensor and split it internally, or it can take
    pre-split scalar and equivariant tensors. It can also return a single concatenated
    tensor or a tuple of (scalars, equivariants).
    """
    def __init__(
        self,
        in_irreps: Union[str, int, o3.Irreps, Tuple[o3.Irreps, o3.Irreps]] = None,
        out_irreps: Union[str, int, o3.Irreps, Tuple[o3.Irreps, o3.Irreps]] = None,
        conditioning_dim: int = 0,
        latent_module=ScalarMLPFunction,
        latent_kwargs={},
        equiv_linear_module=Linear, # Linear is much faster but SO3_Linear can handle irreps with different multiplicities (e.g. 10x0e+5x1o)
        strict_irreps: bool = True,
        output_shape_spec: str = "input", # "flat", "channel_wise", or "input"
    ):
        super().__init__()
        self.input_mode = "split" if isinstance(in_irreps, (Tuple)) else "single"
        self.output_mode = "split" if isinstance(out_irreps, (Tuple)) else "single"
        self.conditioning_dim = conditioning_dim

        self.equiv_linear_module = equiv_linear_module
        # --- Determine Processor Input Irreps ---
        if output_shape_spec not in ["flat", "channel_wise", "input"]:
            raise ValueError(f"output_shape_spec must be 'flat', 'channel_wise', or 'input', but got {output_shape_spec}")

        self.output_shape_spec = output_shape_spec
        if self.input_mode == "single":
            in_irreps = self._convert_to_o3_irreps(in_irreps)
            self.in_irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in in_irreps if ir.l == 0])
            self.in_irreps_equiv = o3.Irreps([(mul, ir) for mul, ir in in_irreps if ir.l > 0])
        elif self.input_mode == "split":
            if not isinstance(in_irreps, tuple) or len(in_irreps) != 2:
                raise ValueError("For input_mode='split', in_irreps must be a tuple of (scalar_irreps, equivariant_irreps).")
            in_irreps_scalar, in_irreps_equiv = in_irreps
            self.in_irreps_scalar = self._convert_to_o3_irreps(in_irreps_scalar)
            self.in_irreps_equiv = self._convert_to_o3_irreps(in_irreps_equiv)
            if not all(ir.l == 0 for _, ir in self.in_irreps_scalar):
                raise ValueError(f"Scalar part of split input contains non-scalar irreps: {self.in_irreps_scalar}")
        else:
            raise ValueError(f"Invalid input_mode: {self.input_mode}")

        # --- Determine Processor Output Irreps ---
        if self.output_mode == "single":
            out_irreps = self._convert_to_o3_irreps(out_irreps)
            self.out_irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l == 0])
            self.out_irreps_equiv = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l > 0])
        elif self.output_mode == "split":
            if not isinstance(out_irreps, tuple) or len(out_irreps) != 2:
                raise ValueError("For output_mode='split', out_irreps must be a tuple of (scalar_irreps, equivariant_irreps).")
            out_irreps_scalar, out_irreps_equiv = out_irreps
            self.out_irreps_scalar = self._convert_to_o3_irreps(out_irreps_scalar)
            self.out_irreps_equiv = self._convert_to_o3_irreps(out_irreps_equiv)
            if not all(ir.l == 0 for _, ir in self.out_irreps_scalar):
                raise ValueError(f"Scalar part of split output contains non-scalar irreps: {self.out_irreps_scalar}")
        else:
            raise ValueError(f"Invalid output_mode: {self.output_mode}")

        # Store multiplicities as attributes for TorchScript compatibility
        self.in_multiplicity: int = 1
        if len(self.in_irreps_scalar) > 0:
            self.in_multiplicity = self.in_irreps_scalar[0].mul
        self.out_multiplicity: int = 1
        if len(self.out_irreps_scalar) > 0:
            self.out_multiplicity = self.out_irreps_scalar[0].mul

        # If output is channel-wise, the number of scalar outputs must be divisible by the output multiplicity.
        # This check is done at init time for clarity.
        if self.output_shape_spec == "channel_wise" and self.output_mode == "single" and len(self.out_irreps_equiv) > 0:
            if self.out_irreps_scalar.dim % self.out_multiplicity != 0:
                raise ValueError(f"When output_shape_spec is 'channel_wise' and output is single, the total dimension of scalar outputs ({self.out_irreps_scalar.dim}) must be divisible by the output multiplicity ({self.out_multiplicity}). Check your `out_irreps`.")

        # --- Feature Properties ---
        self.n_scalars_in = self.in_irreps_scalar.dim
        self.n_scalars_out = sum(mul * ir.dim for mul, ir in self.out_irreps_scalar)
        self.has_invariant_output = self.n_scalars_out > 0
        self.has_equivariant_output = self.out_irreps_equiv.dim > 0
        has_conditioning = self.conditioning_dim > 0

        # --- Conditioning Layers ---
        self.conditioner = None
        if has_conditioning:
            self.conditioner = nn.ModuleDict({
                "film": FiLMFunction(self.conditioning_dim, [], self.n_scalars_in, mlp_nonlinearity=None)
            })

        # --- Invariant (Scalar) Readout ---
        self.scalar_processor = None
        if self.has_invariant_output:
            if self.n_scalars_in == 0:
                raise ValueError("Cannot produce scalar output with no scalar input features.")
            
            # The main MLP for processing scalar features
            self.scalar_processor = latent_module(
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **latent_kwargs,
            )

        # --- Equivariant (Vectorial) Readout ---
        self.eq_readout_internal = None
        self.eq_readout = None # This will be an e3nn.nn.Linear or SO3_Linear
        self.weights_emb = None
        self.output_reshaper = None # Converts flat output of SO3_Linear to channel_wise
        self.output_flattener = None # Converts channel_wise output of Linear to flat

        if self.has_equivariant_output:
            # Case 1: No input scalars AND no conditioning -> Use internal, learnable weights
            if self.n_scalars_in == 0 and not has_conditioning:
                self.eq_readout_internal = equiv_linear_module(self.in_irreps_equiv, self.out_irreps_equiv, internal_weights=True, shared_weights=True, pad_to_alignment=1)
            else:
                # Case 2 & 3: Weights are generated externally from either scalars or conditioning tensor
                self.eq_readout = equiv_linear_module(self.in_irreps_equiv, self.out_irreps_equiv, internal_weights=False, pad_to_alignment=1)
                
                # Case 2: Input scalars are present. Use them to generate weights.
                if self.n_scalars_in > 0:
                    self.weights_emb = latent_module(mlp_input_dimension=self.n_scalars_in, mlp_output_dimension=self.eq_readout.weight_numel, **latent_kwargs)
                # Case 3: No input scalars, but conditioning is present. Use conditioning tensor to generate weights.
                elif has_conditioning:
                    # The MLP generates weight modulations from the conditioning tensor.
                    # We add a learnable bias which acts as the default weights when conditioning is zero.
                    self.weights_emb = latent_module(mlp_input_dimension=self.conditioning_dim, mlp_output_dimension=self.eq_readout.weight_numel, has_bias=True, **latent_kwargs)
                    # Initialize the MLP's final layer weights to zero, so the output is initially just the bias.
                    with torch.no_grad():
                        self.weights_emb.sequential[-1].weight.zero_()
                        b = self.weights_emb.sequential[-1].bias
                        b.fill_(1./math.sqrt(len(b)))
            
            # SO3_Linear always outputs flat tensors. If a channel-wise output is desired,
            # we need to reshape its output.
            if self.equiv_linear_module is SO3_Linear:
                if self.output_shape_spec == "channel_wise" or self.output_shape_spec == "input":
                    self.output_reshaper = reshape_irreps(self.out_irreps_equiv)
            # e3nn.nn.Linear (the default) always outputs channel-wise tensors. If a flat
            # output is desired, we need to flatten its output.
            else: # it is Linear
                if self.output_shape_spec == "flat" or self.output_shape_spec == "input":
                    self.output_flattener = inverse_reshape_irreps(self.out_irreps_equiv)
        elif strict_irreps and self.in_irreps_equiv.dim > 0:
             raise ValueError(
                 f"Input contains non-scalar irreps ({self.in_irreps_equiv}), "
                 f"but output is all scalars ({self.out_irreps_scalar}). Non-scalar features would be unused. "
                 "To allow this, set 'strict_irreps=False'."
             )

    def forward(
        self,
        features: Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        conditioning_tensor: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:

        # -- 1. Unpack inputs --
        if self.input_mode == "split":
            if not isinstance(features, tuple):
                raise ValueError("input_mode is 'split', but 'features' was not a tuple.")
            scalars, equiv = features
            input_is_channel_wise = (equiv is not None and len(equiv.shape) == 3)
        else:
            # Input is a single tensor, split it
            input_is_channel_wise = len(features.shape) == 3
            if input_is_channel_wise:
                # Input is channel-wise: [N, C, D_channel]
                # We need to extract scalars and reshape them to be flat for the MLP.
                # The number of scalar features per channel is self.n_scalars_in // multiplicity
                if not self.has_equivariant_output:
                    raise ValueError("Channel-wise input (3D tensor) is only supported when there are equivariant features.")
                n_scalar_channels = self.n_scalars_in // self.in_multiplicity
                scalars_channel, equiv_channel = torch.split(features, [n_scalar_channels, features.shape[-1] - n_scalar_channels], dim=-1)
                scalars = scalars_channel.reshape(features.shape[0], -1) # [N, C, D_scalar_channel] -> [N, C * D_scalar_channel]
                equiv = equiv_channel # This part remains channel-wise for the equivariant readout
            else:
                # Input is flat (2D tensor)
                if len(features.shape) != 2:
                    raise ValueError(f"Expected a 2D tensor for flat input, but got shape {features.shape}")
                # Input is flat: [N, D_flat]
                scalars, equiv = torch.split(features, [self.n_scalars_in, features.shape[-1] - self.n_scalars_in], dim=-1)

        # -- 2. Apply conditioning to scalar features --
        if self.conditioning_dim > 0 and self.n_scalars_in > 0 and conditioning_tensor is not None:
            assert self.conditioner is not None
            scalars = self.conditioner["film"](scalars, conditioning_tensor)

        # -- 3. Process scalar features --
        out_scalars: Optional[torch.Tensor] = None
        if self.has_invariant_output:
            assert self.scalar_processor is not None
            processed_scalars = self.scalar_processor(scalars)

            output_is_channel_wise = self.output_shape_spec == "channel_wise" or \
                                     (self.output_shape_spec == "input" and input_is_channel_wise)

            # If we have an un-flattened equivariant output, we need to reshape the scalars to match.
            # The equivariant output will be [N, C, D_equiv], so scalars should become [N, C, D_scalar].
            if self.has_equivariant_output and output_is_channel_wise and self.output_mode == "single":
                # Reshape scalars from [N, D_scalar_total] to [N, C, D_scalar_per_channel]
                if self.out_multiplicity == 0:
                    raise ValueError("Cannot have channel-wise output with zero output multiplicity.")
                num_scalar_channels = self.n_scalars_out // self.out_multiplicity
                processed_scalars = processed_scalars.view(processed_scalars.shape[0], self.out_multiplicity, num_scalar_channels)

            out_scalars = processed_scalars

        # -- 4. Process equivariant features --
        out_equiv: Optional[torch.Tensor] = None
        if self.has_equivariant_output:
            if self.eq_readout_internal is not None:
                assert self.eq_readout_internal is not None
                eq_features_out = self.eq_readout_internal(equiv)
            else:
                assert self.eq_readout is not None
                assert self.weights_emb is not None
                if self.n_scalars_in > 0:
                    weights = self.weights_emb(scalars)
                else:
                    assert conditioning_tensor is not None, "Conditioning tensor must be provided when n_scalars_in is 0 and conditioning is enabled."
                    weights = self.weights_emb(conditioning_tensor)
                eq_features_out = self.eq_readout(equiv, weights)
            
            # Determine the desired output shape based on output_shape_spec and input shape
            is_channel_wise_output_desired = (
                self.output_shape_spec == "channel_wise" or
                (self.output_shape_spec == "input" and input_is_channel_wise)
            )
            is_flat_output_desired = not is_channel_wise_output_desired

            # SO3_Linear outputs flat. Reshape to channel-wise if needed.
            if self.equiv_linear_module is SO3_Linear:
                if self.output_reshaper is not None and is_channel_wise_output_desired:
                    out_equiv = self.output_reshaper(eq_features_out)
                else:
                    out_equiv = eq_features_out
            # Linear outputs channel-wise. Flatten if needed.
            else: # it is Linear
                if self.output_flattener is not None and is_flat_output_desired:
                    out_equiv = self.output_flattener(eq_features_out)
                else:
                    out_equiv = eq_features_out

        # -- 5. Combine and return outputs --
        if out_scalars is None and out_equiv is None:
            raise ValueError("EquivariantScalarMLP produced no output features.")

        if self.output_mode == "split":
            return out_scalars, out_equiv

        # output_mode is "single"
        if out_scalars is None:
            return out_equiv
        if out_equiv is None:
            return out_scalars
        
        # The first n_scalars_out dimensions of the concatenated tensor are the scalars
        return torch.cat([out_scalars, out_equiv], dim=-1)

    def _convert_to_o3_irreps(self, irreps_input: Union[str, int, o3.Irreps]) -> o3.Irreps:
        """Converts an integer, string, or existing o3.Irreps object to an o3.Irreps object."""
        if isinstance(irreps_input, int):
            return o3.Irreps(f"{irreps_input}x0e")
        elif isinstance(irreps_input, str):
            return o3.Irreps(irreps_input)
        elif isinstance(irreps_input, o3.Irreps):
            return irreps_input
        else:
            raise TypeError(f"Unsupported irreps type: {type(irreps_input)}. Expected int, str, or o3.Irreps.")