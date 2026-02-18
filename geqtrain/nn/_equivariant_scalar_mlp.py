import math
from typing import Optional, Union, Tuple
 
import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.nn import ScalarMLPFunction, SO3_Linear
from geqtrain.nn._film import FiLMFunction
from geqtrain.nn.allegro import Linear
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.utils.so3 import split_irreps


def _has_common_mul(irreps: o3.Irreps) -> Tuple[bool, int]:
    irreps = o3.Irreps(irreps)
    if len(irreps) == 0:
        return True, 0
    m0 = int(irreps[0].mul)
    return all(int(mul) == m0 for mul, _ in irreps), m0


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
    __constants__ = [
        "input_mode",
        "output_mode",
        "output_is_split",
        "use_so3_linear",
        "output_shape_spec",
        "use_full_feature_set",
        "has_equivariant_input",
        "in_multiplicity",
        "out_multiplicity",
        "in_has_common_mul",
        "out_has_common_mul",
        "equiv_out_has_common_mul",
    ]

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
        self.input_mode = "single" if isinstance(in_irreps, (o3.Irreps, int, str))  else "split"
        self.output_mode = "single" if isinstance(out_irreps, (o3.Irreps, int, str)) else "split"
        self.output_is_split = self.output_mode == "split"
        self.conditioning_dim = conditioning_dim
        self.use_full_feature_set = False
        self.use_so3_linear = False
        # --- Determine Processor Input Irreps ---
        if output_shape_spec not in ["flat", "channel_wise", "input"]:
            raise ValueError(f"output_shape_spec must be 'flat', 'channel_wise', or 'input', but got {output_shape_spec}")

        self.output_shape_spec = output_shape_spec
        full_in_irreps = o3.Irreps("")
        if self.input_mode == "single":
            in_irreps = self._convert_to_o3_irreps(in_irreps)
            self.in_irreps_scalar, self.in_irreps_equiv = split_irreps(in_irreps)
            full_in_irreps = in_irreps
        elif self.input_mode == "split":
            if not isinstance(in_irreps, tuple) or len(in_irreps) != 2:
                raise ValueError("For input_mode='split', in_irreps must be a tuple of (scalar_irreps, equivariant_irreps).")
            in_irreps_scalar, in_irreps_equiv = in_irreps
            self.in_irreps_scalar = self._convert_to_o3_irreps(in_irreps_scalar)
            self.in_irreps_equiv = self._convert_to_o3_irreps(in_irreps_equiv)
            full_in_irreps = self.in_irreps_scalar + self.in_irreps_equiv
            if not all(ir.l == 0 for _, ir in self.in_irreps_scalar):
                raise ValueError(f"Scalar part of split input contains non-scalar irreps: {self.in_irreps_scalar}")
        else:
            raise ValueError(f"Invalid input_mode: {self.input_mode}")

        # --- Determine Processor Output Irreps ---
        full_out_irreps = o3.Irreps("")
        if self.output_mode == "single":
            out_irreps = self._convert_to_o3_irreps(out_irreps)
            self.out_irreps_scalar, self.out_irreps_equiv = split_irreps(out_irreps)
            full_out_irreps = out_irreps
        elif self.output_mode == "split":
            if not isinstance(out_irreps, tuple) or len(out_irreps) != 2:
                raise ValueError("For output_mode='split', out_irreps must be a tuple of (scalar_irreps, equivariant_irreps).")
            out_irreps_scalar, out_irreps_equiv = out_irreps
            self.out_irreps_scalar = self._convert_to_o3_irreps(out_irreps_scalar)
            self.out_irreps_equiv = self._convert_to_o3_irreps(out_irreps_equiv)
            full_out_irreps = self.out_irreps_scalar + self.out_irreps_equiv
            # Determine the correct input irreps for the equivariant readout
            # If output is split and the equivariant part contains scalars,
            # the linear layer needs the full feature set to compute them.
            if any(ir.l == 0 for _, ir in self.out_irreps_equiv):
                self.in_irreps_equiv = (self.in_irreps_scalar + self.in_irreps_equiv)
                self.use_full_feature_set = True

            if not all(ir.l == 0 for _, ir in self.out_irreps_scalar):
                raise ValueError(f"Scalar part of split output contains non-scalar irreps: {self.out_irreps_scalar}")
        else:
            raise ValueError(f"Invalid output_mode: {self.output_mode}")

        self.in_has_common_mul, in_mul = _has_common_mul(full_in_irreps)
        self.out_has_common_mul, out_mul = _has_common_mul(full_out_irreps)
        self.equiv_out_has_common_mul, _ = _has_common_mul(self.out_irreps_equiv)

        # Store multiplicities as attributes for TorchScript compatibility
        self.in_multiplicity: int = 1
        if self.in_has_common_mul and in_mul > 0:
            self.in_multiplicity = int(in_mul)
        elif len(self.in_irreps_scalar) > 0:
            self.in_multiplicity = int(self.in_irreps_scalar[0].mul)
        self.out_multiplicity: int = 1
        if self.out_has_common_mul and out_mul > 0:
            self.out_multiplicity = int(out_mul)
        elif len(self.out_irreps_scalar) > 0:
            self.out_multiplicity = int(self.out_irreps_scalar[0].mul)

        # --- Feature Properties ---
        self.n_scalars_in = self.in_irreps_scalar.dim
        self.n_scalars_out = sum(mul * ir.dim for mul, ir in self.out_irreps_scalar)
        self.has_invariant_output = self.n_scalars_out > 0
        self.has_equivariant_output = self.out_irreps_equiv.dim > 0
        self.has_equivariant_input = self.in_irreps_equiv.dim > 0
        has_conditioning = self.conditioning_dim > 0

        if (
            self.output_shape_spec == "channel_wise"
            and self.output_mode == "single"
            and self.has_equivariant_output
            and not self.out_has_common_mul
        ):
            raise ValueError(
                "output_shape_spec='channel_wise' requires single output irreps with a common multiplicity. "
                f"Got out_irreps scalar={self.out_irreps_scalar}, equiv={self.out_irreps_equiv}."
            )
        if (
            self.output_shape_spec == "channel_wise"
            and self.output_mode == "single"
            and self.has_equivariant_output
            and self.n_scalars_out % self.out_multiplicity != 0
        ):
            raise ValueError(
                f"When output_shape_spec is 'channel_wise', scalar output dim ({self.n_scalars_out}) "
                f"must be divisible by output multiplicity ({self.out_multiplicity})."
            )

        # --- Conditioning Layers ---
        self.conditioner = None
        # Only build FiLM when scalar inputs exist; otherwise the FiLM MLP would
        # have zero outputs and create empty parameters.
        if has_conditioning and self.n_scalars_in > 0:
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
        self.output_reshaper = None # Converts flat output to channel_wise
        self.output_flattener = None # Converts channel_wise output to flat

        if self.has_equivariant_output:
            requested_linear = equiv_linear_module
            # Allegro Linear supports only common multiplicities. Fall back to SO3_Linear when needed.
            if requested_linear is Linear:
                in_ok, _ = _has_common_mul(self.in_irreps_equiv)
                out_ok, _ = _has_common_mul(self.out_irreps_equiv)
                if not (in_ok and out_ok):
                    requested_linear = SO3_Linear
            self.use_so3_linear = requested_linear is SO3_Linear

            # Case 1: No input scalars AND no conditioning -> Use internal, learnable weights
            if self.n_scalars_in == 0 and not has_conditioning:
                try:
                    self.eq_readout_internal = requested_linear(
                        self.in_irreps_equiv,
                        self.out_irreps_equiv,
                        internal_weights=True,
                        shared_weights=True,
                        pad_to_alignment=1,
                    )
                except Exception:
                    if requested_linear is Linear:
                        self.use_so3_linear = True
                        self.eq_readout_internal = SO3_Linear(
                            self.in_irreps_equiv,
                            self.out_irreps_equiv,
                            internal_weights=True,
                            shared_weights=True,
                            pad_to_alignment=1,
                        )
                    else:
                        raise
            else:
                # Case 2 & 3: Weights are generated externally from either scalars or conditioning tensor
                try:
                    self.eq_readout = requested_linear(
                        self.in_irreps_equiv,
                        self.out_irreps_equiv,
                        internal_weights=False,
                        pad_to_alignment=1,
                    )
                except Exception:
                    if requested_linear is Linear:
                        self.use_so3_linear = True
                        self.eq_readout = SO3_Linear(
                            self.in_irreps_equiv,
                            self.out_irreps_equiv,
                            internal_weights=False,
                            pad_to_alignment=1,
                        )
                    else:
                        raise
                
                # Case 2: Input scalars are present. Use them to generate weights.
                if self.n_scalars_in > 0:
                    assert self.eq_readout is not None
                    self.weights_emb = latent_module(mlp_input_dimension=self.n_scalars_in, mlp_output_dimension=self.eq_readout.weight_numel, **latent_kwargs)
                # Case 3: No input scalars, but conditioning is present. Use conditioning tensor to generate weights.
                elif has_conditioning:
                    assert self.eq_readout is not None
                    # The MLP generates weight modulations from the conditioning tensor.
                    # We add a learnable bias which acts as the default weights when conditioning is zero.
                    self.weights_emb = latent_module(mlp_input_dimension=self.conditioning_dim, mlp_output_dimension=self.eq_readout.weight_numel, has_bias=True, **latent_kwargs)
                    # Initialize the MLP's final layer weights to zero, so the output is initially just the bias.
                    with torch.no_grad():
                        self.weights_emb.sequential[-1].weight.zero_()
                        b = self.weights_emb.sequential[-1].bias
                        b.fill_(1./math.sqrt(len(b)))
            
            if self.equiv_out_has_common_mul:
                self.output_reshaper = reshape_irreps(self.out_irreps_equiv)
                self.output_flattener = inverse_reshape_irreps(self.out_irreps_equiv)
        elif strict_irreps and self.in_irreps_equiv.dim > 0:
             raise ValueError(
                 f"Input contains non-scalar irreps ({self.in_irreps_equiv}), "
                 f"but output is all scalars ({self.out_irreps_scalar}). Non-scalar features would be unused. "
                 "To allow this, set 'strict_irreps=False'."
             )

    def forward(
        self,
        features: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        conditioning_tensor: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:

        # -- 1. Unpack inputs --
        equiv = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.input_mode == "split":
            if not torch.jit.isinstance(features, Tuple[torch.Tensor, Optional[torch.Tensor]]):
                raise ValueError("input_mode is 'split', but 'features' was not a tuple.")
            scalars, equiv = features
            input_is_channel_wise = (equiv is not None and len(equiv.shape) == 3)
            if equiv is None and self.has_equivariant_input:
                raise RuntimeError("Equivariant input features are required by in_irreps but were provided as None.")
        else:
            if torch.jit.isinstance(features, Tuple[torch.Tensor, torch.Tensor]):
                raise ValueError("input_mode is 'single', but 'features' was a tuple.")
            if not torch.jit.isinstance(features, torch.Tensor):
                raise ValueError("input_mode is 'single', but 'features' was not a Tensor.")
            features_tensor = features
            # Input is a single tensor, split it
            input_is_channel_wise = len(features_tensor.shape) == 3
            if input_is_channel_wise:
                if not self.in_has_common_mul:
                    raise ValueError(
                        "Channel-wise input was provided, but input irreps do not share a common multiplicity. "
                        "Use flat input/output for mixed multiplicities."
                    )
                # Input is channel-wise: [N, C, D_channel]
                # We need to extract scalars and reshape them to be flat for the MLP.
                # The number of scalar features per channel is self.n_scalars_in // multiplicity
                n_scalar_channels = self.n_scalars_in // self.in_multiplicity
                scalars_channel, equiv_channel = torch.split(features_tensor, [n_scalar_channels, features_tensor.shape[-1] - n_scalar_channels], dim=-1)
                scalars = scalars_channel.reshape(features_tensor.shape[0], -1) # [N, C, D_scalar_channel] -> [N, C * D_scalar_channel]
                equiv = equiv_channel # This part remains channel-wise for the equivariant readout
            else:
                # Input is flat (2D tensor)
                if len(features_tensor.shape) != 2:
                    raise ValueError(f"Expected a 2D tensor for flat input, but got shape {features_tensor.shape}")
                # Input is flat: [N, D_flat]
                scalars, equiv = torch.split(features_tensor, [self.n_scalars_in, features_tensor.shape[-1] - self.n_scalars_in], dim=-1)

        # -- 2. Apply conditioning to scalar features --
        if self.conditioning_dim > 0 and self.n_scalars_in > 0 and conditioning_tensor is not None:
            assert self.conditioner is not None
            scalars = self.conditioner["film"](scalars, conditioning_tensor)

        # -- 3. Process scalar features --
        out_scalars: Optional[torch.Tensor] = None
        if self.has_invariant_output:
            assert self.scalar_processor is not None
            processed_scalars = self.scalar_processor(scalars)

            output_is_channel_wise = (
                self.output_shape_spec == "channel_wise"
                or (self.output_shape_spec == "input" and input_is_channel_wise)
            )
            if (
                output_is_channel_wise
                and self.output_mode == "single"
                and self.has_equivariant_output
                and not self.out_has_common_mul
            ):
                output_is_channel_wise = False

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
            if equiv is None:
                raise RuntimeError("EquivariantScalarMLP expects equivariant inputs but received None.")
            equiv_features = equiv
            # If output is split and the equivariant part contains scalars,
            # the linear layer needs the full feature set to compute them.
            if self.use_full_feature_set:
                if input_is_channel_wise:
                    if not self.in_has_common_mul:
                        raise ValueError(
                            "Cannot merge scalar and equivariant channel-wise features without common input multiplicity."
                        )
                    # Reshape scalars from [N, D_scalar_total] back to [N, C, D_scalar_per_channel]
                    n_scalar_channels = self.n_scalars_in // self.in_multiplicity
                    scalars_channel_wise = scalars.view(scalars.shape[0], self.in_multiplicity, n_scalar_channels)
                    equiv_features = torch.cat([scalars_channel_wise, equiv_features], dim=-1)
                else:
                    # For flat input, simple concatenation is correct
                    equiv_features = torch.cat([scalars, equiv_features], dim=-1)

            if self.eq_readout_internal is not None:
                assert self.eq_readout_internal is not None
                eq_features_out = self.eq_readout_internal(equiv_features)
            else:
                assert self.eq_readout is not None
                assert self.weights_emb is not None
                if self.n_scalars_in > 0:
                    weights = self.weights_emb(scalars)
                else:
                    assert conditioning_tensor is not None, "Conditioning tensor must be provided when n_scalars_in is 0 and conditioning is enabled."
                    weights = self.weights_emb(conditioning_tensor)
                
                eq_features_out = self.eq_readout(equiv_features, weights)
            
            # Determine the desired output shape based on output_shape_spec and input shape
            is_channel_wise_output_desired = (
                self.output_shape_spec == "channel_wise" or
                (self.output_shape_spec == "input" and input_is_channel_wise)
            )
            if (
                is_channel_wise_output_desired
                and self.output_mode == "single"
                and not self.out_has_common_mul
            ):
                is_channel_wise_output_desired = False
            is_flat_output_desired = not is_channel_wise_output_desired

            if is_channel_wise_output_desired:
                if len(eq_features_out.shape) == 3:
                    out_equiv = eq_features_out
                else:
                    if self.output_reshaper is None:
                        raise ValueError(
                            "Channel-wise output requested for equivariant features, but output irreps "
                            "do not support channel-wise reshaping."
                        )
                    out_equiv = self.output_reshaper(eq_features_out)
            elif is_flat_output_desired:
                if len(eq_features_out.shape) == 3:
                    if self.output_flattener is None:
                        raise ValueError(
                            "Flat output requested, but received channel-wise equivariant features without "
                            "a valid flattening map."
                        )
                    out_equiv = self.output_flattener(eq_features_out)
                else:
                    out_equiv = eq_features_out
            else:
                out_equiv = eq_features_out

        # -- 5. Combine and return outputs --
        if out_scalars is None and out_equiv is None:
            raise ValueError("EquivariantScalarMLP produced no output features.")

        if self.output_is_split:
            return torch.jit.annotate(
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
                (out_scalars, out_equiv),
            )
        else:
            # output_mode is "single"
            if out_scalars is None:
                if out_equiv is None:
                    raise ValueError("EquivariantScalarMLP produced no output features.")
                return out_equiv
            elif out_equiv is None:
                return out_scalars
            else:
                if len(out_scalars.shape) != len(out_equiv.shape):
                    if len(out_scalars.shape) == 3 and self.output_flattener is not None:
                        out_scalars = out_scalars.reshape(out_scalars.shape[0], -1)
                    if len(out_equiv.shape) == 3:
                        if self.output_flattener is None:
                            raise ValueError("Cannot concatenate scalar and equivariant outputs with incompatible shapes.")
                        out_equiv = self.output_flattener(out_equiv)
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
