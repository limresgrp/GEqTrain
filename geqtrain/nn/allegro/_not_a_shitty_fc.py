""" Adapted from https://github.com/mir-group/allegro
"""

import torch
from torch import _weight_norm, norm_except_dim
from torch import fx
from typing import List, Optional
from math import sqrt
from e3nn.math import normalize2mom
from e3nn.util.codegen import CodeGenMixin
from geqtrain.nn.nonlinearities import ShiftedSoftPlus, ShiftedSoftPlusModule


class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int
    use_weight_norm: bool
    use_norm_layer: bool
    dim_weight_norm: int

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        use_norm_layer: bool = False,
        use_weight_norm: bool = False,
        dim_weight_norm: int = 0,
        has_bias: bool = False,
    ):
        super().__init__()
        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlusModule ,# ShiftedSoftPlus,
        }[mlp_nonlinearity]

        if nonlinearity is not None:
            if mlp_nonlinearity == "ssp":
                nonlin_const = normalize2mom(ShiftedSoftPlus).cst
            else:
                nonlin_const = normalize2mom(nonlinearity).cst
        else:
            nonlin_const = 1.0

        dimensions = (
            ([mlp_input_dimension] if mlp_input_dimension is not None else [])
            + mlp_latent_dimensions
            + ([mlp_output_dimension] if mlp_output_dimension is not None else [])
        )
        assert len(dimensions) >= 2  # Must have input and output dim_weight_norm
        num_layers = len(dimensions) - 1

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]
        self.use_weight_norm = use_weight_norm
        self.dim_weight_norm = dim_weight_norm
        self.use_norm_layer = use_norm_layer
        norm_from_last: float = 1.0

        self.base = [] #torch.nn.ModuleList([])
        if self.use_norm_layer:
            self.base.append(torch.nn.LayerNorm(dimensions[0]))

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):

            if has_bias:
                lin_layer = torch.nn.Linear(h_in, h_out, bias= True)
                # todo init bias
            else:
                lin_layer = torch.nn.Linear(h_in, h_out, bias= False)

            lin_layer.weight.data = lin_layer.weight.data * (norm_from_last / sqrt(float(h_in)))

            if nonlinearity is not None and layer < num_layers - 1:
                # add nonlinearity
                if mlp_nonlinearity == 'ssp':
                    non_lin_instance = ShiftedSoftPlusModule()
                elif mlp_nonlinearity == "silu":
                    non_lin_instance = torch.nn.SiLU()
                elif mlp_nonlinearity:
                    raise ValueError(f'Nonlinearity {nonlinearity} is not supported')

                self.base.append(torch.nn.Sequential(lin_layer, non_lin_instance))
                # add the normalization const in next layer
                norm_from_last = nonlin_const
            else:
                self.base.append(lin_layer)

        self.sequential = torch.nn.Sequential(*self.base)

    def forward(self, x):
        return self.sequential(x)