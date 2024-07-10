""" Adapted from https://github.com/mir-group/allegro
"""

import torch
from torch import _weight_norm, norm_except_dim
from torch import fx
from typing import List, Optional
from math import sqrt
from collections import OrderedDict
from e3nn.math import normalize2mom
from e3nn.util.codegen import CodeGenMixin
from geqtrain.nn.nonlinearities import ShiftedSoftPlus, ShiftedSoftPlusModule


class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """
        ScalarMLPFunction is a flexible Multi-Layer Perceptron (MLP) module designed to provide various configurations of MLPs, 
        including options for weight normalization, normalization layers, and custom non-linearities.

        Attributes:
            in_features (int): The number of input features to the MLP.
            out_features (int): The number of output features from the MLP.
            use_weight_norm (bool): Flag indicating whether weight normalization is used.
            dim_weight_norm (int): Dimension along which to apply weight normalization.
            use_norm_layer (bool): Flag indicating whether normalization layers are used.

        Methods:
            __init__: Initializes the MLP with the specified parameters.
            forward: Performs a forward pass through the MLP.

        Parameters for Initialization:
            mlp_input_dimension (Optional[int]): Dimension of the input to the MLP.
            mlp_latent_dimensions (List[int]): List of dimensions for the hidden layers.
            mlp_output_dimension (Optional[int]): Dimension of the output of the MLP.
            mlp_nonlinearity (Optional[str]): Type of non-linearity to use ('silu', 'ssp', 'selu', or None).
            use_norm_layer (bool): Whether to use layer normalization.
            use_weight_norm (bool): Whether to use weight normalization.
            dim_weight_norm (int): Dimension along which to apply weight normalization.
            has_bias (bool): Whether the linear layers have bias.
            bias (Optional[List]): List of values to initialize the bias of the last linear layer.
            zero_init_last_layer_weights (bool): Whether to initialize the weights of the last layer to zero.

        Notes:
            - The non-linearity functions available are 'silu', 'ssp', and 'selu'.
            - Weight initialization follows the principles of Kaiming initialization for better convergence.
            - This module leverages the e3nn library for certain mathematical operations.

        Example Usage:
            mlp = ScalarMLPFunction(
                mlp_input_dimension=128,
                mlp_latent_dimensions=[256, 256],
                mlp_output_dimension=10,
                mlp_nonlinearity='silu',
                use_norm_layer=True,
                use_weight_norm=True,
                dim_weight_norm=1,
                has_bias=True,
                bias=[0.1, 0.2, 0.3, ...],
                zero_init_last_layer_weights=True
            )
            output = mlp(torch.randn(32, 128))
    """

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
        bias: Optional[List] = None,
        zero_init_last_layer_weights: bool = False,
    ):
        super().__init__()
        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlusModule,
            "selu": torch.nn.functional.selu,
        }[mlp_nonlinearity]

        nonlin_const = 1.0
        if nonlinearity is not None:
            if mlp_nonlinearity == "ssp":
                nonlin_const = normalize2mom(ShiftedSoftPlus).cst
            elif mlp_nonlinearity == "selu":
                nonlin_const = torch.nn.init.calculate_gain(mlp_nonlinearity, param=None)
            else:
                nonlin_const = normalize2mom(nonlinearity).cst

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

        sequential_dict = OrderedDict()
        if self.use_norm_layer:
            sequential_dict['layer_norm'] = torch.nn.LayerNorm(dimensions[0])
        
        if bias is not None:
            has_bias = True

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            lin_layer = torch.nn.Linear(h_in, h_out, bias=has_bias)

            is_last_layer = num_layers - 1 == layer
            if (nonlinearity is not None) and (not is_last_layer):
                if mlp_nonlinearity == 'ssp':
                    non_lin_instance = ShiftedSoftPlusModule()
                elif mlp_nonlinearity == "silu":
                    non_lin_instance = torch.nn.SiLU()
                elif mlp_nonlinearity == "selu":
                    non_lin_instance = torch.nn.SELU()
                elif mlp_nonlinearity:
                    raise ValueError(f'Nonlinearity {nonlinearity} is not supported')

                with torch.no_grad():
                    # as in: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
                    lin_layer.weight = lin_layer.weight.normal_(0, nonlin_const / sqrt(float(h_in)))
                sequential_dict[f"{layer}_activated"] = torch.nn.Sequential(
                    OrderedDict([
                        ("linear", lin_layer),
                        ("activation", non_lin_instance),
                    ]),
                )

            else:
                with torch.no_grad():
                    # as in: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
                    lin_layer.weight = lin_layer.weight.normal_(0, 1. / sqrt(float(h_in)))
                sequential_dict[f"{layer}"] = lin_layer

        self.sequential = torch.nn.Sequential(sequential_dict)
        if has_bias and bias is not None:
            self.sequential[-1].bias.data = torch.tensor(bias).reshape(*self.sequential[-1].bias.data.shape)
        if zero_init_last_layer_weights:
            self.sequential[-1].weight.data = self.sequential[-1].weight.data * 1.e-3

    def forward(self, x):
        return self.sequential(x)