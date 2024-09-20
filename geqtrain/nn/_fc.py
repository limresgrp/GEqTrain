""" Adapted from https://github.com/mir-group/allegro
"""

import torch
from typing import List, Optional
from math import sqrt
from collections import OrderedDict
from e3nn.math import normalize2mom
from e3nn.util.codegen import CodeGenMixin
from geqtrain.nn.nonlinearities import ShiftedSoftPlus, ShiftedSoftPlusModule


def select_nonlinearity(nonlinearity):
    if nonlinearity == 'ssp':
        non_lin_instance = ShiftedSoftPlusModule()
    elif nonlinearity == "silu":
        non_lin_instance = torch.nn.SiLU()
    elif nonlinearity == "selu":
        non_lin_instance = torch.nn.SELU()
    elif nonlinearity == "relu":
        non_lin_instance = torch.nn.ReLU()
    elif nonlinearity:
        raise ValueError(f'Nonlinearity {nonlinearity} is not supported')
    return non_lin_instance


class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """
        ScalarMLPFunction is a flexible Multi-Layer Perceptron (MLP) module designed to provide various configurations of MLPs,
        including options for weight normalization, normalization layers, and custom non-linearities.

        Attributes:
            in_features (int): The number of input features to the MLP.
            out_features (int): The number of output features from the MLP.
            use_weight_norm (bool): Flag indicating whether weight normalization is used.
            dim_weight_norm (int): Dimension along which to apply weight normalization.
            use_layer_norm (bool): Flag indicating whether normalization layers are used.

        Methods:
            __init__: Initializes the MLP with the specified parameters.
            forward: Performs a forward pass through the MLP.

        Parameters for Initialization:
            mlp_input_dimension (Optional[int]): Dimension of the input to the MLP.
            mlp_latent_dimensions (List[int]): List of dimensions for the hidden layers.
            mlp_output_dimension (Optional[int]): Dimension of the output of the MLP.
            mlp_nonlinearity (Optional[str]): Type of non-linearity to use ('silu', 'ssp', 'selu', or None).
            use_layer_norm (bool): Whether to use layer normalization.
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
                use_layer_norm=True,
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
    use_layer_norm: bool
    use_weight_norm: bool
    dim_weight_norm: int

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        use_layer_norm: bool = False,
        use_weight_norm: bool = False,
        dim_weight_norm: int = 0,
        has_bias: bool = False,
        bias: Optional[List] = None,
        zero_init_last_layer_weights: bool = False,
        dropout: Optional[float] = None,
        dampen: bool = False,
    ):
        super().__init__()

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
        self.use_layer_norm = use_layer_norm

        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlusModule,
            "selu": torch.nn.functional.selu,
            "relu": torch.nn.functional.relu,
        }[mlp_nonlinearity]

        nonlin_const = 1.0
        if nonlinearity is not None:
            if mlp_nonlinearity == "ssp":
                nonlin_const = normalize2mom(ShiftedSoftPlus).cst
            elif mlp_nonlinearity == "selu" or mlp_nonlinearity == "relu":
                nonlin_const = torch.nn.init.calculate_gain(mlp_nonlinearity, param=None)
            else:
                nonlin_const = normalize2mom(nonlinearity).cst

        if bias is not None:
            has_bias = True

        sequential_dict = OrderedDict()

        if self.use_layer_norm:
            sequential_dict['norm'] = torch.nn.LayerNorm(dimensions[0])

        for layer_index, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            bias_condition = False if (layer_index == 0 and self.use_layer_norm) else has_bias
            lin_layer = torch.nn.Linear(h_in, h_out, bias=bias_condition)

            is_last_layer = layer_index == num_layers - 1
            if (nonlinearity is None) or is_last_layer:
                norm_const = 1.
                modules = [(f"linear_{layer_index}", lin_layer)]
            else:
                norm_const = nonlin_const
                non_lin_instance = select_nonlinearity(mlp_nonlinearity)
                modules = [
                        (f"linear_{layer_index}", lin_layer),
                        (f"activation_{layer_index}", non_lin_instance),
                ]
            
            if zero_init_last_layer_weights:
                norm_const = norm_const * 1.e-1

            with torch.no_grad():
                torch.nn.init.orthogonal_(lin_layer.weight, gain=norm_const)
                if lin_layer.bias is not None:
                    if is_last_layer and bias is not None:
                        lin_layer.bias.data = torch.tensor(bias).reshape(*lin_layer.bias.data.shape)
                    else:
                        torch.nn.init.zeros_(lin_layer.bias)

            # Apply weight normalization if specified, must be done after weight initialization
            if self.use_weight_norm:
                if int(torch.__version__.split('.')[0]) >= 2:
                    from torch.nn.utils.parametrizations import weight_norm
                else:
                    from torch.nn.utils import weight_norm
                lin_layer = weight_norm(lin_layer, name='weight', dim=self.dim_weight_norm)

            for module in modules:
                module_name, mod = module
                sequential_dict[module_name] = mod

        if dropout is not None:
            assert 0 <= dropout < 1., f"Dropout must be a float in range [0., 1.). Got {dropout} ({type(dropout)})"
            sequential_dict["dropout"] = torch.nn.Dropout(dropout)

        self.sequential = torch.nn.Sequential(sequential_dict)

        if dampen:
            for p in self.parameters():
                p.tag = 'dampen'

    def forward(self, x):
        return self.sequential(x)