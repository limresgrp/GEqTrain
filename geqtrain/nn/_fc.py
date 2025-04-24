""" Adapted from https://github.com/mir-group/allegro
"""
import math
import torch
from typing import List, Optional
from collections import OrderedDict
from e3nn.math import normalize2mom
from e3nn.util.codegen import CodeGenMixin
from geqtrain.nn.nonlinearities import ShiftedSoftPlus, ShiftedSoftPlusModule, SwiGLUModule
from geqtrain.utils import add_tags_to_module
from e3nn.util.jit import compile_mode


def select_nonlinearity(nonlinearity):
    if nonlinearity == 'ssp': non_lin_instance = ShiftedSoftPlusModule()
    elif nonlinearity == "silu": non_lin_instance = torch.nn.SiLU()
    elif nonlinearity == "selu": non_lin_instance = torch.nn.SELU()
    elif nonlinearity == "relu": non_lin_instance = torch.nn.ReLU()
    elif nonlinearity == "swiglu": non_lin_instance = SwiGLUModule()
    elif nonlinearity == "sigmoid": non_lin_instance = torch.nn.Sigmoid()
    elif nonlinearity: raise ValueError(f'Nonlinearity {nonlinearity} is not supported')
    return non_lin_instance


@compile_mode("script")
class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """
        A Multi-Layer Perceptron module designed to provide various configurations of MLPs,
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
            - Weight initialization follows the principles of Kaiming initialization for better convergence.

        Example Usage:

            build an mlp as: torch.nn.Sequential([nn.Linear(128, 256), nn.Linear(256, 256), nn.Linear(256, 10)])

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
        use_layer_norm: bool = True,
        use_weight_norm: bool = False,
        dim_weight_norm: int = 0,
        has_bias: bool = False,
        bias: Optional[List] = None,
        zero_init_last_layer_weights: bool = False,
        dropout: Optional[float] = None,
        dampen: bool = False,
        wd: bool = False,
        gain: Optional[float] = None,
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
            "swiglu": SwiGLUModule,
            "sigmoid": torch.nn.functional.sigmoid,
        }[mlp_nonlinearity]

        if nonlinearity is None:
            gain = None
        self.gain = gain

        nonlin_const = 1.0
        if self.gain is not None:
            nonlin_const = self.gain
        else:
            if nonlinearity is not None:
                if mlp_nonlinearity == "ssp":
                    nonlin_const = normalize2mom(ShiftedSoftPlus).cst
                elif mlp_nonlinearity in ["selu", "relu", "sigmoid"]:
                    nonlin_const = torch.nn.init.calculate_gain(mlp_nonlinearity, param=None)
                elif mlp_nonlinearity == "silu":
                    nonlin_const = normalize2mom(nonlinearity).cst
                elif mlp_nonlinearity == "swiglu":
                    nonlin_const = 1.55

        if bias is not None:
            has_bias = True

        sequential_dict = OrderedDict()
        for layer_index, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            bias_condition = False if self.use_layer_norm else has_bias
            is_last_layer = layer_index == (num_layers - 1)

            if mlp_nonlinearity == "swiglu" and not is_last_layer:
                h_out = 2*h_out

            lin_layer = torch.nn.Linear(h_in, h_out, bias=bias_condition)
            modules = [(f"linear_{layer_index}", lin_layer)]

            if (nonlinearity is None) or is_last_layer:
                norm_const = 1.
            else:
                norm_const = nonlin_const
                non_lin_instance = select_nonlinearity(mlp_nonlinearity)
                modules.append((f"activation_{layer_index}", non_lin_instance))
                if dropout is not None:
                    assert 0 <= dropout < 1., f"Dropout must be a float in range [0., 1.). Got {dropout} ({type(dropout)})"
                    modules.append((f"dropout_{layer_index}", torch.nn.Dropout(dropout)))

            if zero_init_last_layer_weights and is_last_layer:
                # Scale the weights of the last layer by 1.e-1
                norm_const *= 1.e-1
            
            if self.use_layer_norm:
                modules.insert(0, (f"norm_{layer_index}", torch.nn.LayerNorm(h_in)))

            # initialize weights
            with torch.no_grad():
                # fan_in  preserves the magnitude in the forward pass.
                # fan_out preserves the magnitude in the backward pass.
                # fan_out might work better when the loss oscillates a lot.
                # Check discusison at https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization
                fan_out, fan_in = lin_layer.weight.size()
                std = norm_const / math.sqrt(fan_in)
                torch.nn.init.normal_(lin_layer.weight, mean=0, std=std)
                if lin_layer.bias is not None:
                    if is_last_layer and bias is not None:
                        lin_layer.bias.data = torch.tensor(bias).reshape(*lin_layer.bias.data.shape)
                    else:
                        torch.nn.init.zeros_(lin_layer.bias)

            # Apply weight normalization if specified, must be done after weight initialization
            if self.use_weight_norm:
                if int(torch.__version__.split('.')[0]) >= 2: from torch.nn.utils.parametrizations import weight_norm
                else: from torch.nn.utils import weight_norm
                lin_layer = weight_norm(lin_layer, name='weight', dim=self.dim_weight_norm)

            for module in modules:
                module_name, mod = module
                sequential_dict[module_name] = mod

        self.sequential = torch.nn.Sequential(sequential_dict)

        if dampen:
            add_tags_to_module(self, 'dampen')
        if wd:
            add_tags_to_module(self, '_wd')

    def forward(self, x):
        return self.sequential(x)