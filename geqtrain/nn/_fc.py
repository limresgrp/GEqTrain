""" Adapted from https://github.com/mir-group/allegro
"""
import math
import torch
from typing import List, Optional
from collections import OrderedDict
from e3nn.util.codegen import CodeGenMixin
from geqtrain.nn.nonlinearities import select_nonlinearity, select_nonlinearity_module
from geqtrain.utils import add_tags_to_module
from e3nn.util.jit import compile_mode


class SirenEnvelopeMixedActivation(torch.nn.Module):
    """
    Mixture between a standard activation and an exponentially enveloped
    sinusoid:

        lambda_base * activation(x)
        + lambda_siren * sin(omega * x + phase) * exp(-alpha * x^2)

    Default initialization recovers the wrapped activation exactly:

        lambda_siren = 0
        lambda_base = 1
        alpha ~= 0

    This is intentionally scalar and element-wise. It is appropriate for scalar
    latent features or scalar gates, but it should not be applied directly to
    equivariant tensor components unless the representation constraints are
    explicitly handled elsewhere.
    """

    def __init__(
        self,
        base_activation: torch.nn.Module,
        omega_init: float = 1.0,
        alpha_init: float = 0.0,
        phase_init: float = 0.0,
        lambda_siren_init: float = 0.0,
        lambda_base_init: float = 1.0,
        learnable_omega: bool = True,
        learnable_alpha: bool = True,
        learnable_phase: bool = True,
        learnable_lambdas: bool = True,
    ):
        super().__init__()
        self.base_activation = base_activation

        omega = torch.tensor(float(omega_init))
        phase = torch.tensor(float(phase_init))
        lambda_siren = torch.tensor(float(lambda_siren_init))
        lambda_base = torch.tensor(float(lambda_base_init))

        # softplus^{-1}(alpha). For alpha_init == 0, use a very negative value
        # so alpha starts effectively at zero while remaining non-negative.
        if alpha_init <= 0.0:
            raw_alpha = torch.tensor(-20.0)
        else:
            raw_alpha = torch.log(torch.expm1(torch.tensor(float(alpha_init))))

        if learnable_omega:
            self.omega = torch.nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)

        if learnable_alpha:
            self.raw_alpha = torch.nn.Parameter(raw_alpha)
        else:
            self.register_buffer("raw_alpha", raw_alpha)

        if learnable_phase:
            self.phase = torch.nn.Parameter(phase)
        else:
            self.register_buffer("phase", phase)

        if learnable_lambdas:
            self.lambda_siren = torch.nn.Parameter(lambda_siren)
            self.lambda_base = torch.nn.Parameter(lambda_base)
        else:
            self.register_buffer("lambda_siren", lambda_siren)
            self.register_buffer("lambda_base", lambda_base)

    def forward(self, x):
        alpha = torch.nn.functional.softplus(self.raw_alpha)
        siren_envelope = torch.sin(self.omega * x + self.phase) * torch.exp(-alpha * x.pow(2))
        return self.lambda_base * self.base_activation(x) + self.lambda_siren * siren_envelope

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

        nonlinearity, nonlinearity_gain = select_nonlinearity(mlp_nonlinearity)

        if nonlinearity is None:
            gain = None
        self.gain = gain

        nonlin_const = 1.0
        if self.gain is not None:
            nonlin_const = self.gain
        else:
            if nonlinearity_gain is not None:
                nonlin_const = nonlinearity_gain

        if bias is not None:
            has_bias = True

        sequential_dict = OrderedDict()
        for layer_index, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            bias_condition = False if (layer_index == 0 and self.use_layer_norm) else has_bias
            is_last_layer = layer_index == (num_layers - 1)

            if mlp_nonlinearity == "swiglu" and not is_last_layer:
                h_out = 2*h_out

            lin_layer = torch.nn.Linear(h_in, h_out, bias=bias_condition)
            modules = [(f"linear_{layer_index}", lin_layer)]

            if (nonlinearity is None) or is_last_layer:
                norm_const = 1.
            else:
                norm_const = nonlin_const
                non_lin_instance = select_nonlinearity_module(mlp_nonlinearity)
                modules.append((f"activation_{layer_index}", non_lin_instance))
                if dropout is not None:
                    assert 0 <= dropout < 1., f"Dropout must be a float in range [0., 1.). Got {dropout} ({type(dropout)})"
                    modules.append((f"dropout_{layer_index}", torch.nn.Dropout(dropout)))

            if layer_index == 0 and self.use_layer_norm:
                modules.insert(0, (f"norm_{layer_index}", torch.nn.LayerNorm(h_in)))

            if zero_init_last_layer_weights and is_last_layer:
                # Scale the weights of the last layer by 1.e-1
                norm_const *= 1.e-1

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
                        try:
                            lin_layer.bias.data = torch.tensor(bias).reshape(*lin_layer.bias.data.shape)
                        except:
                            # If provided bias does not match, it means that the bias is not meant for this MLP.
                            # Multiple MLPs can have common kwargs in a module, but only one is the output with the desired shape.
                            torch.nn.init.zeros_(lin_layer.bias)    
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

@compile_mode("script")
class SirenMixedScalarMLPFunction(ScalarMLPFunction):
    """
    ScalarMLPFunction specialization whose hidden activations are replaced by
    SirenEnvelopeMixedActivation modules.

    The base ScalarMLPFunction remains a generic MLP. This subclass is the
    explicit opt-in SIREN/envelope variant.

    By default the initialized network is functionally identical to the base MLP:

        lambda_siren = 0
        lambda_base = 1

    The SIREN branch can then be learned as a residual correction to the chosen
    base nonlinearity.
    """

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
        siren_omega_init: float = 1.0,
        siren_alpha_init: float = 0.0,
        siren_phase_init: float = 0.0,
        siren_lambda_init: float = 0.0,
        base_lambda_init: float = 1.0,
        siren_learnable_omega: bool = True,
        siren_learnable_alpha: bool = True,
        siren_learnable_phase: bool = True,
        siren_learnable_lambdas: bool = True,
    ):
        if mlp_nonlinearity == "swiglu":
            raise ValueError(
                "SirenMixedScalarMLPFunction is incompatible with swiglu, "
                "because swiglu changes the feature dimension while the SIREN branch preserves it."
            )

        super().__init__(
            mlp_input_dimension=mlp_input_dimension,
            mlp_latent_dimensions=mlp_latent_dimensions,
            mlp_output_dimension=mlp_output_dimension,
            mlp_nonlinearity=mlp_nonlinearity,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
            dim_weight_norm=dim_weight_norm,
            has_bias=has_bias,
            bias=bias,
            zero_init_last_layer_weights=zero_init_last_layer_weights,
            dropout=dropout,
            dampen=dampen,
            wd=wd,
            gain=gain,
        )

        self._replace_hidden_activations_with_siren_mixed(
            omega_init=siren_omega_init,
            alpha_init=siren_alpha_init,
            phase_init=siren_phase_init,
            lambda_siren_init=siren_lambda_init,
            lambda_base_init=base_lambda_init,
            learnable_omega=siren_learnable_omega,
            learnable_alpha=siren_learnable_alpha,
            learnable_phase=siren_learnable_phase,
            learnable_lambdas=siren_learnable_lambdas,
        )

    def _replace_hidden_activations_with_siren_mixed(
        self,
        omega_init: float,
        alpha_init: float,
        phase_init: float,
        lambda_siren_init: float,
        lambda_base_init: float,
        learnable_omega: bool,
        learnable_alpha: bool,
        learnable_phase: bool,
        learnable_lambdas: bool,
    ):
        for module_name, module in list(self.sequential.named_children()):
            if module_name.startswith("activation_"):
                self.sequential._modules[module_name] = SirenEnvelopeMixedActivation(
                    base_activation=module,
                    omega_init=omega_init,
                    alpha_init=alpha_init,
                    phase_init=phase_init,
                    lambda_siren_init=lambda_siren_init,
                    lambda_base_init=lambda_base_init,
                    learnable_omega=learnable_omega,
                    learnable_alpha=learnable_alpha,
                    learnable_phase=learnable_phase,
                    learnable_lambdas=learnable_lambdas,
                )
