""" Adapted from https://github.com/mir-group/allegro
"""

import torch
from torch import _weight_norm, norm_except_dim
from torch import fx
from typing import List, Optional
from math import sqrt
from e3nn.math import normalize2mom
from e3nn.util.codegen import CodeGenMixin
from geqtrain.nn.nonlinearities import ShiftedSoftPlus


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
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        if nonlinearity is not None:
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

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        features = Proxy(graph.placeholder("x"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()

        self._layernorm: Optional[torch.nn.LayerNorm] = None # init to None for jit
        if self.use_norm_layer:
            setattr(self, "_layernorm", torch.nn.LayerNorm(dimensions[0]))

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):

            # make weights
            w_v = torch.empty(h_in, h_out)
            w_v.normal_()
            w_v = w_v * (
                norm_from_last / sqrt(float(h_in))
            )

            # make biases if requested
            if has_bias:
                b = torch.empty(h_out)
                b.normal_()
                params[f"_bias_{layer}"] = b
                b = Proxy(graph.get_attr(f"_bias_{layer}"))

            if self.use_weight_norm:
                w_g = norm_except_dim(w_v, 2, self.dim_weight_norm).data
                params[f"_weight_{layer}_g"] = w_g
                w_g = Proxy(graph.get_attr(f"_weight_{layer}_g"))

            # generate code
            params[f"_weight_{layer}_v"] = w_v
            w_v = Proxy(graph.get_attr(f"_weight_{layer}_v"))

            if self.use_weight_norm:
                features = torch.matmul(features, _weight_norm(w_v, w_g, self.dim_weight_norm))
                features = features + b if has_bias else features

            else:
                features = torch.matmul(features, w_v)
                features = features + b if has_bias else features

            # generate nonlinearity code
            if nonlinearity is not None and layer < num_layers - 1:
                features = nonlinearity(features)
                # add the normalization const in next layer
                norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x):
        if self.use_norm_layer:
            assert self._layernorm is not None
            x = self._layernorm(x)
        return self._forward(x)