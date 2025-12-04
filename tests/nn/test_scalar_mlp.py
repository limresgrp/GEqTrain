# tests/test_scalar_mlp_numeric.py
import torch
from geqtrain.nn._fc import ScalarMLPFunction
from tests.utils.numerics import check_grad_flow, check_variance_preservation


def test_scalar_mlp_init_and_grad_flow():
    torch.manual_seed(0)
    mlp = ScalarMLPFunction(
        mlp_input_dimension=64,
        mlp_latent_dimensions=[128, 128],
        mlp_output_dimension=32,
        mlp_nonlinearity="silu",
        use_layer_norm=True,
        use_weight_norm=False,
        has_bias=True,
        zero_init_last_layer_weights=False,
    )

    x = torch.randn(128, 64)
    # 1) No insane variance amplification / collapse
    ratio = check_variance_preservation(mlp, x, var_range=(1e-2, 1e2))

    # 2) Gradients flow through all layers
    check_grad_flow(mlp, x)
