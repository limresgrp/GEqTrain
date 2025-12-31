import torch

from geqtrain.nn.kan import KAN, KANLinear
from tests.utils.deployability import assert_module_deployable


def test_kanlinear_forward_shape():
    layer = KANLinear(
        mlp_input_dimension=4,
        mlp_output_dimension=6,
        grid_size=4,
        spline_order=2,
        use_base=True,
    )
    x = torch.randn(2, 3, 4)
    out = layer(x)
    assert out.shape == (2, 3, 6)


def test_kan_update_grid_smoke():
    model = KAN(
        mlp_input_dimension=3,
        mlp_latent_dimensions=[5],
        mlp_output_dimension=2,
        grid_size=3,
        spline_order=2,
        use_base=True,
        use_layer_norm=False,
    )
    x = torch.randn(8, 3)
    model.update_grid(x)


def test_kan_deployable(tmp_path):
    model = KAN(
        mlp_input_dimension=4,
        mlp_latent_dimensions=[5],
        mlp_output_dimension=3,
        grid_size=4,
        spline_order=2,
        use_base=True,
        use_layer_norm=True,
    )
    x = torch.randn(4, 4)
    out = model(x)
    assert out.shape == (4, 3)
    assert_module_deployable(model, (x,), tmp_path=tmp_path)
