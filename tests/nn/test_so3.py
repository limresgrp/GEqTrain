import pytest
import torch
from e3nn import o3
from e3nn.util.test import assert_equivariant, FLOAT_TOLERANCE

from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.nn.so3 import SO3_Linear, SO3_LayerNorm
from geqtrain.utils.deploy_test import assert_module_deployable


@pytest.fixture(scope="session", autouse=True, params=["float32"])
def float_tolerance(request):
    old_dtype = torch.get_default_dtype()
    dtype = {"float32": torch.float32, "float64": torch.float64}[request.param]
    torch.set_default_dtype(dtype)
    yield FLOAT_TOLERANCE[dtype]
    torch.set_default_dtype(old_dtype)


def test_so3_linear_deployable(tmp_path):
    irreps_in = o3.Irreps("2x0e+1x1o")
    irreps_out = o3.Irreps("1x0e+2x1o")
    module = SO3_Linear(irreps_in, irreps_out, internal_weights=True, bias=True)

    x = irreps_in.randn(4, -1, dtype=torch.get_default_dtype())
    assert_module_deployable(module, (x,), tmp_path=tmp_path)


def test_so3_layernorm_deployable(tmp_path):
    irreps = o3.Irreps("2x0e+2x1o")
    module = SO3_LayerNorm(irreps, bias=True, normalization="std")

    flat_features = irreps.randn(3, -1, dtype=torch.get_default_dtype())
    channel_features = reshape_irreps(irreps)(flat_features)

    # The output should preserve shape and be deployable.
    out = module(channel_features)
    assert out.shape == channel_features.shape

    assert_module_deployable(module, (channel_features,), tmp_path=tmp_path)


def test_so3_linear_equivariant(float_tolerance):
    irreps_in = o3.Irreps("1x0e+1x1o")
    irreps_out = o3.Irreps("1x0e+1x1o")
    module = SO3_Linear(irreps_in, irreps_out, internal_weights=True, bias=True)

    assert_equivariant(
        module,
        irreps_in=[irreps_in],
        irreps_out=[irreps_out],
        tolerance=float_tolerance,
    )


def test_so3_layernorm_equivariant(float_tolerance):
    irreps = o3.Irreps("2x0e+2x1o")
    module = SO3_LayerNorm(irreps, bias=True, normalization="std")
    to_channel = reshape_irreps(irreps)
    to_flat = inverse_reshape_irreps(irreps)

    def wrapped(x):
        channel = to_channel(x)
        return to_flat(module(channel))

    assert_equivariant(
        wrapped,
        irreps_in=[irreps],
        irreps_out=[irreps],
        tolerance=float_tolerance,
    )
