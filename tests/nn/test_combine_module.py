import pytest
import torch
from e3nn import o3

from geqtrain.data import AtomicDataDict
from geqtrain.nn import CombineModule


def _irreps_in(extra):
    base = {
        AtomicDataDict.POSITIONS_KEY: o3.Irreps("1o"),
        AtomicDataDict.EDGE_INDEX_KEY: None,
    }
    base.update(extra)
    return base


def test_combine_module_adds_same_irreps():
    irreps_in = _irreps_in(
        {
            "a": o3.Irreps("1x0e+1x2e"),
            "b": o3.Irreps("1x0e+1x2e"),
        }
    )
    module = CombineModule(fields=["a", "b"], out_field="out", irreps_in=irreps_in)

    a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    b = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])
    out = module({AtomicDataDict.POSITIONS_KEY: torch.randn(1, 3), "a": a, "b": b})

    expected = a + b
    torch.testing.assert_close(out["out"], expected)


def test_combine_module_adds_scalar_into_tensor_scalar_channel_only():
    irreps_in = _irreps_in(
        {
            "paramagnetic": o3.Irreps("1x0e+1x2e"),
            "diamagnetic": o3.Irreps("1x0e"),
        }
    )
    module = CombineModule(
        fields=["paramagnetic", "diamagnetic"],
        out_field="total",
        irreps_in=irreps_in,
    )

    paramagnetic = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ]
    )
    diamagnetic = torch.tensor([[100.0], [200.0]])
    out = module(
        {
            AtomicDataDict.POSITIONS_KEY: torch.randn(2, 3),
            "paramagnetic": paramagnetic,
            "diamagnetic": diamagnetic,
        }
    )

    expected = paramagnetic.clone()
    expected[:, :1] += diamagnetic
    torch.testing.assert_close(out["total"], expected)


def test_combine_module_raises_for_incompatible_irreps():
    irreps_in = _irreps_in(
        {
            "a": o3.Irreps("1x0e+1x2e"),
            "b": o3.Irreps("1x1o"),
        }
    )
    with pytest.raises(ValueError, match="Cannot add irreps"):
        CombineModule(fields=["a", "b"], out_field="out", irreps_in=irreps_in)
