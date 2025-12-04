# tests/test_radial_basis_numeric.py
import torch
import pytest
from geqtrain.nn.radial_basis import (
    BesselBasis, BesselBasisVec, GaussianBasis, PolyBasisVec
)

def test_bessel_basis_no_nans_for_small_r():
    bb = BesselBasis(r_max=5.0, num_basis=4, trainable=False)
    # include zero and tiny radii
    r = torch.tensor([0.0, 1e-6, 1e-3, 1.0])
    out = bb(r)

    assert torch.isfinite(out).all(), "BesselBasis produces non-finite values near r=0"

    # Also check gradients don’t blow up
    r = r.requires_grad_(True)
    out = bb(r)
    loss = out.pow(2).mean()
    loss.backward()
    assert torch.isfinite(r.grad).all(), "Gradients w.r.t r non-finite near r=0"


@pytest.mark.parametrize(
    "Basis, kwargs",
    [
        (BesselBasis, {"r_max": 5.0, "num_basis": 4, "trainable": False}),
        (BesselBasisVec, {"r_max": 5.0, "num_basis": 4, "accuracy": 1e-3, "trainable": False}),
        (GaussianBasis, {"r_max": 5.0, "num_basis": 4, "accuracy": 1e-3, "trainable": False}),
        (PolyBasisVec, {"r_max": 5.0, "num_basis": 4, "accuracy": 1e-3, "trainable": False}),
    ],
)
def test_basis_handles_small_r(Basis, kwargs):
    basis = Basis(**kwargs)
    r = torch.tensor([0.0, 1e-6, 1e-4, 1e-3, 1e-2])

    out = basis(r)
    assert torch.isfinite(out).all(), f"{Basis.__name__} returns non-finite values for small r"

def test_tabulated_bases_not_insanely_discontinuous():
    r_max = 5.0
    for Basis in [BesselBasisVec, GaussianBasis, PolyBasisVec]:
        basis = Basis(r_max=r_max, num_basis=4, accuracy=1e-3, trainable=False)
        r = torch.linspace(1e-3, r_max - 1e-3, steps=1000)  # avoid exact 0
        r_plus = r + 1e-3
        r_plus = torch.clamp(r_plus, max=r_max - 1e-6)

        y = basis(r)
        y2 = basis(r_plus)

        diff = (y2 - y).abs().max().item()
        # "Sanity" upper bound — adjust if too strict.
        assert diff < 1e3, f"{Basis.__name__} jumps too much between neighboring bins: {diff}"
