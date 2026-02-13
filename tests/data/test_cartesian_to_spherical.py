import torch
from e3nn import o3

from cartesian_to_spherical import (
    convert_cartesian_to_spherical,
    convert_spherical_to_cartesian,
)


def _rotate_rank2_tensor(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjk,lk->bil", rotation, tensor, rotation)


def test_cartesian_spherical_roundtrip():
    torch.manual_seed(0)
    cartesian = torch.randn(32, 3, 3, dtype=torch.float64)

    spherical = convert_cartesian_to_spherical(cartesian)
    recovered = convert_spherical_to_cartesian(spherical).reshape(-1, 3, 3)

    assert spherical.shape == (32, 9)
    assert recovered.shape == cartesian.shape
    assert torch.allclose(recovered, cartesian, atol=1e-6, rtol=1e-6)


def test_cartesian_spherical_rotation_laws():
    torch.manual_seed(1)
    cartesian = torch.randn(64, 3, 3, dtype=torch.float64)
    rotation = o3.rand_matrix().to(dtype=torch.float64)

    spherical = convert_cartesian_to_spherical(cartesian)
    rotated_cartesian = _rotate_rank2_tensor(cartesian, rotation)
    rotated_spherical = convert_cartesian_to_spherical(rotated_cartesian)

    # l=0 is invariant.
    assert torch.allclose(rotated_spherical[:, :1], spherical[:, :1], atol=1e-6, rtol=1e-6)

    # l=1 and l=2 follow the corresponding Wigner-D matrices.
    d1 = o3.Irrep(1, 1).D_from_matrix(rotation).to(dtype=torch.float64)
    d2 = o3.Irrep(2, 1).D_from_matrix(rotation).to(dtype=torch.float64)

    l1_expected = torch.einsum("ij,bj->bi", d1, spherical[:, 1:4])
    l2_expected = torch.einsum("ij,bj->bi", d2, spherical[:, 4:9])

    assert torch.allclose(rotated_spherical[:, 1:4], l1_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rotated_spherical[:, 4:9], l2_expected, atol=1e-6, rtol=1e-6)
