import torch
from e3nn.io import CartesianTensor

# e3nn canonical basis for a rank-2 Cartesian tensor:
# 1x0e (trace), 1x1e (antisymmetric), 1x2e (symmetric-traceless)
_CARTESIAN_TENSOR = CartesianTensor("ij")

def convert_spherical_to_cartesian(spherical_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts tensors from e3nn's rank-2 irreps representation to Cartesian.

    Args:
        spherical_tensor: Tensor of shape (n_batch, 9), ordered as
            ``1x0e + 1x1e + 1x2e`` in e3nn's canonical basis.

    Returns:
        A tensor of shape (n_batch, 9) representing the flattened 3x3 matrix.
    """
    if spherical_tensor.ndim != 2 or spherical_tensor.shape[1] != 9:
        raise ValueError("Input tensor must have 9 columns (1+3+5 components).")

    n_batch = spherical_tensor.shape[0]
    cartesian_tensor = _CARTESIAN_TENSOR.to_cartesian(spherical_tensor)
    return cartesian_tensor.reshape(n_batch, 9)


def convert_cartesian_to_spherical(cartesian_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts 3x3 Cartesian tensors to e3nn's rank-2 irreps representation.

    Args:
        cartesian_tensor: A tensor of shape (n_batch, 3, 3) representing the 3x3 matrices.

    Returns:
        Tensor of shape (n_batch, 9), ordered as ``1x0e + 1x1e + 1x2e``.
    """
    if cartesian_tensor.ndim != 3 or cartesian_tensor.shape[1:] != (3, 3):
        raise ValueError("Input tensor must be of shape (n_batch, 3, 3).")
    return _CARTESIAN_TENSOR.from_cartesian(cartesian_tensor)


def test_spherical_cartesian_conversion(input_cartesian_tensors: torch.Tensor):
    """
    Quick numerical diagnostics for the conversion pair.

    Args:
        input_cartesian_tensors: A batch of 3x3 Cartesian tensors to use for testing.
                                 Expected to be torch.float64.
    """
    from e3nn import o3

    print("\n--- Testing Cartesian <-> e3nn Irreps Conversion (float64) ---")
    
    tol = 1e-6

    if input_cartesian_tensors.dtype != torch.float64:
        print(f"Warning: Input tensor is not float64. Forcing to float64 for rigorous test.")
        input_cartesian_tensors = input_cartesian_tensors.to(torch.float64)

    # Step 1: Convert Cartesian to irreps and back
    spherical_components = convert_cartesian_to_spherical(input_cartesian_tensors)
    reconstructed_cartesian_flat = convert_spherical_to_cartesian(spherical_components)
    reconstructed_cartesian_tensors = reconstructed_cartesian_flat.reshape(-1, 3, 3)

    assert torch.allclose(input_cartesian_tensors, reconstructed_cartesian_tensors, atol=tol, rtol=tol), \
        "Round-trip conversion failed: Original and reconstructed Cartesian tensors do not match."
    # Step 2: Check rotation laws for l=0,1,2 channels.
    rotation = o3.rand_matrix().to(dtype=torch.float64, device=input_cartesian_tensors.device)
    rotated_cart = torch.einsum("ij,bjk,lk->bil", rotation, input_cartesian_tensors, rotation)
    rotated_spherical = convert_cartesian_to_spherical(rotated_cart)

    d1 = o3.Irrep(1, 1).D_from_matrix(rotation).to(dtype=torch.float64, device=input_cartesian_tensors.device)
    d2 = o3.Irrep(2, 1).D_from_matrix(rotation).to(dtype=torch.float64, device=input_cartesian_tensors.device)

    assert torch.allclose(rotated_spherical[:, :1], spherical_components[:, :1], atol=tol, rtol=tol)
    assert torch.allclose(
        rotated_spherical[:, 1:4],
        torch.einsum("ij,bj->bi", d1, spherical_components[:, 1:4]),
        atol=tol,
        rtol=tol,
    )
    assert torch.allclose(
        rotated_spherical[:, 4:9],
        torch.einsum("ij,bj->bi", d2, spherical_components[:, 4:9]),
        atol=tol,
        rtol=tol,
    )

    print("\n✅ Round-trip and SO(3) channel checks passed.")
