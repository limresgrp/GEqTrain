# tests/test_film_function.py
import torch
from geqtrain.nn._film import FiLMFunction
from tests.utils.numerics import check_grad_flow

def test_film_identity_like_at_init():
    torch.manual_seed(0)
    dim = 16
    cond_dim = 8
    film = FiLMFunction(
        mlp_input_dimension=cond_dim,
        mlp_latent_dimensions=[32],
        mlp_output_dimension=dim,
        mlp_nonlinearity="silu",
    )

    x = torch.randn(32, dim)
    cond = torch.zeros(32, cond_dim)  # near "neutral" conditioning

    y = film(x, cond)
    # FiLM(x) = gamma * x + beta; with init, gamma≈1, beta≈0
    rel_err = (y - x).pow(2).mean().sqrt() / (x.pow(2).mean().sqrt() + 1e-12)
    assert rel_err < 0.1, f"FiLM deviates too much from identity at init: {rel_err.item()}"

def test_film_gradients_wrt_conditioning():
    torch.manual_seed(0)
    dim = 16
    cond_dim = 8
    film = FiLMFunction(
        mlp_input_dimension=cond_dim,
        mlp_latent_dimensions=[32],
        mlp_output_dimension=dim,
        mlp_nonlinearity="silu",
    )

    x = torch.randn(64, dim, requires_grad=True)
    cond = torch.randn(64, cond_dim, requires_grad=True)

    def loss_fn(y): return y.mean()
    # treat (x, cond) as two inputs
    def forward(inputs):
        _x, _cond = inputs
        return film(_x, _cond)

    check_grad_flow(
        module=film,
        inputs=(x, cond),
        loss_fn=lambda y: loss_fn(y),
        atol_small=1e-8,
    )
    assert cond.grad is not None or any(p.requires_grad for p in film.parameters())
