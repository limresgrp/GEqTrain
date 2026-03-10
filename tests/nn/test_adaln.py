# tests/test_adaln_numeric.py
import torch
from geqtrain.nn.AdaLN import AdaLN
from tests.utils.numerics import check_grad_flow, check_variance_preservation

def test_adaln_shapes_and_variance():
    torch.manual_seed(0)
    dim = 32
    cond_dim = 16
    module = AdaLN(dim=dim, cond_dim=cond_dim, activation="sigmoid")

    x = torch.randn(64, dim)
    cond = torch.randn(64, cond_dim)

    out = module(x, cond)
    assert out.shape == x.shape

    # x_norm already unit-ish; scale/shift shouldn’t blow up variance
    with torch.no_grad():
        out_var = out.var().item()
        assert 1e-3 < out_var < 1e2, f"AdaLN variance insane: {out_var}"

def test_adaln_gradients():
    torch.manual_seed(0)
    dim = 32
    cond_dim = 16
    module = AdaLN(dim=dim, cond_dim=cond_dim, activation="tanh")

    x = torch.randn(32, dim, requires_grad=True)
    cond = torch.randn(32, cond_dim, requires_grad=True)

    def wrapped(inputs: torch.Tensor):
        _x, _c = inputs.split(dim, dim=1)
        return module(_x, _c)
    
    # 1) No insane variance amplification / collapse
    ratio = check_variance_preservation(wrapped, torch.cat([x, cond], dim=1), var_range=(1e-2, 1e2))

    check_grad_flow(module, (x, cond))
    assert x.grad is not None
    assert cond.grad is not None
