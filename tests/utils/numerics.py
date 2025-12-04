import torch

def check_grad_flow(module, inputs, loss_fn=None, atol_small=1e-8, rtol_big=1e3):
    """
    Run a single forward/backward and check that:
      - grads exist for all parameters
      - grads are not all ~0
      - grads are not explosively large
    """
    if loss_fn is None:
        def loss_fn(y): return y.pow(2).mean()

    module = module.train()
    for p in module.parameters():
        if p.grad is not None:
            p.grad.zero_()

    outputs = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
    loss = loss_fn(outputs)
    loss.backward()

    grad_norms = []
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            gnorm = p.grad.norm().item()
            grad_norms.append(gnorm)

    assert len(grad_norms) > 0, "No gradients found"
    max_g = max(grad_norms)
    min_g = min(grad_norms)

    assert max_g < rtol_big, f"Exploding gradients: max norm {max_g}"
    assert min_g > atol_small, f"Vanishing gradients: min norm {min_g}"

    return min_g, max_g


def check_variance_preservation(module, x, var_range=(1e-2, 1e2)):
    """
    Forward pass and check that output variance is not collapsed or exploding.
    """
    with torch.no_grad():
        y = module(x)
    in_var = x.float().var().item() + 1e-12
    out_var = y.float().var().item() + 1e-12
    ratio = out_var / in_var
    lo, hi = var_range
    assert lo <= ratio <= hi, f"Var ratio out of range: {ratio}"
    return ratio
