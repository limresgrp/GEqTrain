from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Tuple

import torch
from e3nn.util.jit import script


def _assert_close(expected: Any, actual: Any, rtol: float, atol: float) -> None:
    """Recursively compare tensors/collections for equality within tolerance."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"Expected tensor, got {type(actual)}"
        assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
        if expected.numel() == 0:
            return
        torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol)
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)), f"Expected {type(expected)}, got {type(actual)}"
        assert len(expected) == len(actual), f"Length mismatch: {len(expected)} vs {len(actual)}"
        for exp_item, act_item in zip(expected, actual):
            _assert_close(exp_item, act_item, rtol, atol)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"Expected dict, got {type(actual)}"
        assert set(expected.keys()) == set(actual.keys()), "Dictionary keys mismatch"
        for key in expected.keys():
            _assert_close(expected[key], actual[key], rtol, atol)
    else:
        assert expected == actual, f"Value mismatch: {expected} vs {actual}"


def assert_module_deployable(
    module: torch.nn.Module,
    example_args: Tuple[Any, ...],
    tmp_path: Path = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> None:
    """
    Assert that a module can be scripted, frozen, saved, and reloaded while preserving outputs.

    Args:
        module: The module to test. It should be on CPU and in eval mode.
        example_args: Example positional arguments for a forward pass.
        tmp_path: Optional directory to write the serialized module to. If omitted, a temp dir is used.
        rtol: Relative tolerance for output comparison.
        atol: Absolute tolerance for output comparison.
    """
    module = module.eval()
    example_args = tuple(example_args)

    with torch.no_grad():
        eager_out = module(*example_args)

    scripted = script(module)
    frozen = torch.jit.freeze(scripted)

    with torch.no_grad():
        scripted_out = frozen(*example_args)
    _assert_close(eager_out, scripted_out, rtol=rtol, atol=atol)

    def _save_and_load(path: Path):
        torch.jit.save(frozen, path)
        loaded = torch.jit.load(path)
        loaded.eval()
        with torch.no_grad():
            return loaded(*example_args)

    if tmp_path is None:
        with TemporaryDirectory() as d:
            loaded_out = _save_and_load(Path(d) / "deploy_test.pt")
    else:
        loaded_out = _save_and_load(Path(tmp_path) / "deploy_test.pt")

    _assert_close(eager_out, loaded_out, rtol=rtol, atol=atol)
