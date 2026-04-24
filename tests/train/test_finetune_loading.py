import logging
import types

import pytest
import torch

from geqtrain.train.components.checkpointing import CheckpointHandler
from geqtrain.utils.config import Config


def _make_handler(model: torch.nn.Module, fine_tune_path, ignore_unexpected: bool = False):
    trainer = types.SimpleNamespace(
        model=model,
        dist=types.SimpleNamespace(is_distributed=False, device=torch.device("cpu")),
        config=Config.from_dict(
            {
                "fine_tune": str(fine_tune_path),
                "fine_tune_ignore_unexpected_keys": bool(ignore_unexpected),
            }
        ),
    )
    handler = CheckpointHandler.__new__(CheckpointHandler)
    handler.trainer = trainer
    return handler


def test_finetune_perfect_match_loads_all_weights(tmp_path):
    source = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    target = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))

    ckpt = tmp_path / "source.pth"
    torch.save(source.state_dict(), ckpt)

    handler = _make_handler(target, ckpt)
    handler.apply_fine_tune_weights()

    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_finetune_unexpected_keys_fail_by_default(tmp_path):
    source = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    target = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())

    ckpt = tmp_path / "source.pth"
    torch.save(source.state_dict(), ckpt)

    handler = _make_handler(target, ckpt, ignore_unexpected=False)
    with pytest.raises(ValueError, match="contains keys not present"):
        handler.apply_fine_tune_weights()


def test_finetune_unexpected_keys_can_be_ignored_with_warning(tmp_path, caplog):
    source = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    target = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())

    ckpt = tmp_path / "source.pth"
    torch.save(source.state_dict(), ckpt)

    handler = _make_handler(target, ckpt, ignore_unexpected=True)
    with caplog.at_level(logging.WARNING):
        handler.apply_fine_tune_weights()

    assert any("Ignoring" in rec.message and "unexpected fine-tune checkpoint keys" in rec.message for rec in caplog.records)


def test_finetune_missing_keys_in_new_model_warn(tmp_path, caplog):
    source = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    target = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))

    ckpt = tmp_path / "source.pth"
    torch.save(source.state_dict(), ckpt)

    handler = _make_handler(target, ckpt, ignore_unexpected=False)
    with caplog.at_level(logging.WARNING):
        handler.apply_fine_tune_weights()

    assert any("no counterpart in fine-tune checkpoint" in rec.message for rec in caplog.records)


def test_finetune_shape_mismatch_fails(tmp_path):
    source = torch.nn.Sequential(torch.nn.Linear(4, 3))
    target = torch.nn.Sequential(torch.nn.Linear(4, 5))

    ckpt = tmp_path / "source.pth"
    torch.save(source.state_dict(), ckpt)

    handler = _make_handler(target, ckpt)
    with pytest.raises(ValueError, match="incompatible shapes"):
        handler.apply_fine_tune_weights()
