import types

import pytest
import torch
import pytorch_warmup as warmup

from geqtrain.train.components.checkpointing import CheckpointHandler
from geqtrain.utils.config import Config


def _build_trainer(new_lr: float, warmup_period: int):
    model = torch.nn.Linear(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=new_lr)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    warmup_sched = warmup.LinearWarmup(optim, warmup_period=warmup_period)
    # warmup.LinearWarmup performs an initial dampening step
    assert warmup_sched.last_step == 0
    return types.SimpleNamespace(
        model=model,
        dist=types.SimpleNamespace(is_distributed=False),
        optim=optim,
        lr_sched=lr_sched,
        warmup_sched=warmup_sched,
        ema=None,
        config=Config.from_dict({"learning_rate": new_lr}),
        iepoch=-1,
        best_epoch=-1,
        best_metrics=float("inf"),
        cumulative_wall=0.0,
    )


def _make_restart_state(trainer, checkpoint_lr: float, old_config_lr: float):
    optim_state = trainer.optim.state_dict()
    for group in optim_state["param_groups"]:
        group["lr"] = checkpoint_lr
        if "initial_lr" in group:
            group["initial_lr"] = checkpoint_lr

    return {
        "config": Config.from_dict({"learning_rate": old_config_lr}),
        "model_state_dict": trainer.model.state_dict(),
        "optim_state_dict": optim_state,
        "sched_state_dict": trainer.lr_sched.state_dict(),
        "ema_state_dict": None,
        "iepoch": 7,
        "best_epoch": 5,
        "best_metrics": 0.123,
        "cumulative_wall": 42.0,
    }


def _make_handler(trainer):
    handler = CheckpointHandler.__new__(CheckpointHandler)
    handler.trainer = trainer
    return handler


def test_restart_warmup_targets_checkpoint_lr_when_learning_rate_not_overridden():
    trainer = _build_trainer(new_lr=1.0e-3, warmup_period=4)
    state = _make_restart_state(trainer, checkpoint_lr=2.0e-4, old_config_lr=1.0e-3)
    handler = _make_handler(trainer)

    handler.apply_state_for_restart(state)

    assert trainer.warmup_sched.lrs == pytest.approx([2.0e-4])
    # After reset, warmup starts again from step 0 -> factor = 1 / warmup_period.
    assert trainer.optim.param_groups[0]["lr"] == pytest.approx(2.0e-4 / 4.0)
    assert trainer.warmup_sched.last_step == 0


def test_restart_warmup_targets_new_config_lr_when_learning_rate_overridden():
    trainer = _build_trainer(new_lr=3.0e-3, warmup_period=6)
    state = _make_restart_state(trainer, checkpoint_lr=2.0e-4, old_config_lr=1.0e-3)
    handler = _make_handler(trainer)

    handler.apply_state_for_restart(state)

    assert trainer.warmup_sched.lrs == pytest.approx([3.0e-3])
    assert trainer.optim.param_groups[0]["lr"] == pytest.approx(3.0e-3 / 6.0)
    assert trainer.warmup_sched.last_step == 0


def test_restart_without_warmup_uses_new_config_lr_when_learning_rate_overridden():
    model = torch.nn.Linear(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=5.0e-3)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    trainer = types.SimpleNamespace(
        model=model,
        dist=types.SimpleNamespace(is_distributed=False),
        optim=optim,
        lr_sched=lr_sched,
        warmup_sched=None,
        ema=None,
        config=Config.from_dict({"learning_rate": 5.0e-3}),
        iepoch=-1,
        best_epoch=-1,
        best_metrics=float("inf"),
        cumulative_wall=0.0,
    )
    state = _make_restart_state(trainer, checkpoint_lr=1.0e-4, old_config_lr=1.0e-3)
    handler = _make_handler(trainer)

    handler.apply_state_for_restart(state)

    assert trainer.optim.param_groups[0]["lr"] == pytest.approx(5.0e-3)
