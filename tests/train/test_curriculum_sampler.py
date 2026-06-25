import numpy as np
import torch

from geqtrain.train.loss import Loss
from geqtrain.train.sampler import CurriculumBatchSampler


class _Dataset:
    def __len__(self):
        return 10


def test_named_loss_uses_display_name_but_keeps_target_key():
    loss = Loss(
        components=[
            {
                "y": [
                    1.0,
                    "L1Loss",
                    {"name": "priority_y"},
                ]
            }
        ]
    )

    pred = {"y": torch.tensor([[3.0]])}
    ref = {"y": torch.tensor([[1.0]])}
    total, contrib = loss(pred=pred, ref=ref)

    assert loss.keys == ["priority_y_0"]
    assert loss.get_target_key("priority_y_0") == "y"
    assert torch.allclose(total, torch.tensor(2.0))
    assert torch.allclose(contrib["priority_y_0"], torch.tensor(2.0))


def test_curriculum_sampler_anchor_then_priority_distribution():
    sampler = CurriculumBatchSampler(
        _Dataset(),
        batch_size=2,
        shuffle=False,
        seed=0,
        anchor_interval=5,
        alpha=1.0,
        beta_warmup_epochs=1,
        gamma=1.0,
        error_ema=0.0,
    )

    sampler.set_epoch(0)
    anchor_batches = list(iter(sampler))
    assert anchor_batches == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    for batch_id, loss_value in enumerate([1.0, 2.0, 4.0, 8.0, 16.0]):
        sampler.current_batch_id = batch_id
        sampler.update_batch_loss(loss_value)
    sampler.on_epoch_end()

    expected = np.array([1, 2, 4, 8, 16], dtype=np.float64)
    expected = expected / expected.sum()
    assert np.allclose(sampler.probabilities, expected)

    sampler.set_epoch(1)
    priority_batches = list(iter(sampler))
    assert len(priority_batches) == len(anchor_batches)
    assert not sampler.last_epoch_was_anchor
    assert sampler.ascii_histogram(bins=3)


def test_curriculum_sampler_state_roundtrip():
    sampler = CurriculumBatchSampler(_Dataset(), batch_size=3, shuffle=False)
    sampler.errors[:] = np.arange(len(sampler), dtype=np.float64) + 1.0
    sampler.on_epoch_end()
    state = sampler.state_dict()

    restored = CurriculumBatchSampler(_Dataset(), batch_size=3, shuffle=False)
    restored.load_state_dict(state)

    assert np.allclose(restored.errors, sampler.errors)
    assert np.allclose(restored.probabilities, sampler.probabilities)
