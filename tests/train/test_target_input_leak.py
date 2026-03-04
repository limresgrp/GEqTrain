import torch

from geqtrain.data import AtomicData
from geqtrain.train.components.inference import run_inference


class _LeakEchoModel(torch.nn.Module):
    def __init__(self, field: str):
        super().__init__()
        self.field = field

    def forward(self, data):
        return {
            "leaked": data[self.field],
            "mask": data[f"{self.field}__mask__"],
        }


class _DummyLoss:
    keys = ["target"]

    @staticmethod
    def remove_suffix(key: str) -> str:
        return key


def _make_batch(n_nodes: int = 4):
    pos = torch.randn(n_nodes, 3)
    data = AtomicData.from_points(
        pos=pos,
        r_max=4.0,
        batch=torch.zeros(n_nodes, dtype=torch.long),
    )
    data["target"] = torch.arange(n_nodes * 2, dtype=torch.float32).reshape(n_nodes, 2)
    return data


def test_target_input_leak_reinjects_removed_target_field():
    batch = _make_batch()
    model = _LeakEchoModel(field="target")
    config = {
        "target_input_leak": {
            "enabled": True,
            "fields": ["target"],
            "ratio": 0.0,
            "train_only": True,
        }
    }

    out, ref, _, _ = run_inference(
        model=model,
        data=batch,
        device=torch.device("cpu"),
        config=config,
        loss_fn=_DummyLoss(),
        is_train=True,
        current_epoch=0,
    )

    assert "leaked" in out
    assert "target__mask__" in ref
    torch.testing.assert_close(out["leaked"], torch.zeros_like(ref["target"]))
    assert not bool(out["mask"].any())


def test_target_input_leak_schedule_boundary_values():
    batch = _make_batch()
    model = _LeakEchoModel(field="target")
    config = {
        "target_input_leak": {
            "enabled": True,
            "fields": [{"source": "target", "target": "target"}],
            "schedule": {
                "start": 1.0,
                "end": 0.0,
                "start_epoch": 0,
                "end_epoch": 10,
            },
            "train_only": True,
        }
    }

    out_start, ref_start, _, _ = run_inference(
        model=model,
        data=batch,
        device=torch.device("cpu"),
        config=config,
        loss_fn=_DummyLoss(),
        is_train=True,
        current_epoch=0,
    )
    torch.testing.assert_close(out_start["leaked"], ref_start["target"])
    assert bool(out_start["mask"].all())

    out_end, ref_end, _, _ = run_inference(
        model=model,
        data=batch,
        device=torch.device("cpu"),
        config=config,
        loss_fn=_DummyLoss(),
        is_train=True,
        current_epoch=10,
    )
    torch.testing.assert_close(out_end["leaked"], torch.zeros_like(ref_end["target"]))
    assert not bool(out_end["mask"].any())


def test_target_input_leak_train_only_still_provides_zero_inputs_in_eval():
    batch = _make_batch()
    model = _LeakEchoModel(field="target")
    config = {
        "target_input_leak": {
            "enabled": True,
            "fields": ["target"],
            "ratio": 0.9,
            "train_only": True,
        }
    }

    out, ref, _, _ = run_inference(
        model=model,
        data=batch,
        device=torch.device("cpu"),
        config=config,
        loss_fn=_DummyLoss(),
        is_train=False,
        current_epoch=0,
    )

    torch.testing.assert_close(out["leaked"], torch.zeros_like(ref["target"]))
    assert not bool(out["mask"].any())
